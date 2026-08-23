#!/usr/bin/env bash
# End-to-end smoke test against a throwaway Postgres + ClickHouse.
#
# Runs the same five commands the CronJobs run, in order, against the real
# source. Small batches: this is a wiring check, not a crawl.
#
#   ./scripts/e2e.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${LAKE_E2E_DIR:-/tmp/claude/lake-e2e}"
export PGDATA="$WORK/pg"
export PGHOST="$WORK/sock"
export LAKE_DATABASE_URL="postgresql://postgres@/workflow_lake?host=$PGHOST"
export LAKE_BLOB_ROOT="$WORK/blobs"
export LAKE_CLICKHOUSE_URL="http://127.0.0.1:18123"
export LAKE_CLICKHOUSE_DATABASE="workflow_lake"

pg() { nix shell nixpkgs#postgresql --command "$@"; }
cleanup() {
  pg pg_ctl -D "$PGDATA" stop >/dev/null 2>&1 || true
  [[ -n "${CH_PID:-}" ]] && kill "$CH_PID" 2>/dev/null || true
}
trap cleanup EXIT

rm -rf "$WORK"
mkdir -p "$PGDATA" "$PGHOST" "$WORK/ch"

echo "== starting throwaway postgres =="
pg initdb -D "$PGDATA" -U postgres --auth=trust >/dev/null
pg pg_ctl -D "$PGDATA" -o "-k $PGHOST -c listen_addresses=''" -l "$WORK/pg.log" start >/dev/null
pg createdb -h "$PGHOST" -U postgres workflow_lake

echo "== starting throwaway clickhouse (http) =="
# Two traps here. ClickHouse takes config overrides only AFTER a bare `--`,
# and `nix shell --command` swallows that `--` itself, so the whole invocation
# goes through `bash -c` as one string.
nix shell nixpkgs#clickhouse --command bash -c \
  "clickhouse server -- --path=$WORK/ch/ --http_port=18123 --tcp_port=0 \
   --mysql_port=0 --listen_host=127.0.0.1" >"$WORK/ch.log" 2>&1 &
CH_PID=$!
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:18123/ping" >/dev/null 2>&1 && break
  sleep 1
done
# Bounded on purpose: a server that never starts must fail the run, not hang it.
curl -sf "http://127.0.0.1:18123/ping" >/dev/null 2>&1 || {
  echo "clickhouse did not come up:"; tail -20 "$WORK/ch.log"; exit 1; }
curl -sf "http://127.0.0.1:18123/" \
  --data-binary "CREATE DATABASE IF NOT EXISTS workflow_lake" >/dev/null

cd "$ROOT"
# The race that broke production: five pods hitting a VIRGIN database at once,
# where the tracking table itself is the contended object. The earlier version
# of this check ran after the table already existed and proved nothing.
echo "== migrate races from an EMPTY database (5 pods, cold start) =="
race_pids=()
for _ in 1 2 3 4 5; do
  uv run --quiet lake migrate >"$WORK/migrate-$RANDOM.log" 2>&1 &
  race_pids+=($!)
done
race_failed=0
for pid in "${race_pids[@]}"; do wait "$pid" || race_failed=1; done
if [[ $race_failed -ne 0 ]]; then
  echo "  FAIL: a cold-start concurrent migrate errored:"; cat "$WORK"/migrate-*.log; exit 1
fi
# Exactly one should have applied them; the rest find nothing to do.
echo "  5 cold-start migrations, no races"

echo "== schema is actually there =="
nix shell nixpkgs#postgresql --command psql -h "$PGHOST" -U postgres -d workflow_lake -tAc \
  "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'" \
  | xargs -I{} sh -c 'echo "  {} tables"; [ {} -ge 6 ] || { echo "  FAIL: schema not applied"; exit 1; }'
nix shell nixpkgs#postgresql --command psql -h "$PGHOST" -U postgres -d workflow_lake -tAc \
  "SELECT count(*) FROM schema_migrations" \
  | xargs -I{} sh -c 'echo "  {} migrations recorded"; [ {} -ge 2 ] || { echo "  FAIL: none recorded"; exit 1; }'

echo "== migrate =="            && uv run --quiet lake migrate
echo "== migrate is idempotent ==" && uv run --quiet lake migrate
echo "== discover-tip =="       && uv run --quiet lake discover-tip --max-pages 2
echo "== discover-backlog =="   && uv run --quiet lake discover-backlog --partitions 2
echo "== fetch =="              && uv run --quiet lake fetch --batch 15
echo "== clickhouse schema =="  && uv run --quiet python -c "
from pipeline.ingest import ClickHouse
ch = ClickHouse(); ch.execute(open('schema.sql').read()); print('applied')"
echo "== ingest =="             && uv run --quiet lake ingest
echo "== status =="             && uv run --quiet lake status

echo "== queryable? =="
ch_q() { curl -sf "http://127.0.0.1:18123/?database=workflow_lake" --data-binary "$1"; }
ch_q "SELECT count(), countIf(workflow_id != '') FROM raw_artifacts FORMAT TSV"
ch_q "SELECT format, count(), sum(node_count) FROM derived_workflows GROUP BY format FORMAT TSV"
ch_q "SELECT class_type, count() n FROM derived_workflow_nodes GROUP BY class_type ORDER BY n DESC LIMIT 3 FORMAT TSV"

echo "== re-ingest is idempotent (ReplacingMergeTree) =="
uv run --quiet lake ingest >/dev/null
ch_q "SELECT count(), uniqExact(workflow_id) FROM derived_workflows FINAL FORMAT TSV"

echo "== migrate --check reports nothing pending =="
uv run --quiet lake migrate --check

echo "== the latch stops the next run =="
uv run --quiet python -c "
from lake.config import Config
from lake.db import Frontier
f = Frontier(Config.from_env().database_url)
for _ in range(3):
    f.record_refusal('civitai.com', 403, 3)
print('latched:', [r['host'] for r in f.active_latches()])
f.close()"
set +e; uv run --quiet lake fetch --batch 5; code=$?; set -e
[[ $code -eq 2 ]] && echo "  fetch refused to start (exit 2) — correct" || { echo "  FAIL: expected exit 2, got $code"; exit 1; }
uv run --quiet lake unlatch civitai.com --by e2e
set +e; uv run --quiet lake fetch --batch 2 >/dev/null; code=$?; set -e
[[ $code -eq 0 ]] && echo "  fetch runs again after unlatch — correct" || { echo "  FAIL: exit $code"; exit 1; }

echo "E2E OK"
