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

# Civitai's tRPC listing is auth-gated, so without a key the crawl half cannot
# run at all. That used to abort the whole script at discover-tip, which meant
# the ingest assertions below — the ones that catch duplication — never ran
# anywhere except a machine holding a credential. CI does not hold one either.
# Offline mode seeds a manifest directly so the ingest half is always covered.
if [[ -n "${CIVITAI_API_KEY:-}" ]]; then
  echo "== discover-tip =="       && uv run --quiet lake discover-tip --max-pages 2
  echo "== discover-backlog =="   && uv run --quiet lake discover-backlog --partitions 2
  echo "== fetch =="              && uv run --quiet lake fetch --batch 15
else
  echo "== no CIVITAI_API_KEY: seeding a manifest instead of crawling =="
  uv run --quiet python - <<'SEED'
import json, os, uuid
from lake.config import Config
from lake.db import Frontier
from lake.storage import open_store, ManifestWriter, BlobWriter
from tests.fixtures import WORKFLOW_JSON

cfg = Config.from_env()
store = open_store(cfg.blob_root)
frontier = Frontier(cfg.database_url)
blobs = BlobWriter(store, frontier, cfg.keep_prefix_days)
mw = ManifestWriter(store, "civitai", str(uuid.uuid4()))
sha, _key, size = blobs.put(WORKFLOW_JSON.encode(), "workflow")
for i in range(3):
    mw.write({
        "capture_id": f"cap-{i}", "crawled_at": "2026-08-23T00:00:00",
        "crawler_version": "e2e", "source": "civitai", "channel": "bytes_head",
        "source_artifact_id": str(1000 + i), "parent_url": "", "fetch_url": "",
        "author": "", "author_id": "", "title": "", "description": "", "tags": [],
        "published_at": None, "stats": {}, "source_meta": "{}",
        "http_status": 206, "content_type": "image/png", "container": "png",
        "total_size": 1000, "bytes_fetched": 500, "needed_bytes": 400,
        "windows": [131072], "from_tail": False,
        "payload_kind": "workflow", "payload_sha256": sha, "payload_bytes": size,
    })
print(" seeded", mw.commit())
frontier.close()
SEED
fi
# No explicit schema step: `lake ingest` creates its own tables now. The old
# harness applied schema.sql by hand, which is exactly why nothing in the
# deployment did.
echo "== ingest =="             && uv run --quiet lake ingest
echo "== status =="             && uv run --quiet lake status

echo "== queryable? =="
ch_q() { curl -sf "http://127.0.0.1:18123/?database=workflow_lake" --data-binary "$1"; }
ch_q "SELECT count(), countIf(workflow_id != '') FROM raw_artifacts FORMAT TSV"
ch_q "SELECT format, count(), sum(node_count) FROM derived_workflows GROUP BY format FORMAT TSV"
ch_q "SELECT class_type, count() n FROM derived_workflow_nodes GROUP BY class_type ORDER BY n DESC LIMIT 3 FORMAT TSV"

# This check used to look at derived_workflows alone — the table that was
# already ReplacingMergeTree — and so passed while three plain-MergeTree tables
# duplicated every row on each re-ingest, reaching 5-6x in production. Forcing
# --since '' replays the whole bucket, which is exactly what the missing
# watermark used to do on every tick.
echo "== re-ingest of the SAME manifests adds no logical rows, every table =="
TABLES="raw_artifacts derived_workflows derived_artifact_workflows derived_workflow_nodes derived_workflow_bindings"
declare -A before_rows
for t in $TABLES; do before_rows[$t]=$(ch_q "SELECT count() FROM $t FINAL FORMAT TSV"); done
uv run --quiet lake ingest --since '' >/dev/null
for t in $TABLES; do
  after=$(ch_q "SELECT count() FROM $t FINAL FORMAT TSV")
  if [[ "${before_rows[$t]}" != "$after" ]]; then
    echo "  FAIL: $t went ${before_rows[$t]} -> $after on re-ingest"; exit 1
  fi
done
# ReplacingMergeTree collapses on merge, so physical > logical is normal until
# one happens. Forcing it proves the duplicates are actually reclaimed rather
# than merely hidden behind FINAL — which is what the storage projection needs.
for t in $TABLES; do
  ch_q "OPTIMIZE TABLE $t FINAL" >/dev/null
  n=$(ch_q "SELECT count() FROM $t FORMAT TSV")
  u=$(ch_q "SELECT count() FROM $t FINAL FORMAT TSV")
  [[ "$n" == "$u" ]] || { echo "  FAIL: $t keeps $n physical rows for $u logical after merge"; exit 1; }
  [[ "${before_rows[$t]}" -gt 0 ]] || { echo "  FAIL: $t is empty — the check proves nothing"; exit 1; }
  echo "  $t: $n rows, deduplicated on merge"
done

echo "== the watermark is persisted, so a second tick reads nothing =="
again=$(uv run --quiet lake ingest)
echo "$again" | grep -q '"manifests": 0' \
  && echo "  second ingest consumed 0 manifests — correct" \
  || { echo "  FAIL: re-read manifests it had already consumed:"; echo "$again"; exit 1; }

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
