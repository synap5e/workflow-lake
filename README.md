# workflow-lake

Crawler and ingest for a **ComfyUI workflow data lake**: crawl public
workflow-sharing sources, capture raw artifacts to a bucket, derive queryable
tables in ClickHouse.

One image, one entrypoint, a subcommand per scheduled job:

```sh
lake init-db          # apply migrations/
lake discover-tip     # page the source's newest listing until it meets known ground
lake discover-backlog # drain partitions (the source's cursor caps, so history is partitioned)
lake fetch            # lease a batch, capture bytes, write blobs + manifest
lake ingest           # manifests + blobs -> ClickHouse
lake gc               # expire byte prefixes past their retention window
lake unlatch <host>   # release a politeness latch (humans only)
lake status
```

Deployed on the `medina` cluster as five CronJobs —
[`apps/workflow-lake/`](https://github.com/synap5e/gcp-k8s/tree/main/apps/workflow-lake)
in `synap5e/gcp-k8s`.

## Where the design came from

This is the production form of a measurement project. The experiments that
produced every policy in here — Range cutoffs, channel ordering, prefilters,
dedup levels — live in `comfy/workflow-lake-experiments` along with the numbers:

| Doc | What it holds |
|---|---|
| `FINDINGS.md` | Measured results per experiment, with the caveats |
| `DESIGN.md` | The pipeline design the numbers imply |
| `SCALING.md` | What crawling the maximum costs and yields |
| `BUILD.md` | Build order and the Postgres frontier schema |
| `DEPLOY-MEDINA.md` | How this lands on the cluster |

That repo keeps the probes and the evidence; this one is what runs. Git history
for these files up to the split is there.

## Why it is shaped like this

The load-bearing decisions, each measured rather than assumed:

- **Capture is irreversible; everything else is a re-run.** The bucket is the
  system of record. ClickHouse and Postgres can both be dropped and rebuilt.
  That is why `fetch` writes verbatim `source_meta` and retains byte prefixes.
- **Range fetching is container-dispatched.** 128 KB of head for PNG; a 128 KB
  *tail* for video, because ffmpeg parks `moov` after `mdat` and 47 of 49
  sampled video workflows were reachable only from the end of the file. Head-only
  finds 2 of them.
- **No single channel is sufficient.** The source's API and its served bytes each
  find workflows the other misses; the union is 88.5% against 72–77% alone.
- **Negatives are nearly free** (p50 0.6 KB), so prefilters have to justify
  themselves on request count, not bytes.
- **The latch is sticky.** Three consecutive 401/403/429 from a host stops the
  crawler until a human clears it. An in-process counter would die with the pod
  and let a refusing source be re-hit every five minutes forever.

## Development

```sh
uv sync --extra dev
uv run pytest              # 40 tests, no network, no database
uv run ruff check . && uv run ruff format --check .
./scripts/e2e.sh           # throwaway Postgres + ClickHouse, real source, full loop
```

`scripts/e2e.sh` is the real check: it runs the same commands the CronJobs run,
from an empty database through to queryable rows, and asserts the latch both
stops and releases a run.

## Configuration

| Variable | Purpose |
|---|---|
| `LAKE_DATABASE_URL` | Frontier Postgres DSN |
| `LAKE_BLOB_ROOT` | `gs://…` (native GCS), `s3://…` (S3 or GCS S3-compat), or a local path |
| `LAKE_CLICKHOUSE_URL` | Or compose from `CLICKHOUSE_HOST` / `CLICKHOUSE_PORT_HTTP` |
| `CLICKHOUSE_USER` / `_PASSWORD` / `_DATABASE` | Sent as headers, never in the query string |
| `LAKE_API_RPS` / `LAKE_CDN_RPS` | Per-host rate budgets |
| `LAKE_USER_AGENT` | Self-identifying, deliberately not a browser string |
| `LAKE_IP_FAMILY` | `auto` \| `4` \| `6`. Set `4` where IPv6 egress is black-holed: httpx has no Happy Eyeballs and stalls ~40s per request on a dead AAAA route |
| `LAKE_KEEP_PREFIX_DAYS` | Byte-prefix retention (90) |

Secrets can be given as `$NAME_FILE` pointing at a mounted file instead of
`$NAME`, which keeps them off command lines and out of `env` dumps.

## Politeness

Not decoration — the crawler talks to third parties that owe us nothing.

- Per-host token bucket, one process per host.
- Exponential backoff honouring `Retry-After`.
- A self-identifying User-Agent. Never a browser string to get past a block.
- Range fetching: 95.7% less bandwidth than downloading the files.
- The sticky latch above. A latched run exits **2**, distinct from a crash.

Public content only. If a source pushes back, that is an answer.
