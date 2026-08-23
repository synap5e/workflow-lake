"""Ingest: manifests + blobs -> ClickHouse.

A pure function of the bucket. It touches no network, holds no state, and can be
dropped and re-run at any time — which is the property the whole raw-layer design
exists to buy. When a derived column changes, this re-runs; nothing is recrawled.

Rows land in `ReplacingMergeTree` tables keyed on `ingest_version`, so a
re-derivation is an insert rather than a migration.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from collections.abc import Iterator
from importlib.resources import files
from typing import Any

import httpx

from lake.config import Config
from lake.derive import Reference, derive
from lake.storage import Store, open_store, read_manifest

INGEST_VERSION = 1

TABLES = (
    "raw_artifacts",
    "derived_workflows",
    "derived_artifact_workflows",
    "derived_workflow_nodes",
    "derived_workflow_bindings",
)


class ClickHouse:
    """ClickHouse over the HTTP interface (8123).

    HTTP rather than the native protocol on purpose: it needs only the `httpx`
    this project already depends on, so the image carries no `clickhouse-client`
    binary and no ClickHouse apt source. Our inserts are single-digit MB batches
    of `JSONEachRow`, where native's bulk advantage buys nothing.

    Config is `$LAKE_CLICKHOUSE_URL` plus `$LAKE_CLICKHOUSE_USER` /
    `$LAKE_CLICKHOUSE_PASSWORD` / `$LAKE_CLICKHOUSE_DATABASE`. The password is
    sent as a header, never in the query string, so it stays out of ClickHouse's
    own `system.query_log`.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.url = (url or _clickhouse_url()).rstrip("/")
        self.database = database or _env(
            "LAKE_CLICKHOUSE_DATABASE", "CLICKHOUSE_DATABASE", default="workflow_lake"
        )
        headers = {}
        user = user or _env("LAKE_CLICKHOUSE_USER", "CLICKHOUSE_USER")
        if password is None:
            password = _env("LAKE_CLICKHOUSE_PASSWORD", "CLICKHOUSE_PASSWORD")
        if user:
            headers["X-ClickHouse-User"] = user
        if password:
            headers["X-ClickHouse-Key"] = password
        self.client = httpx.Client(timeout=timeout, headers=headers)

    def _post(self, query: str, body: str | bytes = b"") -> str:
        resp = self.client.post(
            self.url,
            params={"query": query, "database": self.database},
            content=body.encode() if isinstance(body, str) else body,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"clickhouse {resp.status_code}: {resp.text.strip()[:500]}")
        return resp.text

    def execute(self, sql: str) -> str:
        """Run DDL or a statement batch.

        The HTTP interface takes one statement per request, so a multi-statement
        file is split here. Naive `;` splitting is enough for our schema, which
        has no semicolons inside string literals — asserted by the schema test.
        """
        out = []
        for statement in filter(None, (part.strip() for part in _strip_sql(sql).split(";"))):
            out.append(self._post(statement))
        return "".join(out)

    def insert_jsonl(self, table: str, lines: Iterator[str]) -> int:
        body = "".join(lines)
        if not body:
            return 0
        self._post(f"INSERT INTO {table} FORMAT JSONEachRow", body)
        return body.count("\n")

    def query(self, sql: str) -> str:
        return self._post(sql)

    def close(self) -> None:
        self.client.close()


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _clickhouse_url() -> str:
    """Explicit URL, or compose one from the platform's credential shape.

    medina hands out ClickHouse credentials as a Secret you `envFrom`, in the
    shape `CLICKHOUSE_HOST` / `CLICKHOUSE_PORT_HTTP` / `_DATABASE` / `_USER` /
    `_PASSWORD`. Reading that directly means the manifest mounts the platform's
    secret and sets nothing else — no duplicated values to drift apart.
    """
    explicit = os.environ.get("LAKE_CLICKHOUSE_URL")
    if explicit:
        return explicit
    host = os.environ.get("CLICKHOUSE_HOST")
    if host:
        port = os.environ.get("CLICKHOUSE_PORT_HTTP", "8123")
        return f"http://{host}:{port}"
    return "http://localhost:8123"


def _strip_sql(sql: str) -> str:
    """Drop `--` comments so they cannot swallow the statement separator."""
    return "\n".join(line.split("--", 1)[0] for line in sql.splitlines())


def schema_sql() -> str:
    """The ClickHouse DDL, from the package.

    Packaged for the same reason as the Postgres migrations: an installed
    console script resolves imports from site-packages, so a copy at /app is
    invisible to it.
    """
    text = files("lake").joinpath("schema.sql").read_text()
    if "CREATE TABLE" not in text:
        raise RuntimeError("packaged schema.sql carries no DDL")
    return text


def ensure_schema(ch: ClickHouse) -> None:
    """Create the ClickHouse tables if they are absent.

    Nothing else does this. `lake migrate` handles Postgres; ClickHouse had no
    equivalent, so the first production ingest ran against a database with zero
    tables. It reported success only because it had no manifests to insert —
    had there been any, every INSERT would have failed instead.

    The DDL is all `CREATE TABLE IF NOT EXISTS`, so this is idempotent and cheap
    to run at the head of every ingest.

    It also reconciles the ENGINE, because `IF NOT EXISTS` silently accepts a
    table that already exists with the wrong one. Three tables shipped as plain
    `MergeTree` and duplicated every row on each re-ingest — 5-6x before anyone
    looked — and no amount of corrected DDL would have repaired them, because
    the create was being skipped. Dropping and recreating is safe here in a way
    it would not be for a primary store: every one of these tables is derived
    from manifests in the bucket, which is the system of record, so a rebuild
    costs a re-read and loses nothing.
    """
    ch.execute(schema_sql())

    want = expected_engines()
    have = dict(
        line.split("\t", 1)
        for line in ch.query(
            f"SELECT name, engine FROM system.tables WHERE database = '{ch.database}' FORMAT TSV"
        ).splitlines()
        if "\t" in line
    )
    stale = {t: have[t] for t, engine in want.items() if t in have and have[t] != engine}
    for table, actual in stale.items():
        print(
            f"  schema: {table} is {actual}, want {want[table]} — "
            f"dropping and rebuilding it from the bucket",
            file=sys.stderr,
        )
        ch.execute(f"DROP TABLE IF EXISTS {table}")
    if stale:
        ch.execute(schema_sql())


def expected_engines() -> dict[str, str]:
    """Table -> engine name, parsed from the shipped DDL.

    Parsed rather than listed here, so this check cannot drift away from
    `schema.sql` the way the schema itself drifted away from the deployment.
    """
    out: dict[str, str] = {}
    table = None
    for raw in schema_sql().splitlines():
        line = raw.strip()
        if line.upper().startswith("CREATE TABLE IF NOT EXISTS"):
            table = line.split()[-1].rstrip("(").strip()
        elif line.upper().startswith("ENGINE =") and table:
            out[table] = line.split("=", 1)[1].strip().split("(")[0].strip()
            table = None
    return out


def blob_json(store: Store, sha: str, kind: str) -> Any | None:
    from lake.storage import blob_key

    key = blob_key(bytes.fromhex(sha), kind)
    try:
        return json.loads(gzip.decompress(store.get(key)))
    except Exception:
        return None


def iso(value: Any) -> str | None:
    if not value or not isinstance(value, str):
        return None
    text = value.replace("T", " ").replace("Z", "")
    return text[:19] if len(text) >= 19 else None


def transform(records: list[dict], store: Store, ref: Reference) -> dict[str, list[str]]:
    """Manifest records -> one JSONEachRow batch per table."""
    out: dict[str, list[str]] = {t: [] for t in TABLES}
    seen: set[str] = set()

    for rec in records:
        sha = rec.get("payload_sha256")
        workflow_id = ""

        if sha and rec.get("payload_kind") in ("workflow", "prompt"):
            obj = blob_json(store, sha, rec["payload_kind"])
            if obj is not None:
                d = derive(obj, ref)
                if d.format != "unknown":
                    workflow_id = d.exact_hash
                    if workflow_id not in seen:
                        seen.add(workflow_id)
                        out["derived_workflows"].append(
                            json.dumps(
                                {
                                    "workflow_id": workflow_id,
                                    "canonical_hash": d.canonical_hash,
                                    "structural_hash": d.structural_hash,
                                    "format": d.format,
                                    "node_count": d.node_count,
                                    "link_count": d.link_count,
                                    "group_count": d.group_count,
                                    "muted_or_bypassed": d.muted_or_bypassed,
                                    "model_edges": d.model_edges,
                                    "hidden_prompt_nodes": d.hidden_prompt_nodes,
                                    "frontend_version": d.frontend_version or "",
                                    "graph_version": str(d.graph_version or ""),
                                    "stamped_nodes": d.stamped_nodes,
                                    "cnr_nodes": d.cnr_nodes,
                                    "aux_nodes": d.aux_nodes,
                                    "pack_set": d.pack_set,
                                    "pack_set_size": len(d.pack_set),
                                    "first_source": rec["source"],
                                    "first_seen": iso(rec.get("published_at"))
                                    or iso(rec["crawled_at"]),
                                    "ingest_version": INGEST_VERSION,
                                },
                                default=str,
                            )
                            + "\n"
                        )
                        for node in d.nodes:
                            out["derived_workflow_nodes"].append(
                                json.dumps(
                                    {
                                        "workflow_id": workflow_id,
                                        "node_id": node.node_id,
                                        "class_type": node.class_type,
                                        "is_core": node.is_core,
                                        "cnr_id": node.cnr_id or "",
                                        "ver": node.ver or "",
                                        "aux_id": node.aux_id or "",
                                        "mode": node.mode if node.mode is not None else 0,
                                        "packs": node.packs,
                                        "pack_ambiguous": node.pack_ambiguous,
                                        "pack_unknown": node.pack_unknown,
                                    },
                                    default=str,
                                )
                                + "\n"
                            )
                        for b in d.bindings:
                            out["derived_workflow_bindings"].append(
                                json.dumps(
                                    {
                                        "workflow_id": workflow_id,
                                        "node_id": b.get("node_id", ""),
                                        "class_type": b["class_type"],
                                        "input": b["input"],
                                        "binding": b["binding"],
                                        "evidence": b["source"],
                                    }
                                )
                                + "\n"
                            )
                    out["derived_artifact_workflows"].append(
                        json.dumps(
                            {
                                "workflow_id": workflow_id,
                                "source": rec["source"],
                                "source_artifact_id": rec["source_artifact_id"],
                                "channel": rec["channel"],
                                "payload_kind": rec["payload_kind"],
                                "author": rec.get("author") or "",
                                "published_at": iso(rec.get("published_at")),
                            },
                            default=str,
                        )
                        + "\n"
                    )

        out["raw_artifacts"].append(
            json.dumps(
                {
                    "capture_id": rec["capture_id"],
                    "crawled_at": iso(rec["crawled_at"]),
                    "crawler_version": rec.get("crawler_version") or "",
                    "source": rec["source"],
                    "channel": rec["channel"],
                    "source_artifact_id": rec["source_artifact_id"],
                    "parent_url": rec.get("parent_url") or "",
                    "fetch_url": rec.get("fetch_url") or "",
                    "author": rec.get("author") or "",
                    "author_id": str(rec.get("author_id") or ""),
                    "title": rec.get("title") or "",
                    "description": rec.get("description") or "",
                    "tags": rec.get("tags") or [],
                    "published_at": iso(rec.get("published_at")),
                    "stats": rec.get("stats") or {},
                    "source_meta": json.dumps(rec.get("source_meta") or {}, default=str),
                    "http_status": int(rec.get("http_status") or 0),
                    "content_type": rec.get("content_type") or "",
                    "container": rec.get("container") or "",
                    "total_size": rec.get("total_size"),
                    "bytes_fetched": int(rec.get("bytes_fetched") or 0),
                    "needed_bytes": rec.get("needed_bytes"),
                    "windows": [int(w) for w in (rec.get("windows") or [])],
                    "from_tail": bool(rec.get("from_tail")),
                    "payload_kind": rec.get("payload_kind") or "none",
                    "payload_ref": sha or "",
                    "payload_bytes": int(rec.get("payload_bytes") or 0),
                    "workflow_id": workflow_id,
                },
                default=str,
            )
            + "\n"
        )
    return out


WATERMARK_STREAM = "ingest"


def run(
    cfg: Config,
    *,
    since: str | None = None,
    clickhouse: ClickHouse | None = None,
    ref: Reference | None = None,
    limit_manifests: int | None = None,
    frontier=None,
) -> dict:
    """Load manifests under `raw/` that this has not already loaded.

    The watermark is the manifest key of the last one consumed, and it is
    PERSISTED. Before, `run` computed it, returned it in the stats, and nobody
    wrote it down — so every hourly tick re-read every manifest from the
    beginning of time and re-inserted every row. The tables that could absorb
    that were `ReplacingMergeTree`; the three that could not reached 5-6x
    duplication, and the cost of a run grew with the size of the bucket rather
    than with the work done.

    Both halves of the fix matter and neither is sufficient. The watermark stops
    the pointless re-reading. Idempotent engines mean a lost watermark, a forced
    rebuild, or a re-run of a partially-failed tick stays correct instead of
    silently multiplying rows.
    """
    store = open_store(cfg.blob_root)
    ch = clickhouse or ClickHouse()
    ensure_schema(ch)
    ref = ref or Reference.load()

    if since is None and frontier is not None:
        state = frontier.get_cursor("_ingest", WATERMARK_STREAM) or {}
        since = state.get("cursor")

    keys = [k for k in store.list("raw/") if k.endswith(".jsonl.gz")]
    if since:
        keys = [k for k in keys if k > since]
    if limit_manifests:
        keys = keys[:limit_manifests]

    run_id = frontier.start_run("ingest", "_ingest") if frontier is not None else None
    totals = dict.fromkeys(TABLES, 0)
    records = 0
    consumed = since
    try:
        for key in keys:
            batch = transform(read_manifest(store, key), store, ref)
            records += len(batch["raw_artifacts"])
            for table, lines in batch.items():
                totals[table] += ch.insert_jsonl(table, iter(lines))
            # Advanced per manifest, not once at the end: a run killed halfway
            # then resumes after the last manifest it actually finished.
            consumed = key
            if frontier is not None:
                frontier.set_cursor("_ingest", WATERMARK_STREAM, consumed, None)
            print(f"  {key}: {len(batch['raw_artifacts'])} records", file=sys.stderr)
    finally:
        if run_id is not None:
            frontier.finish_run(run_id, items=records, manifest_key=consumed)

    return {
        "manifests": len(keys),
        "records": records,
        "rows": totals,
        "watermark": consumed,
    }
