"""Postgres frontier: the work queue, discovery cursors, partitions, blob index.

The whole concurrency model is `SELECT ... FOR UPDATE SKIP LOCKED` plus a lease
column. That is what makes CronJobs safe without a queue broker: a job claims a
batch, sets `lease_until`, works, and commits. A pod that dies mid-batch simply
lets its lease expire and the rows return to the pool. Nothing needs to notice
the crash.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


@dataclass
class QueueItem:
    source: str
    artifact_id: str
    post_id: str | None
    author_id: str | None
    published_at: dt.datetime | None
    prefilter: dict[str, Any]


class Frontier:
    def __init__(self, dsn: str) -> None:
        self.conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=False)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[psycopg.Cursor]:
        with self.conn.cursor() as cur:
            try:
                yield cur
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    # --- schema ----------------------------------------------------------

    def migrate(self, sql: str) -> None:
        with self.tx() as cur:
            cur.execute(sql)

    # --- queue -----------------------------------------------------------

    def enqueue(self, items: list[QueueItem], *, priority: int) -> int:
        """Insert discovered artifacts. Already-known ids are left untouched.

        `ON CONFLICT DO NOTHING` is what makes discovery idempotent: re-running
        a tip sweep over the same window costs one statement and changes no
        state, so a CronJob that fires twice is harmless.
        """
        if not items:
            return 0
        with self.tx() as cur:
            cur.executemany(
                """
                INSERT INTO crawl_queue
                    (source, artifact_id, post_id, author_id, published_at,
                     priority, prefilter)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source, artifact_id) DO NOTHING
                """,
                [
                    (
                        i.source,
                        i.artifact_id,
                        i.post_id,
                        i.author_id,
                        i.published_at,
                        priority,
                        Jsonb(i.prefilter),
                    )
                    for i in items
                ],
            )
            return cur.rowcount

    def claim(self, source: str, limit: int, lease_seconds: int = 900) -> list[dict]:
        """Lease a batch of pending work, reclaiming anything whose lease expired."""
        with self.tx() as cur:
            cur.execute(
                """
                WITH claimed AS (
                    SELECT source, artifact_id
                      FROM crawl_queue
                     WHERE source = %s
                       AND (state = 'pending'
                            OR (state = 'leased' AND lease_until < now()))
                     ORDER BY priority DESC, published_at DESC NULLS LAST
                     LIMIT %s
                       FOR UPDATE SKIP LOCKED
                )
                UPDATE crawl_queue q
                   SET state = 'leased',
                       lease_until = now() + make_interval(secs => %s),
                       attempts = q.attempts + 1,
                       updated_at = now()
                  FROM claimed c
                 WHERE q.source = c.source AND q.artifact_id = c.artifact_id
             RETURNING q.source, q.artifact_id, q.post_id, q.author_id,
                       q.published_at, q.prefilter, q.attempts
                """,
                (source, limit, lease_seconds),
            )
            return cur.fetchall()

    def finish(self, source: str, artifact_id: str, state: str, error: str | None = None) -> None:
        with self.tx() as cur:
            cur.execute(
                """
                UPDATE crawl_queue
                   SET state = %s, last_error = %s, lease_until = NULL,
                       updated_at = now()
                 WHERE source = %s AND artifact_id = %s
                """,
                (state, (error or "")[:500] or None, source, artifact_id),
            )

    def release(self, source: str, artifact_ids: list[str]) -> int:
        """Hand leases back immediately instead of waiting for them to expire.

        A job that is killed mid-batch otherwise holds its whole lease window —
        15 minutes during which the next tick finds nothing claimable and exits
        having done nothing. With a batch that cannot finish inside the job
        deadline, that is a livelock: lease everything, do a little, die, idle,
        repeat.
        """
        if not artifact_ids:
            return 0
        with self.tx() as cur:
            cur.execute(
                """
                UPDATE crawl_queue
                   SET state = 'pending', lease_until = NULL, updated_at = now()
                 WHERE source = %s AND artifact_id = ANY(%s) AND state = 'leased'
                """,
                (source, artifact_ids),
            )
            return cur.rowcount

    def posts_already_captured(self, source: str, post_ids: list[str]) -> set[str]:
        """Which of these posts we already have an artifact for.

        One artifact per post is the default policy: 91% of multi-image posts
        share a single workflow topology, so the second artifact is nearly always
        a duplicate graph.
        """
        if not post_ids:
            return set()
        with self.tx() as cur:
            cur.execute(
                """
                SELECT DISTINCT post_id FROM crawl_queue
                 WHERE source = %s AND post_id = ANY(%s)
                   AND state IN ('done', 'leased', 'pending')
                """,
                (source, post_ids),
            )
            return {r["post_id"] for r in cur.fetchall()}

    def queue_stats(self, source: str) -> dict[str, int]:
        with self.tx() as cur:
            cur.execute(
                "SELECT state, count(*) AS n FROM crawl_queue WHERE source = %s GROUP BY state",
                (source,),
            )
            return {r["state"]: r["n"] for r in cur.fetchall()}

    # --- cursors ---------------------------------------------------------

    def get_cursor(self, source: str, stream: str) -> dict | None:
        with self.tx() as cur:
            cur.execute(
                "SELECT cursor, frontier_id FROM crawl_cursor WHERE source=%s AND stream=%s",
                (source, stream),
            )
            return cur.fetchone()

    def set_cursor(
        self, source: str, stream: str, cursor: str | None, frontier_id: str | None
    ) -> None:
        with self.tx() as cur:
            cur.execute(
                """
                INSERT INTO crawl_cursor (source, stream, cursor, frontier_id, updated_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (source, stream) DO UPDATE
                   SET cursor = EXCLUDED.cursor,
                       frontier_id = COALESCE(EXCLUDED.frontier_id, crawl_cursor.frontier_id),
                       updated_at = now()
                """,
                (source, stream, cursor, frontier_id),
            )

    # --- partitions ------------------------------------------------------

    def add_partitions(self, source: str, kind: str, keys: list[str]) -> int:
        if not keys:
            return 0
        with self.tx() as cur:
            cur.executemany(
                """
                INSERT INTO crawl_partition (source, kind, key)
                VALUES (%s, %s, %s)
                ON CONFLICT (source, kind, key) DO NOTHING
                """,
                [(source, kind, k) for k in keys],
            )
            return cur.rowcount

    def claim_partition(self, source: str, kind: str, lease_seconds: int = 1800) -> dict | None:
        """Take the least-recently-swept un-exhausted partition."""
        with self.tx() as cur:
            cur.execute(
                """
                WITH claimed AS (
                    SELECT source, kind, key
                      FROM crawl_partition
                     WHERE source = %s AND kind = %s AND NOT exhausted
                       AND (lease_until IS NULL OR lease_until < now())
                     ORDER BY swept_at NULLS FIRST, discovered_at
                     LIMIT 1
                       FOR UPDATE SKIP LOCKED
                )
                UPDATE crawl_partition p
                   SET lease_until = now() + make_interval(secs => %s)
                  FROM claimed c
                 WHERE p.source=c.source AND p.kind=c.kind AND p.key=c.key
             RETURNING p.source, p.kind, p.key, p.cursor, p.pages_read, p.images_seen
                """,
                (source, kind, lease_seconds),
            )
            return cur.fetchone()

    def update_partition(
        self,
        source: str,
        kind: str,
        key: str,
        *,
        cursor: str | None,
        exhausted: bool,
        pages_read: int,
        images_seen: int,
        error: str | None = None,
    ) -> None:
        with self.tx() as cur:
            cur.execute(
                """
                UPDATE crawl_partition
                   SET cursor = %s, exhausted = %s, swept_at = now(),
                       pages_read = %s, images_seen = %s,
                       lease_until = NULL, last_error = %s
                 WHERE source=%s AND kind=%s AND key=%s
                """,
                (
                    cursor,
                    exhausted,
                    pages_read,
                    images_seen,
                    (error or "")[:500] or None,
                    source,
                    kind,
                    key,
                ),
            )

    def partition_stats(self, source: str) -> dict[str, int]:
        with self.tx() as cur:
            cur.execute(
                """
                SELECT kind,
                       count(*) FILTER (WHERE exhausted)      AS exhausted,
                       count(*) FILTER (WHERE NOT exhausted)  AS pending
                  FROM crawl_partition WHERE source = %s GROUP BY kind
                """,
                (source,),
            )
            out: dict[str, int] = {}
            for row in cur.fetchall():
                out[f"{row['kind']}_exhausted"] = row["exhausted"]
                out[f"{row['kind']}_pending"] = row["pending"]
            return out

    # --- blobs -----------------------------------------------------------

    def blob_exists(self, sha: bytes) -> bool:
        with self.tx() as cur:
            cur.execute("SELECT 1 FROM blob WHERE sha256 = %s", (sha,))
            return cur.fetchone() is not None

    def record_blob(
        self, sha: bytes, *, size: int, kind: str, key: str, expires_at: dt.datetime | None
    ) -> None:
        with self.tx() as cur:
            cur.execute(
                """
                INSERT INTO blob (sha256, bytes, kind, bucket_key, expires_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (sha256) DO NOTHING
                """,
                (sha, size, kind, key, expires_at),
            )

    def expired_blobs(self, limit: int = 1000) -> list[dict]:
        with self.tx() as cur:
            cur.execute(
                """
                SELECT sha256, bucket_key FROM blob
                 WHERE expires_at IS NOT NULL AND expires_at < now()
                 LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()

    def forget_blob(self, sha: bytes) -> None:
        with self.tx() as cur:
            cur.execute("DELETE FROM blob WHERE sha256 = %s", (sha,))

    # --- politeness latch ------------------------------------------------

    def latched(self, host: str) -> dict | None:
        """The active latch for a host, if any. Checked before every request run."""
        with self.tx() as cur:
            cur.execute(
                "SELECT host, latched_at, reason, status_code, consecutive, detail "
                "FROM crawl_latch WHERE host = %s AND cleared_at IS NULL",
                (host,),
            )
            return cur.fetchone()

    def active_latches(self) -> list[dict]:
        with self.tx() as cur:
            cur.execute(
                "SELECT host, latched_at, reason, status_code, consecutive "
                "FROM crawl_latch WHERE cleared_at IS NULL ORDER BY latched_at"
            )
            return cur.fetchall()

    def record_transient(self, host: str, status: int, threshold: int, base_seconds: int) -> int:
        """Count a transient 5xx and, past `threshold`, back the host off.

        Returns the seconds this host is now backed off for, 0 if not yet.
        Doubling from `base_seconds`, capped so a long outage cannot push the
        next attempt beyond a day and silently retire the source.

        Unlike `record_refusal` this is NOT sticky: `record_success` clears it.
        A 503 is a source under load, not a source refusing us, and the two
        deserve different costs to recover from.
        """
        with self.tx() as cur:
            cur.execute(
                """
                INSERT INTO crawl_host_backoff (host, consecutive_transient, last_status,
                                                updated_at)
                VALUES (%s, 1, %s, now())
                ON CONFLICT (host) DO UPDATE
                   SET consecutive_transient = crawl_host_backoff.consecutive_transient + 1,
                       last_status = EXCLUDED.last_status,
                       updated_at = now()
             RETURNING consecutive_transient
                """,
                (host, status),
            )
            consecutive = cur.fetchone()["consecutive_transient"]
            if consecutive < threshold:
                return 0
            seconds = min(base_seconds * (2 ** (consecutive - threshold)), 86400)
            cur.execute(
                "UPDATE crawl_host_backoff SET backoff_until = now() + make_interval(secs => %s) "
                "WHERE host = %s",
                (seconds, host),
            )
            return seconds

    def backing_off(self, host: str) -> dict | None:
        """The active backoff for a host, or None. Expired rows read as None."""
        with self.tx() as cur:
            cur.execute(
                "SELECT host, backoff_until, consecutive_transient, last_status, "
                "       EXTRACT(epoch FROM backoff_until - now())::int AS seconds_left "
                "  FROM crawl_host_backoff "
                " WHERE host = %s AND backoff_until IS NOT NULL AND backoff_until > now()",
                (host,),
            )
            return cur.fetchone()

    def clear_transient(self, host: str) -> None:
        """A success means the outage is over. Self-clearing is the whole point."""
        with self.tx() as cur:
            cur.execute(
                "UPDATE crawl_host_backoff SET consecutive_transient = 0, backoff_until = NULL, "
                "updated_at = now() WHERE host = %s AND consecutive_transient > 0",
                (host,),
            )

    def record_refusal(self, host: str, status: int, threshold: int, detail: str = "") -> bool:
        """Count a 401/403/429 and latch the host once `threshold` land in a row.

        Returns True if this call latched. Sticky: nothing here ever clears a
        latch, and `ON CONFLICT DO NOTHING` keeps the original reason and
        timestamp rather than overwriting them with the latest repeat.
        """
        with self.tx() as cur:
            cur.execute(
                """
                INSERT INTO crawl_host_health (host, consecutive_refusals, last_refusal_at,
                                               last_status)
                VALUES (%s, 1, now(), %s)
                ON CONFLICT (host) DO UPDATE
                   SET consecutive_refusals = crawl_host_health.consecutive_refusals + 1,
                       last_refusal_at = now(),
                       last_status = EXCLUDED.last_status
             RETURNING consecutive_refusals
                """,
                (host, status),
            )
            consecutive = cur.fetchone()["consecutive_refusals"]
            if consecutive < threshold:
                return False
            cur.execute(
                """
                INSERT INTO crawl_latch (host, reason, status_code, consecutive, detail)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (host) DO NOTHING
                """,
                (
                    host,
                    f"{consecutive} consecutive HTTP {status}",
                    status,
                    consecutive,
                    detail[:500] or None,
                ),
            )
            return True

    def record_success(self, host: str) -> None:
        """Reset the tally. A single refusal between successes must not accumulate."""
        with self.tx() as cur:
            cur.execute(
                """
                INSERT INTO crawl_host_health (host, consecutive_refusals, last_success_at)
                VALUES (%s, 0, now())
                ON CONFLICT (host) DO UPDATE
                   SET consecutive_refusals = 0, last_success_at = now()
                """,
                (host,),
            )

    def clear_latch(self, host: str, by: str) -> bool:
        """Release a latch. Only ever called by a human via `lake unlatch`."""
        with self.tx() as cur:
            cur.execute(
                "UPDATE crawl_latch SET cleared_at = now(), cleared_by = %s "
                "WHERE host = %s AND cleared_at IS NULL",
                (by, host),
            )
            cleared = cur.rowcount > 0
            cur.execute(
                "UPDATE crawl_host_health SET consecutive_refusals = 0 WHERE host = %s", (host,)
            )
            return cleared

    # --- runs ------------------------------------------------------------

    def start_run(self, job: str, source: str) -> str:
        run_id = str(uuid.uuid4())
        with self.tx() as cur:
            cur.execute(
                "INSERT INTO crawl_run (run_id, job, source) VALUES (%s, %s, %s)",
                (run_id, job, source),
            )
        return run_id

    def finish_run(self, run_id: str, **fields: Any) -> None:
        allowed = ("items", "workflows", "requests", "bytes_down", "manifest_key", "error")
        sets = ", ".join(f"{k} = %s" for k in fields if k in allowed)
        values = [fields[k] for k in fields if k in allowed]
        with self.tx() as cur:
            cur.execute(
                f"UPDATE crawl_run SET finished_at = now(){',' if sets else ''} {sets} "
                "WHERE run_id = %s",
                (*values, run_id),
            )


def json_default(value: Any) -> str:
    return json.dumps(value, default=str)
