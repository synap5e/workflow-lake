"""Schema migration, safe to run from every pod on every tick.

The deployment runs this as an initContainer on all five CronJobs rather than as
a one-shot Job someone has to remember. That choice is deliberate: the failure it
prevents is exactly the one that occurred — manifests that assume a schema
nobody creates — and an initContainer makes forgetting structurally impossible.
A new migration ships inside a new image and the first job to tick applies it,
with no human step and no ordering dependency between Flux Kustomizations.

Running it ~300 times a day only works if it is genuinely cheap and genuinely
safe, which needs two things the naive version lacks:

* **An advisory lock.** Five CronJobs can overlap (`concurrencyPolicy: Forbid`
  is per-CronJob, not global), and concurrent `CREATE TABLE IF NOT EXISTS` in
  Postgres can fail with a duplicate-key error on `pg_type` — the IF NOT EXISTS
  check and the create are not atomic against each other. One session-level lock
  serialises them.
* **A version table.** Re-executing every file on every pod start is wasteful and
  makes the logs useless. After the first run this is one SELECT.

Checksums are recorded so that editing an already-applied migration in place is
an error rather than a silent divergence between environments.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass

# Fixed 64-bit key for pg_advisory_lock. Derived from the name so it cannot
# collide by accident with another application's lock on a shared instance.
LOCK_KEY = int.from_bytes(hashlib.sha256(b"workflow-lake:migrate").digest()[:8], "big") % (2**63)

BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text        PRIMARY KEY,
    checksum   text        NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


@dataclass
class Migration:
    filename: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode()).hexdigest()[:16]


def load(directory: str | pathlib.Path) -> list[Migration]:
    """Every .sql file, in filename order. Numbering is the ordering contract."""
    path = pathlib.Path(directory)
    return [Migration(p.name, p.read_text()) for p in sorted(path.glob("*.sql"))]


def pending(conn, migrations: list[Migration]) -> list[Migration]:
    """Which migrations are not yet recorded, verifying the ones that are.

    A checksum mismatch means someone edited an applied migration rather than
    adding a new one. That is a divergence between what this database ran and
    what the repo says it ran, so it fails loudly instead of being papered over.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT filename, checksum FROM schema_migrations")
        applied = {row["filename"]: row["checksum"] for row in cur.fetchall()}

    out: list[Migration] = []
    for migration in migrations:
        recorded = applied.get(migration.filename)
        if recorded is None:
            out.append(migration)
        elif recorded != migration.checksum:
            raise RuntimeError(
                f"{migration.filename} was applied with checksum {recorded} but the file "
                f"now hashes to {migration.checksum}. Applied migrations are immutable — "
                f"add a new file rather than editing this one."
            )
    return out


def run(frontier, directory: str | pathlib.Path, *, dry_run: bool = False) -> dict:
    """Apply pending migrations under an advisory lock. Idempotent."""
    conn = frontier.conn
    migrations = load(directory)

    with conn.cursor() as cur:
        cur.execute(BOOTSTRAP)
    conn.commit()

    if dry_run:
        names = [m.filename for m in pending(conn, migrations)]
        conn.rollback()
        return {"pending": names, "applied": [], "dry_run": True}

    applied: list[str] = []
    with conn.cursor() as cur:
        # Session-level, released explicitly below. Held across the whole batch
        # so a second pod waits rather than interleaving with a half-applied one.
        cur.execute("SELECT pg_advisory_lock(%s)", (LOCK_KEY,))
    try:
        # Re-read inside the lock: another pod may have applied them while we
        # waited, which is the normal case when several jobs tick together.
        for migration in pending(conn, migrations):
            with conn.cursor() as cur:
                cur.execute(migration.sql)
                cur.execute(
                    "INSERT INTO schema_migrations (filename, checksum) VALUES (%s, %s) "
                    "ON CONFLICT (filename) DO NOTHING",
                    (migration.filename, migration.checksum),
                )
            conn.commit()
            applied.append(migration.filename)
    except Exception:
        conn.rollback()
        raise
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
        conn.commit()

    return {"applied": applied, "already_current": not applied, "total": len(migrations)}
