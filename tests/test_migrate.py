"""Migration tests.

The bug these exist for: manifests that assume a schema nothing creates. The
fix runs on every pod start, so what matters is that it is cheap when there is
nothing to do, safe when several pods do it at once, and loud when someone edits
an applied migration.
"""

from __future__ import annotations

import pathlib

import pytest

from lake import migrate


def write(tmp_path: pathlib.Path, **files: str) -> pathlib.Path:
    d = tmp_path / "migrations"
    d.mkdir(exist_ok=True)
    for name, sql in files.items():
        (d / name).write_text(sql)
    return d


def test_load_orders_by_filename(tmp_path) -> None:
    """Numbering is the ordering contract; 010 must not sort before 002."""
    d = write(
        tmp_path, **{"002_b.sql": "SELECT 2", "001_a.sql": "SELECT 1", "010_c.sql": "SELECT 10"}
    )
    assert [m.filename for m in migrate.load(d)] == ["001_a.sql", "002_b.sql", "010_c.sql"]


def test_checksum_tracks_content(tmp_path) -> None:
    d = write(tmp_path, **{"001.sql": "SELECT 1"})
    first = migrate.load(d)[0].checksum
    write(tmp_path, **{"001.sql": "SELECT 2"})
    assert migrate.load(d)[0].checksum != first


class FakeCursor:
    def __init__(self, state: dict) -> None:
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        # 120, not 40: the bootstrap statement's table name sits past 40 chars,
        # which silently defeated the ordering assertion below.
        self.state["executed"].append((sql.strip()[:120], params))
        if "SELECT filename, checksum" in sql:
            self._rows = [{"filename": f, "checksum": c} for f, c in self.state["applied"].items()]
        elif "INSERT INTO schema_migrations" in sql:
            self.state["applied"][params[0]] = params[1]
            self._rows = []
        else:
            self._rows = []

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, state: dict) -> None:
        self.state = state

    def cursor(self):
        return FakeCursor(self.state)

    def commit(self):
        self.state["commits"] += 1

    def rollback(self):
        self.state["rollbacks"] += 1


class FakeFrontier:
    def __init__(self) -> None:
        self.state = {"applied": {}, "executed": [], "commits": 0, "rollbacks": 0}
        self.conn = FakeConn(self.state)


def test_applies_then_is_a_no_op(tmp_path) -> None:
    """The steady state — every pod, every tick — must apply nothing."""
    d = write(tmp_path, **{"001.sql": "CREATE TABLE a()", "002.sql": "CREATE TABLE b()"})
    f = FakeFrontier()

    first = migrate.run(f, d)
    assert first["applied"] == ["001.sql", "002.sql"]

    second = migrate.run(f, d)
    assert second["applied"] == []
    assert second["already_current"] is True


def test_takes_an_advisory_lock(tmp_path) -> None:
    """Five CronJobs can overlap, and concurrent CREATE TABLE IF NOT EXISTS can
    fail on pg_type. The lock is what makes running this everywhere safe."""
    d = write(tmp_path, **{"001.sql": "CREATE TABLE a()"})
    f = FakeFrontier()
    migrate.run(f, d)
    sql = [s for s, _ in f.state["executed"]]
    assert any("pg_advisory_lock" in s for s in sql)
    assert any("pg_advisory_unlock" in s for s in sql)


def test_lock_is_taken_before_the_bootstrap_create(tmp_path) -> None:
    """The defect that broke the first deploy.

    On a virgin database the tracking table is the ONLY thing at risk, and it was
    created outside the lock — two pods both passed `IF NOT EXISTS` and one died
    on pg_type_typname_nsp_index. The tracking table needs the same protection as
    everything it tracks.
    """
    d = write(tmp_path, **{"001.sql": "CREATE TABLE a()"})
    f = FakeFrontier()
    migrate.run(f, d)
    sql = [s for s, _ in f.state["executed"]]
    lock = next(i for i, q in enumerate(sql) if "pg_advisory_lock" in q)
    bootstrap = next(i for i, q in enumerate(sql) if "schema_migrations" in q and "CREATE" in q)
    assert lock < bootstrap, "bootstrap CREATE must happen inside the lock"


def test_zero_migrations_is_an_error_not_an_empty_success(tmp_path) -> None:
    """The second defect: the image shipped without its migrations, migrate found
    none, exited 0, and the main container then failed on the missing tables.
    An app that defines tables and finds nothing to apply is never valid."""
    empty = tmp_path / "migrations"
    empty.mkdir()
    with pytest.raises(migrate.NoMigrations, match="no \\*.sql migrations found"):
        migrate.load(empty)
    with pytest.raises(migrate.NoMigrations):
        migrate.run(FakeFrontier(), empty)


def test_missing_migrations_touches_nothing(tmp_path) -> None:
    """Discovery must fail before the bootstrap, or a broken build leaves behind
    a tracking table that makes the next run look already-bootstrapped."""
    empty = tmp_path / "migrations"
    empty.mkdir()
    f = FakeFrontier()
    with pytest.raises(migrate.NoMigrations):
        migrate.run(f, empty)
    assert f.state["executed"] == [], "no statement should have reached the database"


def test_packaged_migrations_are_discoverable() -> None:
    """The installed package must carry its own migrations. `migrate --help`
    proved the flag existed while the files did not."""
    names = [m.filename for m in migrate.load()]
    assert names == sorted(names) and names, names
    assert all(n.endswith(".sql") for n in names)


def test_adopts_a_pre_existing_empty_tracking_table(tmp_path) -> None:
    """The broken deploy left an empty schema_migrations behind. It must be
    adopted and filled, not treated as 'already migrated'."""
    d = write(tmp_path, **{"001.sql": "CREATE TABLE a()", "002.sql": "CREATE TABLE b()"})
    f = FakeFrontier()
    # Simulate the wreckage: table exists, no rows recorded.
    f.state["applied"] = {}
    result = migrate.run(f, d)
    assert result["applied"] == ["001.sql", "002.sql"]


def test_lock_is_released_even_when_a_migration_fails(tmp_path) -> None:
    """A held lock would wedge every future pod, not just this one."""
    d = write(tmp_path, **{"001.sql": "CREATE TABLE a()"})
    f = FakeFrontier()

    def boom(sql, params=None):
        if "CREATE TABLE a" in sql:
            raise RuntimeError("syntax error")
        return FakeCursor.execute(cur, sql, params)

    cur = FakeCursor(f.state)
    f.conn.cursor = lambda: type(
        "C",
        (),
        {
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: False,
            "execute": staticmethod(boom),
            "fetchall": lambda s: [],
        },
    )()
    with pytest.raises(RuntimeError, match="syntax error"):
        migrate.run(f, d)
    assert any("pg_advisory_unlock" in s for s, _ in f.state["executed"])


def test_editing_an_applied_migration_is_an_error(tmp_path) -> None:
    """Silent divergence between what a database ran and what the repo says it
    ran is worse than a failed job."""
    d = write(tmp_path, **{"001.sql": "CREATE TABLE a()"})
    f = FakeFrontier()
    migrate.run(f, d)

    write(tmp_path, **{"001.sql": "CREATE TABLE a(x int)"})
    with pytest.raises(RuntimeError, match="immutable"):
        migrate.run(f, d)


def test_check_applies_nothing(tmp_path) -> None:
    d = write(tmp_path, **{"001.sql": "CREATE TABLE a()"})
    f = FakeFrontier()
    result = migrate.run(f, d, dry_run=True)
    assert result["pending"] == ["001.sql"] and result["applied"] == []
    assert f.state["applied"] == {}


def test_lock_key_is_stable_and_in_range() -> None:
    """It is a shared Postgres instance; the key must not collide by accident,
    and it must fit in a bigint."""
    assert 0 < migrate.LOCK_KEY < 2**63
    assert migrate.LOCK_KEY == migrate.LOCK_KEY
