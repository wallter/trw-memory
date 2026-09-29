"""A schema migration runs under the EXCLUSIVE ``migrate`` op, with a pre-migration backup (PRD-CORE-306 S3).

A store that needs migrating is decided before any shared hold is taken, then
opened under ``migrate``: never beside another process's open connection, and
never left half-migrated by a kill. The backup taken first is the store as the
older build left it, so rolling back to it (to 4.0.1, schema 8) loses nothing.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.test_store_lock import Probe
from trw_memory import _store_lock
from trw_memory.exceptions import StoreBusyError
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._schema import SCHEMA_VERSION
from trw_memory.storage._schema_backup import BACKUP_DIR_NAME, KEEP_SNAPSHOTS
from trw_memory.storage.sqlite_backend import SQLiteBackend

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX advisory locks")

_R8 = 8  # trw-memory 4.0.1's schema: the version a rollback must be able to reopen


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "memory.db"


@pytest.fixture
def probe(db: Path) -> Iterator[Probe]:
    child = Probe(db)
    yield child
    child.close()


@pytest.fixture
def short_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_store_lock.WAITS, "open", 0.3)


def _r8_store(path: Path, rows: int = 12) -> set[str]:
    """A populated store in 4.0.1's shape: no ``vec_index.space_key``, stamped schema 8."""
    store = SQLiteBackend(path)
    try:
        for index in range(rows):
            store.store(MemoryEntry(id=f"L-{index:03d}", content=f"row {index}", namespace="project:r8"))
    finally:
        store.close()
    with contextlib.closing(sqlite3.connect(path, isolation_level=None)) as conn:
        if "space_key" in {row[1] for row in conn.execute("PRAGMA table_info(vec_index)")}:
            conn.execute("DROP INDEX IF EXISTS idx_vec_index_space")
            conn.execute("ALTER TABLE vec_index DROP COLUMN space_key")
        conn.execute(f"PRAGMA user_version = {_R8}")
    return _ids(path)


def _read(path: Path, sql: str) -> list[tuple[object, ...]]:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return conn.execute(sql).fetchall()


def _version(path: Path) -> int:
    return int(_read(path, "PRAGMA user_version")[0][0])  # type: ignore[call-overload]


def _ids(path: Path) -> set[str]:
    return {str(row[0]) for row in _read(path, "SELECT id FROM memories")}


def _backups(path: Path) -> list[Path]:
    return sorted((path.parent / BACKUP_DIR_NAME).glob(f"{path.name}.pre-schema-*"))


# --- the migration never runs beside another process's open ---------------------------------


def test_a_due_migration_refuses_while_another_process_has_the_store_open(
    db: Path, probe: Probe, short_waits: None
) -> None:
    ids = _r8_store(db)
    assert probe.ask("open") == ["ok"]

    with pytest.raises(StoreBusyError, match="migrate"):
        SQLiteBackend(db)
    assert (_version(db), _ids(db), _backups(db)) == (_R8, ids, []), "nothing was migrated or copied"

    assert probe.ask("release") == ["ok"]
    SQLiteBackend(db).close()
    assert (_version(db), _ids(db)) == (SCHEMA_VERSION, ids)


def test_an_opener_waiting_on_a_migration_another_process_finished_opens_beside_it(db: Path, probe: Probe) -> None:
    ids = _r8_store(db)
    assert probe.ask("backend") == ["ok"]  # migrates, then keeps the store open for good

    SQLiteBackend(db).close()  # no migration is due any more, so no exclusive hold is wanted
    assert (_version(db), _ids(db)) == (SCHEMA_VERSION, ids)


# --- the pre-migration backup, and rolling back to it ---------------------------------------


def test_an_r8_store_rolled_back_to_its_pre_migration_backup_is_intact(db: Path) -> None:
    ids = _r8_store(db)
    SQLiteBackend(db).close()
    [backup] = _backups(db)
    assert backup.name.startswith(f"memory.db.pre-schema-{SCHEMA_VERSION}.")

    for sidecar in ("-wal", "-shm"):
        Path(f"{db}{sidecar}").unlink(missing_ok=True)
    shutil.copyfile(backup, db)

    assert _read(db, "PRAGMA integrity_check") == [("ok",)]
    assert _version(db) == _R8, "an older build refuses a newer user_version"
    assert _ids(db) == ids
    assert "space_key" not in {row[1] for row in _read(db, "PRAGMA table_info(vec_index)")}, "the 4.0.1 shape"
    SQLiteBackend(db).close()  # and this build migrates it again
    assert (_version(db), _ids(db)) == (SCHEMA_VERSION, ids)


def test_pre_migration_backups_are_bounded(db: Path) -> None:
    _r8_store(db)
    folder = db.parent / BACKUP_DIR_NAME
    folder.mkdir()
    old = [folder / f"memory.db.pre-schema-{n}.2026010{n}T000000Z" for n in range(1, KEEP_SNAPSHOTS + 2)]
    for age, path in enumerate(old):
        path.write_bytes(b"old")
        os.utime(path, (1_000_000 + age, 1_000_000 + age))
    other = folder / "other.db.pre-schema-5.20260101T000000Z"
    other.write_bytes(b"another store's")

    SQLiteBackend(db).close()

    kept = _backups(db)
    assert len(kept) == KEEP_SNAPSHOTS
    assert set(old[-(KEEP_SNAPSHOTS - 1) :]) <= set(kept), "the newest older snapshots stay"
    assert other.exists()


def test_pruning_matches_the_store_name_literally(tmp_path: Path) -> None:
    """A glob character in the store's name never selects another store's backups (sol r1 P2)."""
    db = tmp_path / "mem?.db"
    _r8_store(db)
    folder = tmp_path / BACKUP_DIR_NAME
    folder.mkdir()
    neighbour = [folder / f"memA.db.pre-schema-{n}.2026010{n}T000000Z" for n in range(1, KEEP_SNAPSHOTS + 2)]
    for path in neighbour:
        path.write_bytes(b"another store's")

    SQLiteBackend(db).close()

    assert all(path.exists() for path in neighbour)


# --- a kill mid-migration leaves the old store or the new one, never half -------------------

_MIGRATOR = r"""
import sys, time
from pathlib import Path
from trw_memory.storage import _init_helpers, _schema
where = sys.argv[2]
if where == "delta":
    original = _schema._MIGRATIONS[9]
    def pause(cursor):
        original(cursor)
        print("migrating", flush=True)
        time.sleep(60)
    _schema._MIGRATIONS[9] = pause
else:
    original = _init_helpers.ensure_schema
    def pause(conn):
        original(conn)
        print("migrating", flush=True)
        time.sleep(60)
    _init_helpers.ensure_schema = pause
from trw_memory.storage.sqlite_backend import SQLiteBackend
SQLiteBackend(Path(sys.argv[1]))
"""


@pytest.mark.parametrize(("where", "after_kill"), [("delta", _R8), ("committed", SCHEMA_VERSION)])
def test_a_kill_mid_migration_leaves_a_whole_store_and_the_backup(
    db: Path, short_waits: None, where: str, after_kill: int
) -> None:
    ids = _r8_store(db)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    child = subprocess.Popen(
        [sys.executable, "-c", _MIGRATOR, str(db), where], stdout=subprocess.PIPE, text=True, env=env
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "migrating"
        with pytest.raises(StoreBusyError):  # the migration holds the store: nothing opens beside it
            SQLiteBackend(db)
        os.kill(child.pid, signal.SIGKILL)
        child.wait()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()

    assert (_version(db), _ids(db)) == (after_kill, ids)
    assert len(_backups(db)) == 1
    started = time.monotonic()
    SQLiteBackend(db).close()  # the kernel dropped the killed migrator's lock at once
    assert time.monotonic() - started < 5
    assert (_version(db), _ids(db)) == (SCHEMA_VERSION, ids)
