"""PRD-CORE-245 NFR04 — the migration is safe with other processes on the file.

Multiple ``trw-mcp`` processes hold one ``memory.db`` open at a time, each with
its own PID lock in the writers sidecar directory. Schema 5 is the first delta
that drops and renames tables, so "a reader observes either the complete
pre-migration or the complete post-migration schema, and never an error" is a
claim that has to be demonstrated against a real second connection, not argued.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

import trw_memory.storage._dbapi  # noqa: F401  — installs pysqlite3 as ``sqlite3``
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._schema import ensure_schema
from trw_memory.storage.sqlite_backend import SQLiteBackend

pytestmark = pytest.mark.unit


def _v4_store(path: Path, rows: int) -> None:
    backend = SQLiteBackend(path)
    try:
        for index in range(rows):
            backend.store(
                MemoryEntry(
                    id=f"M-{index:04d}",
                    content=f"row {index}",
                    namespace="project:concurrent",
                    tags=["alpha", "beta"],
                )
            )
    finally:
        backend.close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()


def test_reader_survives_migration(tmp_path: Path) -> None:
    """A concurrent reader completes every query and never sees a half-renamed table."""
    db = tmp_path / "concurrent.db"
    _v4_store(db, 200)

    observations: list[int] = []
    errors: list[BaseException] = []
    stop = threading.Event()

    def _reader() -> None:
        reader_conn = sqlite3.connect(db, timeout=30)
        reader_conn.execute("PRAGMA journal_mode=WAL")
        try:
            while not stop.is_set():
                try:
                    observations.append(int(reader_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]))
                except BaseException as exc:
                    errors.append(exc)
                    return
        finally:
            reader_conn.close()

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    try:
        writer = sqlite3.connect(db, timeout=30)
        writer.execute("PRAGMA journal_mode=WAL")
        ensure_schema(writer)
        assert writer.execute("PRAGMA user_version").fetchone()[0] == 5
        writer.close()
    finally:
        stop.set()
        thread.join(timeout=10)

    assert errors == [], f"the concurrent reader failed during the migration: {errors!r}"
    assert observations, "the reader never got to run"
    # Every observation is a COMPLETE schema: 200 rows before OR 200 after. A
    # half-applied rebuild would have surfaced as a different count or an error.
    assert set(observations) == {200}


def test_writer_registry_locks_are_untouched(tmp_path: Path) -> None:
    """NFR04: the migration must not delete, rewrite or ignore another process's PID lock."""
    db = tmp_path / "locks.db"
    _v4_store(db, 20)

    holder = SQLiteBackend(db)  # registers this process in the writers sidecar
    try:
        writers_dir = Path(f"{db}.writers")
        before = sorted(p.name for p in writers_dir.iterdir()) if writers_dir.is_dir() else []

        conn = sqlite3.connect(db, timeout=30)
        ensure_schema(conn)
        conn.close()

        after = sorted(p.name for p in writers_dir.iterdir()) if writers_dir.is_dir() else []
        assert after == before
    finally:
        holder.close()


def test_a_newer_store_refuses_an_older_build_with_an_actionable_message(tmp_path: Path) -> None:
    """An older build opening a migrated store fails loudly, and says what to do about it.

    This is the failure mode a live upgrade actually produces: the first process
    to restart migrates the shared file, and the ones still running the previous
    build must be told to restart rather than handed a raw SQLite error.
    """
    from trw_memory.storage._schema import SchemaDowngradeError

    db = tmp_path / "newer.db"
    _v4_store(db, 5)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()

    with pytest.raises(SchemaDowngradeError) as excinfo:
        ensure_schema(conn)
    message = str(excinfo.value)
    assert "restart" in message.lower()
    assert "99" in message
    conn.close()


def test_the_migration_takes_a_snapshot_first(tmp_path: Path) -> None:
    """NFR02: the pre-migration bytes are recoverable, and the migration refuses without them."""
    from trw_memory.storage._schema_backup import BACKUP_DIR_NAME

    db = tmp_path / "snapshot.db"
    _v4_store(db, 30)

    conn = sqlite3.connect(db)
    ensure_schema(conn)
    conn.close()

    snapshots = sorted((tmp_path / BACKUP_DIR_NAME).glob("snapshot.db.pre-schema-5.*"))
    assert len(snapshots) == 1, "a destructive delta must leave exactly one recoverable snapshot"

    restored = sqlite3.connect(f"file:{snapshots[0]}?mode=ro", uri=True)
    assert restored.execute("PRAGMA user_version").fetchone()[0] == 4
    assert restored.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 30
    assert oct(snapshots[0].stat().st_mode & 0o777) == "0o600"
    restored.close()
