"""An unreadable row probe must refuse the migration, not skip the snapshot.

:func:`~trw_memory.storage._schema_backup.snapshot_before_migration` skips the
snapshot for a store with no ``memories`` rows, because a fresh bootstrap has
no prior state a rollback could want. The probe that decides this used to
swallow every ``sqlite3.Error`` and answer "empty" -- so a locked store, a disk
fault or a corrupt page produced the same answer as a brand-new file, and
``ensure_schema`` then ran the destructive schema-5 delta over a populated store
with no way back. That is the one outcome the module's docstring promises
cannot happen.

Only "no such table" is evidence of emptiness. Everything else is evidence of
nothing, and the migration is refused.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage import _schema_backup
from trw_memory.storage._schema import ensure_schema
from trw_memory.storage._schema_backup import BACKUP_DIR_NAME, SchemaBackupError, snapshot_before_migration
from trw_memory.storage.sqlite_backend import SQLiteBackend

pytestmark = pytest.mark.unit

# The module under test may be running against the pysqlite3 shim rather than
# the stdlib module, so derive the connection class from what it actually uses.
_sqlite = _schema_backup.sqlite3  # type: ignore[attr-defined]


class _ProbeFailingConnection(_sqlite.Connection):  # type: ignore[misc,name-defined]
    """A real connection whose ``memories`` row probe raises a chosen error.

    Everything else -- the PRAGMAs, the cursor the migration storm runs on --
    is the genuine engine, so the only thing simulated is the fault itself.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.probe_error: BaseException | None = None

    def execute(self, sql: str, *args: object, **kwargs: object) -> object:
        """Raise ``probe_error`` for the row probe; pass everything else through."""
        error = self.probe_error
        if error is not None and "from memories" in sql.lower():
            raise error
        return super().execute(sql, *args, **kwargs)


def _v4_store(path: Path, rows: int = 3) -> None:
    """Write a populated store stamped at the pre-destructive schema version."""
    backend = SQLiteBackend(path)
    try:
        for index in range(rows):
            backend.store(
                MemoryEntry(
                    id=f"M-{index:04d}",
                    content=f"row {index}",
                    namespace="project:probe",
                    tags=["alpha"],
                )
            )
    finally:
        backend.close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()


def _observed(path: Path) -> tuple[int, int]:
    """Return ``(user_version, memories row count)`` from an independent handle."""
    conn = sqlite3.connect(path)
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        count = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
    finally:
        conn.close()
    return version, count


@pytest.mark.parametrize(
    "probe_error",
    [
        pytest.param(_sqlite.OperationalError("database is locked"), id="locked"),
        pytest.param(_sqlite.DatabaseError("database disk image is malformed"), id="corrupt"),
        pytest.param(_sqlite.OperationalError("disk I/O error"), id="disk_io"),
    ],
)
def test_unreadable_probe_refuses_migration_and_leaves_store_untouched(
    tmp_path: Path, probe_error: BaseException
) -> None:
    """A probe that cannot answer blocks the destructive delta entirely."""
    db = tmp_path / "memory.db"
    _v4_store(db)
    before = _observed(db)
    assert before == (4, 3)

    conn = _sqlite.connect(db, factory=_ProbeFailingConnection)
    try:
        conn.probe_error = probe_error
        with pytest.raises(SchemaBackupError) as raised:
            ensure_schema(conn)
    finally:
        conn.close()

    message = str(raised.value)
    assert str(probe_error) in message, "the refusal must name the underlying cause"
    assert "refusing to migrate" in message
    assert _observed(db) == before, "a refused migration must not touch the store"
    assert not (tmp_path / BACKUP_DIR_NAME).exists(), "nothing was snapshotted, so nothing was migrated"


def test_missing_memories_table_is_still_treated_as_empty(tmp_path: Path) -> None:
    """The one failure that really does mean "no data" keeps skipping the snapshot."""
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (id TEXT)")
    conn.commit()
    try:
        assert snapshot_before_migration(conn, from_version=4, to_version=5) is None
    finally:
        conn.close()
    assert not (tmp_path / BACKUP_DIR_NAME).exists()


def test_readable_populated_store_still_snapshots_then_migrates(tmp_path: Path) -> None:
    """The happy path is unchanged: snapshot first, then the delta lands."""
    db = tmp_path / "memory.db"
    _v4_store(db)

    conn = sqlite3.connect(db)
    try:
        ensure_schema(conn)
    finally:
        conn.close()

    backups = sorted((tmp_path / BACKUP_DIR_NAME).glob("memory.db.pre-schema-5.*"))
    assert len(backups) == 1, f"expected exactly one pre-migration snapshot, got {backups!r}"
    assert _observed(db)[0] == 5
