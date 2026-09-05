"""The pre-migration snapshot must close its destination connection.

``with sqlite3.connect(...)`` is a *transaction* context manager, not a closing
one: it commits or rolls back and leaves the connection open. On the failure
path that mattered — the raised :class:`SchemaBackupError` chains the original
exception, whose traceback pins the frame that still references the connection,
so the descriptor on a half-written snapshot survives the migration refusal.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trw_memory.storage import _schema_backup
from trw_memory.storage._schema_backup import (
    BACKUP_DIR_NAME,
    SchemaBackupError,
    snapshot_before_migration,
)

# The module under test may be running against the pysqlite3 shim rather than
# the stdlib module, so derive the connection class from what it actually uses.
_sqlite = _schema_backup.sqlite3  # type: ignore[attr-defined]


class _RecordingConnection(_sqlite.Connection):  # type: ignore[misc,name-defined]
    """A real connection that records whether ``close()`` was ever called."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


@pytest.fixture
def recorded_targets(monkeypatch: pytest.MonkeyPatch) -> list[_RecordingConnection]:
    """Capture every snapshot-destination connection the module opens."""
    opened: list[_RecordingConnection] = []
    real_connect = _sqlite.connect

    def _connect(database: object, *args: object, **kwargs: object) -> object:
        is_snapshot = BACKUP_DIR_NAME in Path(str(database)).parts
        if is_snapshot:
            kwargs.setdefault("factory", _RecordingConnection)
        conn = real_connect(database, *args, **kwargs)
        if is_snapshot and isinstance(conn, _RecordingConnection):
            opened.append(conn)
        return conn

    monkeypatch.setattr(_sqlite, "connect", _connect)
    return opened


def _source_db(tmp_path: Path) -> sqlite3.Connection:
    """A file-backed store with one ``memories`` row, so a snapshot is taken."""
    conn = sqlite3.connect(tmp_path / "memory.db")
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO memories VALUES ('a', 'content')")
    conn.commit()
    return conn


def test_snapshot_closes_destination_connection_on_success(
    tmp_path: Path, recorded_targets: list[_RecordingConnection]
) -> None:
    conn = _source_db(tmp_path)
    try:
        destination = snapshot_before_migration(conn, from_version=4, to_version=5)
    finally:
        conn.close()

    assert destination is not None and destination.exists()
    assert len(recorded_targets) == 1
    assert recorded_targets[0].close_calls == 1


def test_snapshot_closes_destination_connection_when_backup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_targets: list[_RecordingConnection],
) -> None:
    """A refused migration must not leave the snapshot connection open."""
    from trw_memory.storage import _permissions

    real_prepare = _permissions.prepare_db_file_mode

    def _prepare_then_corrupt(db_path: Path | str) -> None:
        real_prepare(db_path)
        # Deterministically make ``conn.backup(target)`` raise: a destination
        # that is not a SQLite file at all.
        Path(db_path).write_bytes(b"this is not a database" * 8)

    monkeypatch.setattr(_permissions, "prepare_db_file_mode", _prepare_then_corrupt)

    conn = _source_db(tmp_path)
    try:
        with pytest.raises(SchemaBackupError):
            snapshot_before_migration(conn, from_version=4, to_version=5)
    finally:
        conn.close()

    assert len(recorded_targets) == 1
    assert recorded_targets[0].close_calls == 1


def test_snapshot_skipped_for_empty_store_opens_no_connection(
    tmp_path: Path, recorded_targets: list[_RecordingConnection]
) -> None:
    conn = sqlite3.connect(tmp_path / "memory.db")
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY)")
    conn.commit()
    try:
        assert snapshot_before_migration(conn, from_version=4, to_version=5) is None
    finally:
        conn.close()

    assert recorded_targets == []
    assert not (tmp_path / BACKUP_DIR_NAME).exists()
