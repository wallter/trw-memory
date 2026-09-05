"""PRD-CORE-231-FR02: additive ``verification_status`` schema migration.

Proves the column is created on a fresh DB, added idempotently to a DB that
predates the migration, and that pre-migration rows read back as ``None``
rather than raising.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._schema import ensure_schema
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _column_names(conn: sqlite3.Connection) -> set[str]:
    return {str(row[1]) for row in conn.execute("PRAGMA table_info(memories)").fetchall()}


def test_fresh_db_has_verification_status_column(tmp_path: Path) -> None:
    """A fresh database carries the column with a NULL default."""
    backend = SQLiteBackend(tmp_path / "fresh.db")
    cols = _column_names(backend._conn)
    assert "verification_status" in cols

    now = datetime.now(timezone.utc)
    backend.store(MemoryEntry(id="M-VS-001", content="fresh", created_at=now, updated_at=now))
    stored = backend.get("M-VS-001", namespace="default")
    assert stored is not None
    assert stored.verification_status is None


def test_additive_migration_idempotent() -> None:
    """Running ``ensure_schema`` twice neither errors nor duplicates the column."""
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_schema(conn)

    names = [str(row[1]) for row in conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert names.count("verification_status") == 1


def test_pre_migration_rows_read_as_none(tmp_path: Path) -> None:
    """A row written before the migration reads back with ``verification_status=None``."""
    db_path = tmp_path / "legacy.db"

    # Write a real row, then physically remove the FR02 column so the file is
    # byte-for-byte a pre-migration database holding a pre-migration row.
    now = datetime.now(timezone.utc)
    seed = SQLiteBackend(db_path)
    seed.store(MemoryEntry(id="M-LEGACY-001", content="legacy content", created_at=now, updated_at=now))
    seed.close()

    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE memories DROP COLUMN verification_status")
    # Roll user_version back to the pre-FR02 schema so ``ensure_schema`` does
    # not short-circuit on its fast path (the reason the column needs a
    # registered _MIGRATIONS delta, not just a _migrate_cols entry).
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    assert "verification_status" not in _column_names(conn)
    conn.close()

    # Reopening runs the idempotent migration, which re-adds the column.
    backend = SQLiteBackend(db_path)
    assert "verification_status" in _column_names(backend._conn)
    entry = backend.get("M-LEGACY-001", namespace="default")
    assert entry is not None
    assert entry.verification_status is None
