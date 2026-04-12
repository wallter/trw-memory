"""Tests for SQLite storage of assertions column.

PRD-CORE-086: Assertions persistence in SQLite backend.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from trw_memory.models.memory import Assertion, AssertionType, MemoryEntry
from trw_memory.storage._schema import ensure_schema
from trw_memory.storage._shared import ENTRY_COLUMNS
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _make_entry(
    entry_id: str = "M-001",
    content: str = "test content",
    assertions: list[Assertion] | None = None,
) -> MemoryEntry:
    """Helper to create a MemoryEntry with optional assertions."""
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        created_at=now,
        updated_at=now,
        assertions=assertions or [],
    )


class TestEntryColumnsCount:
    """Verify ENTRY_COLUMNS tuple size matches expectations."""

    def test_entry_columns_count(self) -> None:
        assert len(ENTRY_COLUMNS) == 52  # +3 for recall_count, helpful_count, unhelpful_count (PRD-CORE-132)
        assert ENTRY_COLUMNS[-1] == "unhelpful_count"


class TestFreshDbSchema:
    """Test that a fresh database has the assertions column."""

    def test_fresh_db_has_column(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "test.db")
        # Store an entry with assertions to verify the column works
        entry = _make_entry(
            assertions=[
                Assertion(type=AssertionType.GREP_PRESENT, pattern="hello", target="*.py"),
            ]
        )
        backend.store(entry)
        retrieved = backend.get("M-001")
        assert retrieved is not None
        assert len(retrieved.assertions) == 1
        backend.close()

    def test_fresh_db_column_in_pragma(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "test2.db")
        conn = backend._conn
        cursor = conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "assertions" in columns
        backend.close()


class TestMigration:
    """Test schema migration adds the assertions column."""

    def test_migration_adds_column(self, tmp_path: Path) -> None:
        """Simulate pre-migration DB (no assertions column), then run migration."""
        db_path = tmp_path / "migrate.db"
        # Create DB with old schema (without assertions)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                detail TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                evidence TEXT DEFAULT '[]',
                importance REAL DEFAULT 0.5,
                status TEXT DEFAULT 'active',
                recurrence INTEGER DEFAULT 1,
                namespace TEXT DEFAULT 'default',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT,
                access_count INTEGER DEFAULT 0,
                q_value REAL DEFAULT 0.5,
                q_observations INTEGER DEFAULT 0,
                source TEXT DEFAULT 'agent',
                source_identity TEXT DEFAULT '',
                merged_from TEXT DEFAULT '[]',
                consolidated_from TEXT DEFAULT '[]',
                consolidated_into TEXT,
                metadata TEXT DEFAULT '{}',
                vector_clock TEXT DEFAULT '{}',
                remote_id TEXT,
                published_to_platform INTEGER DEFAULT 0,
                pending_delete INTEGER DEFAULT 0,
                cross_validated INTEGER DEFAULT 0,
                outcome_history TEXT DEFAULT '[]'
            )
        """)
        conn.commit()
        conn.close()

        # Now open with SQLiteBackend which runs migrations
        backend = SQLiteBackend(db_path)
        cursor = backend._conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "assertions" in columns
        backend.close()

    def test_idempotent_migration(self, tmp_path: Path) -> None:
        """Running ensure_schema twice should not error."""
        backend = SQLiteBackend(tmp_path / "idempotent.db")
        # Call ensure_schema again — should not raise
        ensure_schema(backend._conn)
        # Verify column still exists
        cursor = backend._conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "assertions" in columns
        backend.close()


class TestAssertionsRoundTrip:
    """Test storing and retrieving entries with assertions."""

    def test_assertions_round_trip(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "roundtrip.db")
        assertions = [
            Assertion(type=AssertionType.GREP_PRESENT, pattern="def hello", target="src/*.py"),
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="README.md"),
            Assertion(
                type=AssertionType.GREP_ABSENT,
                pattern="eval\\(",
                target="**/*.py",
                last_result=True,
                last_evidence="correctly absent",
            ),
        ]
        entry = _make_entry(entry_id="M-RT", assertions=assertions)
        backend.store(entry)

        retrieved = backend.get("M-RT")
        assert retrieved is not None
        assert len(retrieved.assertions) == 3
        assert retrieved.assertions[0].type == "grep_present"
        assert retrieved.assertions[0].pattern == "def hello"
        assert retrieved.assertions[1].type == "glob_exists"
        assert retrieved.assertions[1].target == "README.md"
        assert retrieved.assertions[2].type == "grep_absent"
        assert retrieved.assertions[2].last_result is True
        assert retrieved.assertions[2].last_evidence == "correctly absent"
        backend.close()

    def test_empty_assertions_round_trip(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "empty.db")
        entry = _make_entry(entry_id="M-EMPTY")
        backend.store(entry)

        retrieved = backend.get("M-EMPTY")
        assert retrieved is not None
        assert retrieved.assertions == []
        backend.close()

    def test_28_tuple_destructuring(self, tmp_path: Path) -> None:
        """Verify the 28-element tuple destructuring works end-to-end."""
        backend = SQLiteBackend(tmp_path / "tuple.db")
        assertions = [
            Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="*.tmp"),
        ]
        entry = _make_entry(entry_id="M-TUPLE", assertions=assertions)
        backend.store(entry)

        # Retrieve and verify all fields survived
        retrieved = backend.get("M-TUPLE")
        assert retrieved is not None
        assert retrieved.id == "M-TUPLE"
        assert retrieved.content == "test content"
        assert len(retrieved.assertions) == 1
        assert retrieved.assertions[0].type == "glob_absent"
        assert retrieved.assertions[0].target == "*.tmp"
        backend.close()

    def test_pre_migration_graceful_degradation(self, tmp_path: Path) -> None:
        """DB created without assertions column returns empty list after migration."""
        db_path = tmp_path / "premigrate.db"
        # Create old-schema DB and insert a row
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                detail TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                evidence TEXT DEFAULT '[]',
                importance REAL DEFAULT 0.5,
                status TEXT DEFAULT 'active',
                recurrence INTEGER DEFAULT 1,
                namespace TEXT DEFAULT 'default',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT,
                access_count INTEGER DEFAULT 0,
                q_value REAL DEFAULT 0.5,
                q_observations INTEGER DEFAULT 0,
                source TEXT DEFAULT 'agent',
                source_identity TEXT DEFAULT '',
                merged_from TEXT DEFAULT '[]',
                consolidated_from TEXT DEFAULT '[]',
                consolidated_into TEXT,
                metadata TEXT DEFAULT '{}',
                vector_clock TEXT DEFAULT '{}',
                remote_id TEXT,
                published_to_platform INTEGER DEFAULT 0,
                pending_delete INTEGER DEFAULT 0,
                cross_validated INTEGER DEFAULT 0,
                outcome_history TEXT DEFAULT '[]'
            )
        """)
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO memories (id, content, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("M-OLD", "old entry", now, now),
        )
        conn.commit()
        conn.close()

        # Open with SQLiteBackend (triggers migration)
        backend = SQLiteBackend(db_path)
        retrieved = backend.get("M-OLD")
        assert retrieved is not None
        assert retrieved.assertions == []  # Default empty list
        backend.close()
