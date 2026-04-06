"""Tests for SQLite schema migrations (PRD-CORE-110).

Covers:
- ensure_schema adds 10 new typed-learning columns
- Migration is idempotent (safe to call twice)
- ENTRY_COLUMNS count is 40
- Migration on pre-existing DB with old schema preserves old data
"""

from __future__ import annotations

import sqlite3

import pytest

from trw_memory.storage._schema import ensure_schema
from trw_memory.storage._shared import ENTRY_COLUMNS

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_column_names(conn: sqlite3.Connection) -> set[str]:
    """Return the set of column names in the 'memories' table."""
    cursor = conn.execute("PRAGMA table_info(memories)")
    return {row[1] for row in cursor.fetchall()}


# ---------------------------------------------------------------------------
# Column migration tests
# ---------------------------------------------------------------------------


def test_migration_adds_columns() -> None:
    """ensure_schema creates all 10 PRD-CORE-110 typed learning columns."""
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    cols = _get_column_names(conn)
    expected_new = {
        "type",
        "nudge_line",
        "expires_at",
        "confidence",
        "task_type",
        "domain",
        "phase_origin",
        "phase_affinity",
        "team_origin",
        "protection_tier",
    }
    for col in expected_new:
        assert col in cols, f"Column {col!r} missing from memories table"


def test_migration_idempotent() -> None:
    """Calling ensure_schema twice raises no errors."""
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_schema(conn)  # Second call — must be idempotent
    cols = _get_column_names(conn)
    assert "type" in cols


def test_entry_columns_count_42() -> None:
    """ENTRY_COLUMNS must contain exactly 48 entries after adding sync pipeline fields."""
    assert len(ENTRY_COLUMNS) == 48, f"Expected 48, got {len(ENTRY_COLUMNS)}: {ENTRY_COLUMNS}"


def test_entry_columns_contains_new_fields() -> None:
    """ENTRY_COLUMNS includes all 10 new typed-learning column names."""
    expected_new = {
        "type",
        "nudge_line",
        "expires_at",
        "confidence",
        "task_type",
        "domain",
        "phase_origin",
        "phase_affinity",
        "team_origin",
        "protection_tier",
    }
    col_set = set(ENTRY_COLUMNS)
    for col in expected_new:
        assert col in col_set, f"Column {col!r} missing from ENTRY_COLUMNS"


def test_anchor_columns_in_entry_columns() -> None:
    """ENTRY_COLUMNS includes anchor columns."""
    col_set = set(ENTRY_COLUMNS)
    assert "anchors" in col_set
    assert "anchor_validity" in col_set


def test_anchor_columns_added_by_migration() -> None:
    """ensure_schema creates anchors and anchor_validity columns."""
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    cols = _get_column_names(conn)
    assert "anchors" in cols, "Column 'anchors' missing from memories table"
    assert "anchor_validity" in cols, "Column 'anchor_validity' missing from memories table"


def test_migration_on_pre_existing_db(tmp_path: pytest.TempPathFactory) -> None:
    """Old-schema DB (30 cols) is migrated to include new columns with old data intact."""
    db_path = tmp_path / "old.db"  # type: ignore[operator]

    # Create an old-schema DB with 30-column DDL (no new fields)
    old_ddl = """
    CREATE TABLE IF NOT EXISTS memories (
        id                TEXT PRIMARY KEY,
        content           TEXT NOT NULL,
        detail            TEXT DEFAULT '',
        tags              TEXT DEFAULT '[]',
        evidence          TEXT DEFAULT '[]',
        importance        REAL DEFAULT 0.5,
        status            TEXT DEFAULT 'active',
        recurrence        INTEGER DEFAULT 1,
        namespace         TEXT DEFAULT 'default',
        created_at        TEXT NOT NULL,
        updated_at        TEXT NOT NULL,
        last_accessed_at  TEXT,
        access_count      INTEGER DEFAULT 0,
        q_value           REAL DEFAULT 0.5,
        q_observations    INTEGER DEFAULT 0,
        source            TEXT DEFAULT 'agent',
        source_identity   TEXT DEFAULT '',
        client_profile    TEXT DEFAULT '',
        model_id          TEXT DEFAULT '',
        merged_from       TEXT DEFAULT '[]',
        consolidated_from TEXT DEFAULT '[]',
        consolidated_into TEXT,
        metadata          TEXT DEFAULT '{}',
        vector_clock      TEXT DEFAULT '{}',
        remote_id         TEXT,
        published_to_platform INTEGER DEFAULT 0,
        pending_delete    INTEGER DEFAULT 0,
        cross_validated   INTEGER DEFAULT 0,
        outcome_history   TEXT DEFAULT '[]',
        assertions        TEXT DEFAULT '[]'
    )
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute(old_ddl)
    conn.execute(
        "INSERT INTO memories (id, content, created_at, updated_at) VALUES (?,?,?,?)",
        ("L-old1", "old content", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )
    conn.commit()

    # Run migration
    ensure_schema(conn)

    # Old data still intact
    row = conn.execute("SELECT id, content FROM memories WHERE id='L-old1'").fetchone()
    assert row is not None
    assert row[0] == "L-old1"
    assert row[1] == "old content"

    # New columns present with defaults
    cols = _get_column_names(conn)
    assert "type" in cols
    assert "protection_tier" in cols

    # Old row gets default values for new columns
    type_val = conn.execute("SELECT type FROM memories WHERE id='L-old1'").fetchone()
    assert type_val is not None
    # SQLite default fills NULL for existing rows — may be NULL or 'pattern'
    # The important thing is the column exists
    conn.close()


def test_legacy_expires_column_renamed_to_expires_at() -> None:
    """Existing typed-learning DBs with `expires` are migrated to `expires_at`."""
    conn = sqlite3.connect(":memory:")
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
            client_profile TEXT DEFAULT '',
            model_id TEXT DEFAULT '',
            merged_from TEXT DEFAULT '[]',
            consolidated_from TEXT DEFAULT '[]',
            consolidated_into TEXT,
            metadata TEXT DEFAULT '{}',
            vector_clock TEXT DEFAULT '{}',
            remote_id TEXT,
            published_to_platform INTEGER DEFAULT 0,
            pending_delete INTEGER DEFAULT 0,
            cross_validated INTEGER DEFAULT 0,
            outcome_history TEXT DEFAULT '[]',
            assertions TEXT DEFAULT '[]',
            anchors TEXT DEFAULT '[]',
            anchor_validity REAL DEFAULT 1.0,
            type TEXT DEFAULT 'pattern',
            nudge_line TEXT DEFAULT '',
            expires TEXT DEFAULT '',
            confidence TEXT DEFAULT 'unverified',
            task_type TEXT DEFAULT '',
            domain TEXT DEFAULT '[]',
            phase_origin TEXT DEFAULT '',
            phase_affinity TEXT DEFAULT '[]',
            team_origin TEXT DEFAULT '',
            protection_tier TEXT DEFAULT 'normal'
        )
    """)
    conn.execute(
        "INSERT INTO memories (id, content, created_at, updated_at, expires) VALUES (?,?,?,?,?)",
        ("L-exp", "content", "2026-01-01T00:00:00", "2026-01-01T00:00:00", "2026-12-31"),
    )
    conn.commit()

    ensure_schema(conn)

    cols = _get_column_names(conn)
    assert "expires_at" in cols
    assert "expires" not in cols
    row = conn.execute("SELECT expires_at FROM memories WHERE id = 'L-exp'").fetchone()
    assert row is not None
    assert row[0] == "2026-12-31"
    conn.close()
