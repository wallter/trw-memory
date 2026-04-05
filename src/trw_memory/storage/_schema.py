"""DDL definitions and schema-management helpers for the SQLite backend.

Contains all ``CREATE TABLE``, ``CREATE INDEX``, and migration logic
extracted from :mod:`sqlite_backend` so that the backend module stays
focused on read/write operations.
"""

from __future__ import annotations

import contextlib
import sqlite3

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

CREATE_MEMORIES = """
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
    assertions        TEXT DEFAULT '[]',
    anchors           TEXT DEFAULT '[]',
    anchor_validity   REAL DEFAULT 1.0,
    type              TEXT DEFAULT 'pattern',
    nudge_line        TEXT DEFAULT '',
    expires_at        TEXT DEFAULT '',
    confidence        TEXT DEFAULT 'unverified',
    task_type         TEXT DEFAULT '',
    domain            TEXT DEFAULT '[]',
    phase_origin      TEXT DEFAULT '',
    phase_affinity    TEXT DEFAULT '[]',
    team_origin       TEXT DEFAULT '',
    protection_tier   TEXT DEFAULT 'normal',
    sessions_surfaced INTEGER DEFAULT 0,
    avg_rework_delta  TEXT DEFAULT NULL,
    outcome_correlation TEXT DEFAULT ''
)
"""

CREATE_IDX_NAMESPACE = "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)"
CREATE_IDX_STATUS = "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)"

CREATE_GRAPH_EDGES = """
CREATE TABLE IF NOT EXISTS memory_graph_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    edge_type       TEXT NOT NULL,
    weight          REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    created_at      TEXT NOT NULL,
    edge_metadata   TEXT DEFAULT '{}',
    UNIQUE (source_id, target_id, edge_type)
)
"""

CREATE_IDX_MGE_SOURCE = "CREATE INDEX IF NOT EXISTS idx_mge_source ON memory_graph_edges(source_id, edge_type)"
CREATE_IDX_MGE_TARGET = "CREATE INDEX IF NOT EXISTS idx_mge_target ON memory_graph_edges(target_id, edge_type)"

CREATE_NAMESPACES = """
CREATE TABLE IF NOT EXISTS memory_namespaces (
    namespace_id  TEXT PRIMARY KEY,
    team_id       TEXT,
    created_at    TEXT NOT NULL,
    expires_at    TEXT,
    status        TEXT NOT NULL DEFAULT 'active'
)
"""

CREATE_IDX_MN_STATUS = "CREATE INDEX IF NOT EXISTS idx_mn_status ON memory_namespaces(status, expires_at)"


# ---------------------------------------------------------------------------
# Schema bootstrap + migrations
# ---------------------------------------------------------------------------


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create core tables, run column migrations, and build indexes.

    Safe to call multiple times (all operations are idempotent).

    Args:
        conn: An open SQLite connection.
    """
    cursor = conn.cursor()
    try:
        cursor.execute(CREATE_MEMORIES)
        cursor.execute(CREATE_GRAPH_EDGES)
        cursor.execute(CREATE_NAMESPACES)

        # Migration: rename columns from older schema versions.
        # Must run BEFORE index creation since indexes reference new names.
        _rename_cols = [
            ("memories", "impact", "importance"),
            ("memories", "expires", "expires_at"),
            ("memory_graph_edges", "relation", "edge_type"),
        ]
        for table, old_name, new_name in _rename_cols:
            with contextlib.suppress(sqlite3.OperationalError):
                cursor.execute(f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}")

        cursor.execute(CREATE_IDX_NAMESPACE)
        cursor.execute(CREATE_IDX_STATUS)
        cursor.execute(CREATE_IDX_MGE_SOURCE)
        cursor.execute(CREATE_IDX_MGE_TARGET)

        # Migration: add missing columns to memory_namespaces
        for col_name, col_def in [
            ("team_id", "TEXT"),
            ("expires_at", "TEXT"),
            ("status", "TEXT NOT NULL DEFAULT 'active'"),
        ]:
            with contextlib.suppress(sqlite3.OperationalError):
                cursor.execute(f"ALTER TABLE memory_namespaces ADD COLUMN {col_name} {col_def}")

        cursor.execute(CREATE_IDX_MN_STATUS)

        # Migration: add new columns for sync + graph (Sprint 37)
        _migrate_cols = [
            ("vector_clock", "TEXT DEFAULT '{}'"),
            ("remote_id", "TEXT"),
            ("published_to_platform", "INTEGER DEFAULT 0"),
            ("pending_delete", "INTEGER DEFAULT 0"),
            ("cross_validated", "INTEGER DEFAULT 0"),
            ("outcome_history", "TEXT DEFAULT '[]'"),
            ("assertions", "TEXT DEFAULT '[]'"),
        ]
        # Migration: add provenance columns (PRD-CORE-099)
        _migrate_cols += [
            ("client_profile", "TEXT DEFAULT ''"),
            ("model_id", "TEXT DEFAULT ''"),
        ]
        # Migration: add PRD-CORE-110 typed entry fields
        _migrate_cols += [
            ("type", "TEXT DEFAULT 'pattern'"),
            ("nudge_line", "TEXT DEFAULT ''"),
            ("expires_at", "TEXT DEFAULT ''"),
            ("confidence", "TEXT DEFAULT 'unverified'"),
            ("task_type", "TEXT DEFAULT ''"),
            ("domain", "TEXT DEFAULT '[]'"),
            ("phase_origin", "TEXT DEFAULT ''"),
            ("phase_affinity", "TEXT DEFAULT '[]'"),
            ("team_origin", "TEXT DEFAULT ''"),
            ("protection_tier", "TEXT DEFAULT 'normal'"),
        ]
        # Migration: add PRD-CORE-111 anchor fields
        _migrate_cols += [
            ("anchors", "TEXT DEFAULT '[]'"),
            ("anchor_validity", "REAL DEFAULT 1.0"),
        ]
        # Migration: add PRD-CORE-108 outcome attribution fields
        _migrate_cols += [
            ("sessions_surfaced", "INTEGER DEFAULT 0"),
            ("avg_rework_delta", "TEXT DEFAULT NULL"),
            ("outcome_correlation", "TEXT DEFAULT ''"),
        ]
        for col_name, col_def in _migrate_cols:
            with contextlib.suppress(sqlite3.OperationalError):
                cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")

        # Migration: add edge_metadata column for PRD-CORE-107 typed edges
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute(
                "ALTER TABLE memory_graph_edges ADD COLUMN edge_metadata TEXT DEFAULT '{}'"
            )

        conn.commit()
    finally:
        cursor.close()


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Create the ``vec_memories`` virtual table and its companion index.

    Args:
        conn: An open SQLite connection (with sqlite-vec loaded).
        dim: Embedding vector dimension.
    """
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(embedding float[{dim}])")
    # Companion table to map rowid <-> entry_id
    conn.execute(
        "CREATE TABLE IF NOT EXISTS vec_index (rowid INTEGER PRIMARY KEY AUTOINCREMENT, entry_id TEXT UNIQUE NOT NULL)"
    )
    conn.commit()
