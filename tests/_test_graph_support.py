"""Shared support for the ``test_graph*`` test family."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from trw_memory.graph import _merge_cross_validated_entry
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry

_CREATE_MEMORIES = """
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
    merged_from       TEXT DEFAULT '[]',
    consolidated_from TEXT DEFAULT '[]',
    consolidated_into TEXT,
    metadata          TEXT DEFAULT '{}',
    vector_clock      TEXT DEFAULT '{}',
    remote_id         TEXT,
    published_to_platform INTEGER DEFAULT 0,
    pending_delete    INTEGER DEFAULT 0,
    cross_validated   INTEGER DEFAULT 0,
    outcome_history   TEXT DEFAULT '[]'
)
"""

_CREATE_GRAPH_EDGES = """
CREATE TABLE IF NOT EXISTS memory_graph_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    created_at  TEXT NOT NULL,
    edge_metadata TEXT DEFAULT '{}',
    UNIQUE (source_id, target_id, edge_type)
)
"""

_CREATE_IDX_MGE_SOURCE = "CREATE INDEX IF NOT EXISTS idx_mge_source ON memory_graph_edges(source_id, edge_type)"
_CREATE_IDX_MGE_TARGET = "CREATE INDEX IF NOT EXISTS idx_mge_target ON memory_graph_edges(target_id, edge_type)"


def _make_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite connection with the required schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute(_CREATE_MEMORIES)
    conn.execute(_CREATE_GRAPH_EDGES)
    conn.execute(_CREATE_IDX_MGE_SOURCE)
    conn.execute(_CREATE_IDX_MGE_TARGET)
    conn.commit()
    return conn


def _make_entry(
    entry_id: str,
    content: str = "content",
    tags: list[str] | None = None,
    importance: float = 0.5,
    consolidated_from: list[str] | None = None,
    cross_validated: bool = False,
    outcome_history: list[str] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content,
        tags=tags or [],
        importance=importance,
        consolidated_from=consolidated_from or [],
        cross_validated=cross_validated,
        outcome_history=outcome_history or [],
    )


def _insert_memory_row(conn: sqlite3.Connection, entry_id: str, **overrides: object) -> None:
    """Insert a minimal memory row into the in-memory DB."""
    now = datetime.now(timezone.utc).isoformat()
    defaults = {
        "id": entry_id,
        "content": "content",
        "detail": "",
        "tags": "[]",
        "evidence": "[]",
        "importance": 0.5,
        "status": "active",
        "recurrence": 1,
        "namespace": "default",
        "created_at": now,
        "updated_at": now,
        "last_accessed_at": None,
        "access_count": 0,
        "q_value": 0.5,
        "q_observations": 0,
        "source": "agent",
        "source_identity": "",
        "merged_from": "[]",
        "consolidated_from": "[]",
        "consolidated_into": None,
        "metadata": "{}",
        "vector_clock": "{}",
        "remote_id": None,
        "published_to_platform": 0,
        "pending_delete": 0,
        "cross_validated": 0,
        "outcome_history": "[]",
    }
    defaults.update(overrides)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join(["?"] * len(defaults))
    conn.execute(
        f"INSERT INTO memories ({cols}) VALUES ({placeholders})",
        tuple(defaults.values()),
    )
    conn.commit()


def _insert_edge(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
) -> None:
    """Insert an edge directly for test setup."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_graph_edges (source_id, target_id, edge_type, weight, created_at) VALUES (?, ?, ?, ?, ?)",
        (source_id, target_id, edge_type, weight, now),
    )
    conn.commit()


def _count_edges(conn: sqlite3.Connection) -> int:
    """Count total edges in the graph table."""
    row = conn.execute("SELECT COUNT(*) FROM memory_graph_edges").fetchone()
    return int(row[0]) if row else 0


def _merge_cross_validation_in_subprocess(storage_path: str, project_id: str) -> None:
    cfg = MemoryConfig(storage_backend="sqlite", storage_path=storage_path)
    with create_backend_from_config(cfg, "project:default") as storage:
        _merge_cross_validated_entry(storage, "e1", project_id, 0.97)


_V1 = [1.0, 0.0, 0.0, 0.0]
_V2 = [0.99, 0.1, 0.0, 0.0]
_V3 = [0.0, 0.0, 0.0, 1.0]
