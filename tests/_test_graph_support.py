"""Shared support for the ``test_graph*`` test family."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from trw_memory.graph import _merge_cross_validated_entry
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._schema import (
    CREATE_GRAPH_EDGES,
    CREATE_IDX_MEMORY_TAGS_ENTRY,
    CREATE_IDX_MGE_SOURCE,
    CREATE_IDX_MGE_TARGET,
    CREATE_MEMORIES,
    CREATE_MEMORY_TAGS,
)

# PRD-CORE-245: these used to be hand-copied CREATE TABLE statements that drifted
# from the shipped schema the moment a column moved. They now ALIAS the real DDL,
# so a schema change is a test failure in the assertion rather than a silent
# divergence in the fixture.
_CREATE_MEMORIES = CREATE_MEMORIES
_CREATE_GRAPH_EDGES = CREATE_GRAPH_EDGES
_CREATE_MEMORY_TAGS = CREATE_MEMORY_TAGS
_CREATE_IDX_MGE_SOURCE = CREATE_IDX_MGE_SOURCE
_CREATE_IDX_MGE_TARGET = CREATE_IDX_MGE_TARGET
_CREATE_IDX_MEMORY_TAGS_ENTRY = CREATE_IDX_MEMORY_TAGS_ENTRY


def _make_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite connection with the required schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute(_CREATE_MEMORIES)
    conn.execute(_CREATE_GRAPH_EDGES)
    conn.execute(_CREATE_MEMORY_TAGS)
    conn.execute(_CREATE_IDX_MGE_SOURCE)
    conn.execute(_CREATE_IDX_MGE_TARGET)
    conn.execute(_CREATE_IDX_MEMORY_TAGS_ENTRY)
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
        # ``_make_entry`` leaves the row in the model default namespace even
        # though the store file is opened for project:default, so the merge is
        # addressed at the namespace the ROW carries (PRD-CORE-245 FR03).
        _merge_cross_validated_entry(storage, "e1", project_id, 0.97, namespace="default")


_V1 = [1.0, 0.0, 0.0, 0.0]
_V2 = [0.99, 0.1, 0.0, 0.0]
_V3 = [0.0, 0.0, 0.0, 1.0]
