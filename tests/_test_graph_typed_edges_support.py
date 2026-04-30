"""Shared helpers for the ``test_graph_typed_edges*`` test family."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import cast

from trw_memory.storage._schema import ensure_schema


def _make_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite connection with the full schema."""
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn


def _insert_memory_row(
    conn: sqlite3.Connection,
    entry_id: str,
    *,
    content: str = "content",
    importance: float = 0.5,
    anchors_json: str = "[]",
    outcome_history_json: str = "[]",
) -> None:
    """Insert a minimal memory row for typed-edge tests."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memories (id, content, created_at, updated_at, importance, anchors, outcome_history) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (entry_id, content, now, now, importance, anchors_json, outcome_history_json),
    )
    conn.commit()


def _insert_edge(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    metadata: dict[str, str] | None = None,
) -> None:
    """Insert a graph edge directly for test setup."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_graph_edges (source_id, target_id, edge_type, weight, created_at, edge_metadata) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (source_id, target_id, edge_type, weight, now, json.dumps(metadata) if metadata else "{}"),
    )
    conn.commit()


def _count_edges(conn: sqlite3.Connection, edge_type: str | None = None) -> int:
    """Count edges, optionally filtered by edge type."""
    if edge_type is None:
        row = conn.execute("SELECT COUNT(*) FROM memory_graph_edges").fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) FROM memory_graph_edges WHERE edge_type = ?",
            (edge_type,),
        ).fetchone()
    return int(row[0]) if row else 0


def _get_edge_metadata(conn: sqlite3.Connection, source_id: str, target_id: str, edge_type: str) -> dict[str, str]:
    """Read the serialized edge metadata for a specific edge."""
    row = conn.execute(
        "SELECT edge_metadata FROM memory_graph_edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
        (source_id, target_id, edge_type),
    ).fetchone()
    if row and row[0]:
        return cast("dict[str, str]", json.loads(cast("str", row[0])))
    return {}
