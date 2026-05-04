"""Conflict detection + co-anchored edges for the graph layer.

Belongs to the ``graph.py`` facade. Re-exported there for back-compat.

3 helpers covering the conflict-edge subsystem:

- ``create_co_anchored_edges`` — write ``co_anchored`` edges for
  entries sharing anchor files (capped per file to prevent explosion).
- ``get_conflicts`` — return ``conflicts_with`` edges involving a
  given entry id (both directions).
- ``filter_conflicts`` — suppress lower-importance side of
  ``conflicts_with`` pairs in a result list (equal-importance pairs
  kept).

Looks up ``_upsert_edge`` via the parent ``graph`` module so test
monkeypatches still propagate.

Extracted as PRD-DIST-245 Phase 2 batch 97.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _graph_module() -> Any:
    """Return the parent graph module for indirection lookups."""
    from trw_memory import graph as _graph

    return _graph


def create_co_anchored_edges(
    conn: sqlite3.Connection,
    entry_id: str,
    anchor_files: list[str],
    max_per_file: int = 50,
) -> int:
    """Create ``co_anchored`` edges for entries sharing anchor files.

    Capped at *max_per_file* per anchor file to prevent explosion.
    """
    g = _graph_module()
    now = datetime.now(timezone.utc).isoformat()
    created = 0

    for anchor_file in anchor_files:
        rows = conn.execute(
            "SELECT DISTINCT m.id FROM memories m, json_each(m.anchors) je "
            "WHERE json_extract(je.value, '$.file') = ? "
            "AND m.id != ? "
            "LIMIT ?",
            (anchor_file, entry_id, max_per_file),
        ).fetchall()

        for (other_id,) in rows:
            meta = {"anchor_file": anchor_file}
            g._upsert_edge(conn, entry_id, other_id, "co_anchored", 0.8, now, metadata=meta)
            created += 1

    if created:
        conn.commit()
    logger.debug("co_anchored_edges_created", entry_id=entry_id, count=created)
    return created


def get_conflicts(
    conn: sqlite3.Connection,
    entry_id: str,
) -> list[dict[str, str]]:
    """Return ``conflicts_with`` edges involving *entry_id* (both directions)."""
    rows = conn.execute(
        "SELECT source_id, target_id, edge_metadata "
        "FROM memory_graph_edges "
        "WHERE edge_type = 'conflicts_with' "
        "AND (source_id = ? OR target_id = ?)",
        (entry_id, entry_id),
    ).fetchall()

    return [
        {
            "source_id": row[0],
            "target_id": row[1],
            "edge_metadata": row[2] or "{}",
        }
        for row in rows
    ]


def filter_conflicts(
    entries: list[dict[str, object]],
    conn: sqlite3.Connection,
) -> list[dict[str, object]]:
    """Suppress lower-importance side of ``conflicts_with`` pairs in *entries*.

    Equal-importance pairs are kept (no suppression).
    """
    if len(entries) < 2:
        return list(entries)

    entry_ids = {str(e["id"]) for e in entries}
    importance_map: dict[str, float] = {
        str(e["id"]): float(e.get("importance", 0.5))  # type: ignore[arg-type]
        for e in entries
    }

    suppressed: set[str] = set()

    for entry in entries:
        eid = str(entry["id"])
        if eid in suppressed:
            continue
        conflicts = get_conflicts(conn, eid)
        for conflict in conflicts:
            other_id = conflict["target_id"] if conflict["source_id"] == eid else conflict["source_id"]
            if other_id not in entry_ids or other_id in suppressed:
                continue

            my_imp = importance_map.get(eid, 0.5)
            other_imp = importance_map.get(other_id, 0.5)

            if my_imp > other_imp:
                suppressed.add(other_id)
                logger.debug(
                    "conflict_suppressed",
                    kept=eid,
                    suppressed_id=other_id,
                    importance_kept=my_imp,
                    importance_suppressed=other_imp,
                )
            elif other_imp > my_imp:
                suppressed.add(eid)
                logger.debug(
                    "conflict_suppressed",
                    kept=other_id,
                    suppressed_id=eid,
                    importance_kept=other_imp,
                    importance_suppressed=my_imp,
                )
                break

    return [e for e in entries if str(e["id"]) not in suppressed]
