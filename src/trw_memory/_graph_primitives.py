"""Graph primitives — edge upsert + cosine-similarity helper.

Belongs to the ``graph.py`` facade. Re-exported there for back-compat.

Two primitives shared by every helper cluster:

- ``_safe_cosine_similarity`` — :mod:`retrieval.dense.cosine_similarity`
  wrapper that returns 0.0 on dimension mismatch instead of raising.
- ``_upsert_edge`` — INSERT/UPDATE edge row in ``memory_graph_edges``
  with edge-type validation and 4096-byte metadata cap.

Validates ``edge_type`` against ``VALID_EDGE_TYPES`` imported lazily
from the parent ``graph`` module so test patches on
``trw_memory.graph.VALID_EDGE_TYPES`` propagate.

Extracted as PRD-DIST-245 Phase 2 batch 98.
"""

from __future__ import annotations

import json
import sqlite3

import structlog

from trw_memory.exceptions import DimensionMismatchError
from trw_memory.retrieval.dense import cosine_similarity

logger = structlog.get_logger(__name__)


def _safe_cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity with graceful degradation for graph operations.

    Delegates to ``retrieval.dense.cosine_similarity`` but returns 0.0
    on dimension mismatch.  Other ``ValueError`` subclasses are re-raised
    so callers can distinguish true zero-similarity from incompatible vectors.
    """
    try:
        return cosine_similarity(a, b)
    except DimensionMismatchError:
        logger.debug(
            "cosine_dimension_mismatch",
            len_a=len(a),
            len_b=len(b),
        )
        return 0.0


def _upsert_edge(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    created_at: str,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    """Insert or update an edge in the graph.

    Args:
        metadata: Optional key-value metadata stored as JSON alongside the edge.

    Raises:
        ValueError: If *edge_type* is not in ``trw_memory.graph.VALID_EDGE_TYPES``.
    """
    from trw_memory import graph as _graph

    if edge_type not in _graph.VALID_EDGE_TYPES:
        raise ValueError(
            f"Invalid edge type {edge_type!r}. "
            f"Must be one of: {', '.join(sorted(_graph.VALID_EDGE_TYPES))}"
        )
    meta_json = json.dumps(metadata) if metadata else "{}"
    if len(meta_json) > 4096:
        raise ValueError(f"edge metadata exceeds 4096 byte limit ({len(meta_json)} bytes)")
    conn.execute(
        "INSERT INTO memory_graph_edges "
        "(source_id, target_id, edge_type, weight, created_at, edge_metadata) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (source_id, target_id, edge_type) "
        "DO UPDATE SET weight = ?, edge_metadata = ?",
        (source_id, target_id, edge_type, weight, created_at, meta_json, weight, meta_json),
    )
