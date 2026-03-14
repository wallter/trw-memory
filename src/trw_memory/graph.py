"""Knowledge graph -- edge creation, traversal, cross-validation, importance ops.

Stores edges in the memory_graph_edges SQLite table. Three edge types:
- similarity: cosine similarity > 0.75 between embeddings
- tag_cooccurrence: 2+ shared tags (Jaccard weight)
- consolidation: consolidated_from lineage (weight 1.0)

Graph traversal via BFS up to depth 3.
"""

from __future__ import annotations

import sqlite3
from collections import deque
from datetime import datetime, timedelta, timezone

import structlog

from trw_memory.exceptions import DimensionMismatchError
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.dense import cosine_similarity

logger = structlog.get_logger()

MAX_TRAVERSAL_DEPTH = 3
SIMILARITY_THRESHOLD = 0.75
CROSS_VALIDATION_THRESHOLD = 0.92
TAG_COOCCURRENCE_MIN_SHARED = 2
CANDIDATE_LIMIT = 500
IMPORTANCE_BOOST = 0.05
DECAY_DELTA = 0.1


def create_similarity_edges(
    entry: MemoryEntry,
    conn: sqlite3.Connection,
    embedding: list[float] | None = None,
    candidate_embeddings: list[tuple[str, list[float]]] | None = None,
) -> int:
    """Create similarity edges between entry and candidates above threshold.

    Args:
        entry: The newly written entry.
        conn: SQLite connection (must have memory_graph_edges table).
        embedding: The entry's embedding vector.
        candidate_embeddings: List of (entry_id, embedding) pairs to compare against.

    Returns:
        Number of edges created.
    """
    if embedding is None or candidate_embeddings is None:
        return 0

    created = 0
    now = datetime.now(timezone.utc).isoformat()

    for cand_id, cand_emb in candidate_embeddings:
        if cand_id == entry.id:
            continue
        sim = _safe_cosine_similarity(embedding, cand_emb)
        if sim > SIMILARITY_THRESHOLD:
            _upsert_edge(conn, entry.id, cand_id, "similarity", round(sim, 4), now)
            _upsert_edge(conn, cand_id, entry.id, "similarity", round(sim, 4), now)
            created += 2

    conn.commit()
    logger.debug("similarity_edges_created", entry_id=entry.id, count=created)
    return created


def create_tag_cooccurrence_edges(
    entry: MemoryEntry,
    conn: sqlite3.Connection,
    candidate_entries: list[MemoryEntry] | None = None,
) -> int:
    """Create tag co-occurrence edges for entries sharing 2+ tags.

    Uses Jaccard similarity as weight.
    Limited to 500 most recent candidates.
    """
    if not entry.tags or candidate_entries is None:
        return 0

    entry_tags = set(entry.tags)
    created = 0
    now = datetime.now(timezone.utc).isoformat()

    for cand in candidate_entries[:CANDIDATE_LIMIT]:
        if cand.id == entry.id or not cand.tags:
            continue
        cand_tags = set(cand.tags)
        shared = entry_tags & cand_tags
        if len(shared) >= TAG_COOCCURRENCE_MIN_SHARED:
            union_size = len(entry_tags | cand_tags)
            jaccard = len(shared) / union_size if union_size > 0 else 0.0
            _upsert_edge(conn, entry.id, cand.id, "tag_cooccurrence", round(jaccard, 4), now)
            _upsert_edge(conn, cand.id, entry.id, "tag_cooccurrence", round(jaccard, 4), now)
            created += 2

    conn.commit()
    logger.debug("tag_edges_created", entry_id=entry.id, count=created)
    return created


def create_consolidation_edges(
    entry: MemoryEntry,
    conn: sqlite3.Connection,
) -> int:
    """Create consolidation lineage edges from consolidated_from."""
    if not entry.consolidated_from:
        return 0

    created = 0
    now = datetime.now(timezone.utc).isoformat()

    for source_id in entry.consolidated_from:
        # Check source exists
        row = conn.execute(
            "SELECT id FROM memories WHERE id = ?", (source_id,)
        ).fetchone()
        if row is None:
            logger.debug("consolidation_edge_skip_missing", source_id=source_id)
            continue
        _upsert_edge(conn, entry.id, source_id, "consolidation", 1.0, now)
        created += 1

    conn.commit()
    logger.debug("consolidation_edges_created", entry_id=entry.id, count=created)
    return created


def graph_query(
    conn: sqlite3.Connection,
    root_ids: list[str],
    depth: int = 2,
    edge_types: list[str] | None = None,
) -> list[dict[str, str | int | float]]:
    """BFS traversal from root nodes up to specified depth.

    Args:
        conn: SQLite connection.
        root_ids: Starting node IDs.
        depth: Max traversal depth (clamped to 3).
        edge_types: Filter by edge type(s). None = all types.

    Returns:
        List of {"id": str, "depth": int, "edge_type": str, "weight": float}
        for each discovered node, excluding root nodes.
    """
    if not root_ids:
        return []

    if depth > MAX_TRAVERSAL_DEPTH:
        logger.debug("graph_query_depth_clamped", requested=depth, clamped=MAX_TRAVERSAL_DEPTH)
        depth = MAX_TRAVERSAL_DEPTH

    visited: set[str] = set(root_ids)
    results: list[dict[str, str | int | float]] = []
    queue: deque[tuple[str, int]] = deque()

    for rid in root_ids:
        queue.append((rid, 0))

    while queue:
        node_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue

        # Build query with optional edge type filter
        if edge_types:
            placeholders = ", ".join("?" for _ in edge_types)
            sql = (
                f"SELECT target_id, edge_type, weight FROM memory_graph_edges "
                f"WHERE source_id = ? AND edge_type IN ({placeholders})"
            )
            params: tuple[str, ...] = (node_id, *edge_types)
        else:
            sql = "SELECT target_id, edge_type, weight FROM memory_graph_edges WHERE source_id = ?"
            params = (node_id,)

        for row in conn.execute(sql, params).fetchall():
            target_id, edge_type, weight = row
            if target_id not in visited:
                visited.add(target_id)
                results.append({
                    "id": target_id,
                    "depth": current_depth + 1,
                    "edge_type": edge_type,
                    "weight": weight,
                })
                queue.append((target_id, current_depth + 1))

    return results


def detect_cross_validation(
    entry: MemoryEntry,
    conn: sqlite3.Connection,
    embedding: list[float] | None = None,
    remote_entries: list[tuple[str, str, list[float]]] | None = None,
) -> bool:
    """Check if entry is cross-validated by another project.

    Args:
        entry: The entry to check.
        conn: SQLite connection.
        embedding: Entry's embedding.
        remote_entries: List of (entry_id, project_id, embedding) from other projects.

    Returns:
        True if cross-validation detected.
    """
    if embedding is None or remote_entries is None:
        return False

    for _remote_id, project_id, remote_emb in remote_entries:
        sim = _safe_cosine_similarity(embedding, remote_emb)
        if sim > CROSS_VALIDATION_THRESHOLD:
            logger.debug(
                "cross_validation_detected",
                entry_id=entry.id,
                project_id=project_id,
                similarity=round(sim, 4),
            )
            return True

    return False


def apply_importance_boost(
    entry: MemoryEntry,
    reason: str = "cross_validated",
    delta: float = IMPORTANCE_BOOST,
) -> MemoryEntry:
    """Apply an importance boost to an entry, capped at 1.0.

    Records the boost in outcome_history.
    """
    new_importance = min(round(entry.importance + delta, 4), 1.0)
    now = datetime.now(timezone.utc).isoformat()
    outcome = (
        f"importance_boost:delta=+{delta:.2f}:reason={reason}:"
        f"new_value={new_importance:.4f}:timestamp={now}"
    )

    return entry.model_copy(update={
        "importance": new_importance,
        "outcome_history": [*entry.outcome_history, outcome],
        "cross_validated": True,
        "updated_at": datetime.now(timezone.utc),
    })


def apply_importance_decay(
    entry: MemoryEntry,
    delta: float = DECAY_DELTA,
) -> MemoryEntry:
    """Apply importance decay for unused shared memories.

    Floors at 0.0. Records in outcome_history.
    """
    new_importance = max(round(entry.importance - delta, 4), 0.0)
    now = datetime.now(timezone.utc).isoformat()
    outcome = (
        f"importance_decay:delta=-{delta:.2f}:reason=unused_90d:"
        f"new_value={new_importance:.4f}:timestamp={now}"
    )

    return entry.model_copy(update={
        "importance": new_importance,
        "outcome_history": [*entry.outcome_history, outcome],
        "updated_at": datetime.now(timezone.utc),
    })


def memory_decay_pass(
    conn: sqlite3.Connection,
    cutoff_days: int = 90,
    batch_size: int = 1000,
) -> dict[str, int]:
    """Run decay pass on cross-validated memories unused for cutoff_days.

    Returns:
        {"processed": int, "remaining": int, "total_decayed": int}
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=cutoff_days)).isoformat()

    rows = conn.execute(
        "SELECT id FROM memories WHERE cross_validated = 1 "
        "AND (last_accessed_at IS NULL OR last_accessed_at < ?) "
        "LIMIT ?",
        (cutoff, batch_size),
    ).fetchall()

    total = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE cross_validated = 1 "
        "AND (last_accessed_at IS NULL OR last_accessed_at < ?)",
        (cutoff,),
    ).fetchone()
    total_qualifying = total[0] if total else 0

    # Direct SQL for batch performance — StorageBackend.update() is
    # per-entry and would create N+1 round-trips for 1000-row batches.
    decayed = 0
    try:
        for (entry_id,) in rows:
            now = datetime.now(timezone.utc).isoformat()
            outcome = f"importance_decay:delta=-{DECAY_DELTA:.2f}:reason=unused_90d:timestamp={now}"
            conn.execute(
                "UPDATE memories SET importance = MAX(importance - ?, 0.0), "
                "outcome_history = json_insert(outcome_history, '$[#]', ?) "
                "WHERE id = ?",
                (DECAY_DELTA, outcome, entry_id),
            )
            decayed += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "processed": decayed,
        "remaining": max(total_qualifying - decayed, 0),
        "total_decayed": decayed,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
) -> None:
    """Insert or update an edge in the graph."""
    conn.execute(
        "INSERT INTO memory_graph_edges (source_id, target_id, edge_type, weight, created_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET weight = ?",
        (source_id, target_id, edge_type, weight, created_at, weight),
    )
