"""Knowledge graph -- edge creation, traversal, cross-validation, importance ops.

Supports 13 typed edge types (PRD-CORE-107).  Graph traversal via BFS up to depth 3.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections import deque
from datetime import datetime, timedelta, timezone

import structlog

__all__ = [
    "VALID_EDGE_TYPES",
    "apply_importance_boost",
    "apply_importance_decay",
    "create_co_anchored_edges",
    "create_consolidation_edges",
    "create_similarity_edges",
    "create_tag_cooccurrence_edges",
    "detect_clusters",
    "detect_cross_validation",
    "filter_conflicts",
    "get_conflicts",
    "graph_query",
    "memory_decay_pass",
    "propagate_impact",
    "update_entry_graph",
]

from trw_memory.exceptions import DimensionMismatchError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.retrieval.dense import cosine_similarity
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

MAX_TRAVERSAL_DEPTH = 3
SIMILARITY_THRESHOLD = 0.75
CROSS_VALIDATION_THRESHOLD = 0.92
TAG_COOCCURRENCE_MIN_SHARED = 2
CANDIDATE_LIMIT = 500
IMPORTANCE_BOOST = 0.05
DECAY_DELTA = 0.1

# PRD-CORE-107: All valid edge types (13 total)
VALID_EDGE_TYPES: frozenset[str] = frozenset(
    {
        # Existing types
        "similarity",
        "tag_cooccurrence",
        "consolidation",
        # New typed relationships
        "anchored_to",
        "related_to",
        "same_root_cause",
        "depends_on",
        "produced",
        "motivated_by",
        "co_anchored",
        "supersedes",
        "evidence_for",
        "conflicts_with",
    }
)

# Propagation rates for impact spreading along edge types
_PROPAGATION_RATES: dict[str, float] = {
    "evidence_for": 0.3,
    "co_anchored": 0.2,
    "same_root_cause": 0.15,
    "related_to": 0.1,
    "depends_on": 0.1,
}
_NEGATIVE_MULTIPLIER = 0.5


def _optional_lock(lock: threading.Lock | None) -> contextlib.AbstractContextManager[bool]:
    """Return a context manager that acquires *lock* if provided, else no-op."""
    if lock is not None:
        return lock
    return contextlib.nullcontext(True)


def update_entry_graph(
    entry: MemoryEntry,
    backend: StorageBackend,
    *,
    embedding: list[float] | None = None,
    config: MemoryConfig | None = None,
) -> dict[str, int]:
    """Best-effort graph enrichment for a freshly written entry.

    The graph is a secondary index over the canonical memory row. If the active
    backend does not expose a SQLite connection, graph updates are skipped
    without affecting the primary write path.
    """
    conn = getattr(backend, "_conn", None)
    if not isinstance(conn, sqlite3.Connection):
        logger.debug("graph_update_skipped", entry_id=entry.id, reason="no_sqlite_connection")
        return {"similarity_edges": 0, "tag_edges": 0, "consolidation_edges": 0}

    candidate_entries = backend.list_entries(
        status=MemoryStatus.ACTIVE,
        namespace=entry.namespace,
        limit=CANDIDATE_LIMIT,
    )
    candidate_ids = [candidate.id for candidate in candidate_entries if candidate.id != entry.id]
    candidate_embeddings = (
        list(backend.get_stored_embeddings(candidate_ids).items()) if embedding is not None and candidate_ids else None
    )
    lock = getattr(backend, "_lock", None)

    similarity_edges = create_similarity_edges(
        entry,
        conn,
        embedding=embedding,
        candidate_embeddings=candidate_embeddings,
        lock=lock,
    )
    tag_edges = create_tag_cooccurrence_edges(
        entry,
        conn,
        candidate_entries=candidate_entries,
        lock=lock,
    )
    consolidation_edges = create_consolidation_edges(
        entry,
        conn,
        lock=lock,
    )
    cross_validated_projects = _apply_cross_project_validation(
        entry,
        backend,
        conn,
        embedding=embedding,
        config=config,
    )
    return {
        "similarity_edges": similarity_edges,
        "tag_edges": tag_edges,
        "consolidation_edges": consolidation_edges,
        "cross_validated_projects": cross_validated_projects,
    }


def _project_scope_key(namespace: str) -> str | None:
    """Return a stable project key for project-scoped namespaces."""
    if namespace == "default":
        return "default"
    if namespace.startswith("project:"):
        return namespace.split(":", 1)[1]
    return None


def _cross_validation_prefix(project_id: str) -> str:
    return f"cross_validated:project_id={project_id}:"


def _entry_has_cross_validation(entry: MemoryEntry, project_id: str) -> bool:
    prefix = _cross_validation_prefix(project_id)
    return any(event.startswith(prefix) for event in entry.outcome_history)


def _append_cross_validation(entry: MemoryEntry, project_id: str, similarity: float) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    outcome = (
        f"cross_validated:project_id={project_id}:similarity={similarity:.4f}:"
        f"timestamp={now.isoformat()}"
    )
    return entry.model_copy(
        update={
            "cross_validated": True,
            "outcome_history": [*entry.outcome_history, outcome],
            "updated_at": now,
        }
    )


def _persist_cross_validated_entry(
    backend: StorageBackend,
    original: MemoryEntry,
    updated: MemoryEntry,
) -> None:
    if updated == original:
        return
    backend.update(
        original.id,
        cross_validated=updated.cross_validated,
        importance=updated.importance,
        outcome_history=updated.outcome_history,
        updated_at=updated.updated_at,
    )


def _apply_cross_project_validation(
    entry: MemoryEntry,
    backend: StorageBackend,
    conn: sqlite3.Connection,
    *,
    embedding: list[float] | None = None,
    config: MemoryConfig | None = None,
) -> int:
    """Cross-validate against sibling project stores when embeddings exist.

    Package-local cross-project evidence comes from sibling on-disk project
    namespaces. This keeps the feature usable without waiting on a platform-side
    embedding feed while still failing closed when embeddings are unavailable.
    """
    if embedding is None:
        return 0

    current_project = _project_scope_key(entry.namespace)
    if current_project is None:
        return 0

    from trw_memory.integrations._backend import discover_namespace_backends

    cfg = config or MemoryConfig()
    matched_projects = 0
    updated_entry = entry

    with discover_namespace_backends(cfg) as stores:
        for namespaces, remote_backend in stores:
            project_namespaces = [
                namespace
                for namespace in namespaces
                if (project_id := _project_scope_key(namespace)) is not None and project_id != current_project
            ]
            for namespace in project_namespaces:
                project_id = _project_scope_key(namespace)
                if project_id is None:
                    continue

                remote_entries = remote_backend.list_entries(
                    status=MemoryStatus.ACTIVE,
                    namespace=namespace,
                    limit=CANDIDATE_LIMIT,
                )
                if not remote_entries:
                    continue

                remote_embeddings = remote_backend.get_stored_embeddings([candidate.id for candidate in remote_entries])
                remote_candidates = [
                    (candidate, remote_embedding)
                    for candidate in remote_entries
                    if (remote_embedding := remote_embeddings.get(candidate.id)) is not None
                ]
                remote_payload = [
                    (candidate.id, project_id, remote_embedding)
                    for candidate, remote_embedding in remote_candidates
                ]
                if not detect_cross_validation(
                    updated_entry,
                    conn,
                    embedding=embedding,
                    remote_entries=remote_payload,
                ):
                    continue

                for remote_entry, remote_embedding in remote_candidates:
                    similarity = _safe_cosine_similarity(embedding, remote_embedding)
                    if similarity <= CROSS_VALIDATION_THRESHOLD:
                        continue

                    if not _entry_has_cross_validation(updated_entry, project_id):
                        updated_entry = _append_cross_validation(updated_entry, project_id, similarity)
                        updated_entry = apply_importance_boost(updated_entry)
                        matched_projects += 1

                    if _entry_has_cross_validation(remote_entry, current_project):
                        continue

                    updated_remote = _append_cross_validation(remote_entry, current_project, similarity)
                    updated_remote = apply_importance_boost(updated_remote)
                    _persist_cross_validated_entry(remote_backend, remote_entry, updated_remote)

    _persist_cross_validated_entry(backend, entry, updated_entry)
    return matched_projects


def create_similarity_edges(
    entry: MemoryEntry,
    conn: sqlite3.Connection,
    embedding: list[float] | None = None,
    candidate_embeddings: list[tuple[str, list[float]]] | None = None,
    *,
    lock: threading.Lock | None = None,
) -> int:
    """Create similarity edges between entry and candidates above threshold.

    Args:
        entry: The newly written entry.
        conn: SQLite connection (must have memory_graph_edges table).
        embedding: The entry's embedding vector.
        candidate_embeddings: List of (entry_id, embedding) pairs to compare against.
        lock: Optional threading lock for thread-safe commit.

    Returns:
        Number of edges created.
    """
    if embedding is None or candidate_embeddings is None:
        return 0

    created = 0
    now = datetime.now(timezone.utc).isoformat()

    with _optional_lock(lock):
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
    *,
    lock: threading.Lock | None = None,
) -> int:
    """Create tag co-occurrence edges for entries sharing 2+ tags.

    Uses Jaccard similarity as weight.
    Limited to 500 most recent candidates.

    Args:
        lock: Optional threading lock for thread-safe commit.
    """
    if not entry.tags or candidate_entries is None:
        return 0

    entry_tags = set(entry.tags)
    created = 0
    now = datetime.now(timezone.utc).isoformat()

    with _optional_lock(lock):
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
    *,
    lock: threading.Lock | None = None,
) -> int:
    """Create consolidation lineage edges from consolidated_from.

    Args:
        lock: Optional threading lock for thread-safe commit.
    """
    if not entry.consolidated_from:
        return 0

    created = 0
    now = datetime.now(timezone.utc).isoformat()

    with _optional_lock(lock):
        for source_id in entry.consolidated_from:
            # Check source exists
            row = conn.execute("SELECT id FROM memories WHERE id = ?", (source_id,)).fetchone()
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
                f"SELECT target_id, edge_type, weight FROM memory_graph_edges "  # noqa: S608 — placeholders is ? repeated (no user input in SQL structure); values are parameterized
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
                results.append(
                    {
                        "id": target_id,
                        "depth": current_depth + 1,
                        "edge_type": edge_type,
                        "weight": weight,
                    }
                )
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
    outcome = f"importance_boost:delta=+{delta:.2f}:reason={reason}:new_value={new_importance:.4f}:timestamp={now}"

    return entry.model_copy(
        update={
            "importance": new_importance,
            "outcome_history": [*entry.outcome_history, outcome],
            "cross_validated": True,
            "updated_at": datetime.now(timezone.utc),
        }
    )


def apply_importance_decay(
    entry: MemoryEntry,
    delta: float = DECAY_DELTA,
) -> MemoryEntry:
    """Apply importance decay for unused shared memories.

    Floors at 0.0. Records in outcome_history.
    """
    new_importance = max(round(entry.importance - delta, 4), 0.0)
    now = datetime.now(timezone.utc).isoformat()
    outcome = f"importance_decay:delta=-{delta:.2f}:reason=unused_90d:new_value={new_importance:.4f}:timestamp={now}"

    return entry.model_copy(
        update={
            "importance": new_importance,
            "outcome_history": [*entry.outcome_history, outcome],
            "updated_at": datetime.now(timezone.utc),
        }
    )


def memory_decay_pass(
    conn: sqlite3.Connection,
    cutoff_days: int = 90,
    batch_size: int = 1000,
    *,
    lock: threading.Lock | None = None,
) -> dict[str, int]:
    """Run decay pass on cross-validated memories unused for cutoff_days.

    Args:
        lock: Optional threading lock for thread-safe commit.

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
    # Capture timestamp once before the loop: all entries in one decay pass
    # should share the same timestamp for consistency and performance.
    batch_now = datetime.now(timezone.utc).isoformat()
    with _optional_lock(lock):
        try:
            for (entry_id,) in rows:
                outcome = f"importance_decay:delta=-{DECAY_DELTA:.2f}:reason=unused_90d:timestamp={batch_now}"
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
            logger.exception("memory_decay_pass_failed")
            raise

    return {
        "processed": decayed,
        "remaining": max(total_qualifying - decayed, 0),
        "total_decayed": decayed,
    }


# ---------------------------------------------------------------------------
# PRD-CORE-107: Typed relationships — co-anchoring, conflicts, clusters,
#               impact propagation
# ---------------------------------------------------------------------------


def create_co_anchored_edges(
    conn: sqlite3.Connection,
    entry_id: str,
    anchor_files: list[str],
    max_per_file: int = 50,
) -> int:
    """Create ``co_anchored`` edges for entries sharing anchor files.

    Capped at *max_per_file* per anchor file to prevent explosion.
    """
    now = datetime.now(timezone.utc).isoformat()
    created = 0

    for anchor_file in anchor_files:
        # SQLite JSON: find entries whose anchors array contains an object
        # with a matching "file" value.  json_each expands the array.
        rows = conn.execute(
            "SELECT DISTINCT m.id FROM memories m, json_each(m.anchors) je "
            "WHERE json_extract(je.value, '$.file') = ? "
            "AND m.id != ? "
            "LIMIT ?",
            (anchor_file, entry_id, max_per_file),
        ).fetchall()

        for (other_id,) in rows:
            meta = {"anchor_file": anchor_file}
            _upsert_edge(conn, entry_id, other_id, "co_anchored", 0.8, now, metadata=meta)
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
    """Suppress lower-importance side of ``conflicts_with`` edges in *entries*.

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
            # Determine the other side of the conflict
            other_id = (
                conflict["target_id"]
                if conflict["source_id"] == eid
                else conflict["source_id"]
            )
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
                break  # this entry is suppressed, stop checking its conflicts

    return [e for e in entries if str(e["id"]) not in suppressed]


def detect_clusters(
    conn: sqlite3.Connection,
    min_size: int = 5,
    min_connectivity: float = 0.6,
) -> list[dict[str, object]]:
    """Detect dense subgraphs via greedy clique-expansion bounded by graph size."""
    # Build adjacency from all edges (deduplicated, undirected)
    rows = conn.execute(
        "SELECT DISTINCT source_id, target_id FROM memory_graph_edges"
    ).fetchall()

    adj: dict[str, set[str]] = {}
    for src, tgt in rows:
        adj.setdefault(src, set()).add(tgt)
        adj.setdefault(tgt, set()).add(src)

    if len(adj) < min_size:
        return []

    # Sort by degree descending for seed selection
    nodes_by_degree = sorted(adj.keys(), key=lambda n: len(adj[n]), reverse=True)
    used: set[str] = set()
    clusters: list[dict[str, object]] = []

    for seed in nodes_by_degree:
        if seed in used:
            continue

        # Greedy expansion: start with seed, add neighbors that keep connectivity
        cluster = {seed}
        candidates = sorted(adj[seed] - used, key=lambda n: len(adj.get(n, set())), reverse=True)

        for cand in candidates:
            trial = cluster | {cand}
            n = len(trial)
            max_edges = n * (n - 1) / 2
            if max_edges == 0:
                continue
            # Count edges among trial members
            actual = sum(
                1 for node in trial for neighbor in adj.get(node, set()) if neighbor in trial and neighbor > node
            )
            connectivity = actual / max_edges
            if connectivity >= min_connectivity:
                cluster.add(cand)

        if len(cluster) >= min_size:
            n = len(cluster)
            max_edges = n * (n - 1) / 2
            actual = sum(
                1 for node in cluster for neighbor in adj.get(node, set()) if neighbor in cluster and neighbor > node
            )
            connectivity = actual / max_edges if max_edges > 0 else 0.0

            # Propose domain name from member metadata (tags or id patterns)
            domain_name = _propose_domain_name(conn, list(cluster))
            clusters.append(
                {
                    "domain_name": domain_name,
                    "member_ids": sorted(cluster),
                    "connectivity": round(connectivity, 4),
                }
            )
            used.update(cluster)

    logger.debug("clusters_detected", count=len(clusters))
    return clusters


def propagate_impact(
    conn: sqlite3.Connection,
    entry_id: str,
    importance_delta: float,
    max_depth: int = 2,
    max_affected: int = 50,
) -> list[tuple[str, float]]:
    """BFS importance propagation from *entry_id*.  Negative deltas at 0.5x rate."""
    affected: list[tuple[str, float]] = []
    visited: set[str] = {entry_id}
    queue: deque[tuple[str, int, float]] = deque()
    queue.append((entry_id, 0, importance_delta))
    now = datetime.now(timezone.utc).isoformat()

    while queue and len(affected) < max_affected:
        node_id, depth, current_delta = queue.popleft()
        if depth >= max_depth:
            continue

        # Find neighbors with propagatable edge types
        placeholders = ", ".join("?" for _ in _PROPAGATION_RATES)
        rows = conn.execute(
            f"SELECT target_id, edge_type FROM memory_graph_edges "  # noqa: S608
            f"WHERE source_id = ? AND edge_type IN ({placeholders})",
            (node_id, *_PROPAGATION_RATES.keys()),
        ).fetchall()

        for target_id, edge_type in rows:
            if target_id in visited or len(affected) >= max_affected:
                continue

            rate = _PROPAGATION_RATES.get(edge_type, 0.0)
            if rate == 0.0:
                continue

            propagated = current_delta * rate
            if importance_delta < 0:
                propagated *= _NEGATIVE_MULTIPLIER

            # Apply the delta to the target entry
            conn.execute(
                "UPDATE memories SET "
                "importance = MIN(MAX(importance + ?, 0.0), 1.0), "
                "outcome_history = json_insert(outcome_history, '$[#]', ?) "
                "WHERE id = ?",
                (
                    propagated,
                    f"impact_propagation:delta={propagated:+.4f}:from={entry_id}:via={edge_type}:timestamp={now}",
                    target_id,
                ),
            )

            visited.add(target_id)
            affected.append((target_id, round(propagated, 6)))
            # Propagate further with the derived delta
            queue.append((target_id, depth + 1, propagated))

    if affected:
        conn.commit()
    logger.debug(
        "impact_propagated",
        entry_id=entry_id,
        delta=importance_delta,
        affected_count=len(affected),
    )
    return affected


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _propose_domain_name(conn: sqlite3.Connection, member_ids: list[str]) -> str:
    """Propose a domain name for a cluster from member tags or IDs.

    Uses the most common tag among cluster members.  Falls back to a
    generic label based on cluster size.
    """
    if not member_ids:
        return "unknown"
    placeholders = ", ".join("?" for _ in member_ids)
    rows = conn.execute(
        f"SELECT tags FROM memories WHERE id IN ({placeholders})",  # noqa: S608
        tuple(member_ids),
    ).fetchall()

    tag_counts: dict[str, int] = {}
    for (tags_json,) in rows:
        try:
            tags = json.loads(tags_json) if tags_json else []
        except (json.JSONDecodeError, TypeError):
            tags = []
        for tag in tags:
            if isinstance(tag, str) and tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if tag_counts:
        return max(tag_counts, key=lambda t: tag_counts[t])
    return f"cluster-{len(member_ids)}"


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
        ValueError: If *edge_type* is not in :data:`VALID_EDGE_TYPES`.
    """
    if edge_type not in VALID_EDGE_TYPES:
        raise ValueError(
            f"Invalid edge type {edge_type!r}. "
            f"Must be one of: {', '.join(sorted(VALID_EDGE_TYPES))}"
        )
    meta_json = json.dumps(metadata) if metadata else "{}"
    if len(meta_json) > 4096:
        raise ValueError(
            f"edge metadata exceeds 4096 byte limit ({len(meta_json)} bytes)"
        )
    conn.execute(
        "INSERT INTO memory_graph_edges "
        "(source_id, target_id, edge_type, weight, created_at, edge_metadata) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (source_id, target_id, edge_type) "
        "DO UPDATE SET weight = ?, edge_metadata = ?",
        (source_id, target_id, edge_type, weight, created_at, meta_json, weight, meta_json),
    )
