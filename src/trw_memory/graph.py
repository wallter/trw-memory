# ruff: noqa: F401,I001
"""Knowledge graph -- edge creation, traversal, cross-validation, importance ops.

Supports 13 typed edge types (PRD-CORE-107).  Graph traversal via BFS up to depth 3.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections import deque
from pathlib import Path
from time import monotonic

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
    "list_org_shared_entries",
    "memory_decay_pass",
    "propagate_impact",
    "schedule_graph_update",
    "update_entry_graph",
    "wait_for_graph_updates",
]

from trw_memory.exceptions import StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
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

# _ENTRY_UPDATE_LOCKS / _ENTRY_UPDATE_LOCKS_GUARD are owned by
# _graph_cross_project and re-imported below for back-compat; no local copy here.
_BACKGROUND_GRAPH_THREADS: set[threading.Thread] = set()
_BACKGROUND_GRAPH_THREADS_GUARD = threading.Lock()


def _optional_lock(lock: threading.Lock | None) -> contextlib.AbstractContextManager[bool]:
    """Return a context manager that acquires *lock* if provided, else no-op."""
    if lock is not None:
        return lock
    return contextlib.nullcontext(True)


def _track_graph_thread(thread: threading.Thread) -> None:
    with _BACKGROUND_GRAPH_THREADS_GUARD:
        _BACKGROUND_GRAPH_THREADS.add(thread)


def _untrack_graph_thread(thread: threading.Thread) -> None:
    with _BACKGROUND_GRAPH_THREADS_GUARD:
        _BACKGROUND_GRAPH_THREADS.discard(thread)


def _derive_graph_config(
    backend: StorageBackend,
    config: MemoryConfig | None,
) -> MemoryConfig | None:
    if config is not None:
        return config

    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.storage.yaml_backend import YAMLBackend

    if isinstance(backend, SQLiteBackend):
        db_path = Path(str(backend._db_path))
        return MemoryConfig(
            storage_backend="sqlite",
            storage_path=str(db_path.parent.parent),
            sqlite_db_name=db_path.name,
            embedding_dim=backend._dim,
        )

    if isinstance(backend, YAMLBackend):
        entries_dir = Path(str(backend._dir))
        return MemoryConfig(
            storage_backend="yaml",
            storage_path=str(entries_dir.parent.parent),
        )

    return None


def _run_scheduled_graph_update(
    entry: MemoryEntry,
    config: MemoryConfig,
    embedding: list[float] | None,
) -> None:
    from trw_memory.integrations._backend import create_backend_from_config

    # Reopen the namespace backend inside the worker so the caller can return
    # immediately without sharing a soon-to-close SQLite connection across threads.
    with create_backend_from_config(config, entry.namespace) as backend:
        update_entry_graph(entry, backend, embedding=embedding, config=config)


def schedule_graph_update(
    entry: MemoryEntry,
    backend: StorageBackend,
    *,
    embedding: list[float] | None = None,
    config: MemoryConfig | None = None,
) -> bool:
    """Dispatch best-effort graph enrichment off the write critical path."""
    resolved_config = _derive_graph_config(backend, config)
    if resolved_config is None:
        logger.debug("graph_update_skipped", entry_id=entry.id, reason="missing_background_config")
        return False

    def worker() -> None:
        try:
            _run_scheduled_graph_update(entry, resolved_config, embedding)
        except (StorageError, sqlite3.Error, ValueError):
            logger.warning("graph_update_background_failed", entry_id=entry.id, exc_info=True)
        finally:
            _untrack_graph_thread(threading.current_thread())

    thread = threading.Thread(
        target=worker,
        name=f"trw-memory-graph-{entry.id}",
        daemon=True,
    )
    _track_graph_thread(thread)
    thread.start()
    return True


def wait_for_graph_updates(timeout: float = 5.0) -> None:
    """Block until scheduled graph-update threads finish or *timeout* elapses."""
    deadline = monotonic() + timeout
    while True:
        with _BACKGROUND_GRAPH_THREADS_GUARD:
            threads = [thread for thread in _BACKGROUND_GRAPH_THREADS if thread.is_alive()]
        if not threads:
            return

        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for background graph updates")

        threads[0].join(min(0.05, remaining))


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


# Cross-project validation cluster extracted to _graph_cross_project.py
# (PRD-DIST-245 batch 93). Re-exports preserve back-compat names.
from trw_memory._graph_cross_project import (  # noqa: E402
    _ENTRY_UPDATE_LOCKS as _ENTRY_UPDATE_LOCKS,
    _ENTRY_UPDATE_LOCKS_GUARD as _ENTRY_UPDATE_LOCKS_GUARD,
    append_cross_validation as _append_cross_validation,
    apply_cross_project_validation as _apply_cross_project_validation,
    backend_update_guard as _backend_update_guard,
    cross_validation_prefix as _cross_validation_prefix,
    entry_has_cross_validation as _entry_has_cross_validation,
    entry_update_lock as _entry_update_lock,
    merge_cross_validated_entry as _merge_cross_validated_entry,
    persist_cross_validated_entry as _persist_cross_validated_entry,
    project_scope_key as _project_scope_key,
)

# Importance boost / decay cluster extracted to _graph_decay.py
# (PRD-DIST-245 batch 94). Re-exports preserve back-compat names.
from trw_memory._graph_decay import (  # noqa: E402
    apply_importance_boost as apply_importance_boost,
    apply_importance_decay as apply_importance_decay,
    memory_decay_pass as memory_decay_pass,
)

# Edge-creation cluster extracted to _graph_edges.py (PRD-DIST-245 batch 95).
from trw_memory._graph_edges import (  # noqa: E402
    create_consolidation_edges as create_consolidation_edges,
    create_similarity_edges as create_similarity_edges,
    create_tag_cooccurrence_edges as create_tag_cooccurrence_edges,
)

# Cluster detection + impact propagation extracted to _graph_clusters.py
# (PRD-DIST-245 batch 96).
from trw_memory._graph_clusters import (  # noqa: E402
    _propose_domain_name as _propose_domain_name,
    detect_clusters as detect_clusters,
    propagate_impact as propagate_impact,
)

# Conflict detection + co-anchored edges extracted to _graph_conflicts.py
# (PRD-DIST-245 batch 97).
from trw_memory._graph_conflicts import (  # noqa: E402
    create_co_anchored_edges as create_co_anchored_edges,
    filter_conflicts as filter_conflicts,
    get_conflicts as get_conflicts,
)


def list_org_shared_entries(
    config: MemoryConfig,
    namespace: str,
    *,
    min_importance: float = 0.8,
    limit: int = 25,
    exclude_keys: set[tuple[str, str]] | None = None,
) -> list[MemoryEntry]:
    """Return high-importance cross-validated memories from sibling projects."""
    current_project = _project_scope_key(namespace)
    if current_project is None:
        return []

    from trw_memory.integrations._backend import discover_namespace_backends

    seen = set(exclude_keys or set())
    matches: list[MemoryEntry] = []

    with discover_namespace_backends(config) as stores:
        for namespaces, backend in stores:
            for candidate_namespace in namespaces:
                project_id = _project_scope_key(candidate_namespace)
                if project_id is None or project_id == current_project:
                    continue

                # Push status + min_importance into the storage layer so we
                # only hydrate the high-importance rows we can actually keep,
                # instead of materialising up to 10k full MemoryEntry objects
                # per sibling namespace and discarding most in Python. The
                # cross_validated + dedup + final sort still run below; the
                # limit stays high so the candidate set the sort sees is
                # unchanged from the pre-filter behaviour.
                entries = backend.list_entries(
                    status=MemoryStatus.ACTIVE,
                    namespace=candidate_namespace,
                    min_importance=min_importance,
                    limit=10_000,
                )
                for entry in entries:
                    entry_key = (entry.namespace, entry.id)
                    if entry_key in seen:
                        continue
                    if not entry.cross_validated or entry.importance < min_importance:
                        continue
                    seen.add(entry_key)
                    matches.append(entry)

    matches.sort(key=lambda entry: (entry.importance, entry.updated_at), reverse=True)
    return matches[:limit]


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


# Graph primitives extracted to _graph_primitives.py (PRD-DIST-245 batch 98).
from trw_memory._graph_primitives import (  # noqa: E402
    _safe_cosine_similarity as _safe_cosine_similarity,
    _upsert_edge as _upsert_edge,
)
