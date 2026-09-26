# ruff: noqa: F401,I001
"""Knowledge graph -- edge creation, traversal, cross-validation, importance ops.

Supports 13 typed edge types (PRD-CORE-107).  Graph traversal via BFS up to depth 3.
"""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import replace
from heapq import nsmallest

from trw_memory.retrieval.recall_selection import RecallInvocation
from trw_memory.storage.interface import EntryCursor
from pathlib import Path
from typing import Any

import structlog

__all__ = [
    "VALID_EDGE_TYPES",
    "apply_importance_boost",
    "apply_importance_decay",
    "backfill_graph_page",
    "create_co_anchored_edges",
    "create_consolidation_edges",
    "create_similarity_edges",
    "detect_clusters",
    "filter_conflicts",
    "get_conflicts",
    "graph_query",
    "list_org_shared_entries",
    "memory_decay_pass",
    "propagate_impact",
    "schedule_graph_update",
    "schedule_graph_update_many",
    "update_entries_graph",
    "update_entry_graph",
    "wait_for_graph_updates",
]

from trw_memory.exceptions import AuthorizationError, StorageError
from trw_memory._graph_config import derive_graph_config as _derive_graph_config
from trw_memory._graph_worker_pool import submit_graph_job as _submit_graph_job, worker_backend as _worker_backend
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage._sql_utils import iter_bind_chunks

# Background graph-update thread registry extracted to _graph_threads.py.
# Re-export the shims + join primitive so trw_memory.graph.<name> keeps working
# for the 4 prod + 4 test importers of wait_for_graph_updates.
from trw_memory._graph_threads import (
    _REGISTRY as _GRAPH_THREAD_REGISTRY,
    _track_graph_thread as _track_graph_thread,
    _untrack_graph_thread as _untrack_graph_thread,
    wait_for_graph_updates as wait_for_graph_updates,
)

# Back-compat aliases for the pre-extraction module globals. They point at the
# registry's live internals (mutated in place by track/untrack), so any external
# reader still observes the real registry state rather than a detached copy.
_BACKGROUND_GRAPH_THREADS = _GRAPH_THREAD_REGISTRY._threads
_BACKGROUND_GRAPH_THREADS_GUARD = _GRAPH_THREAD_REGISTRY._guard

logger = structlog.get_logger(__name__)

SIMILARITY_THRESHOLD = 0.75
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
# Background graph-update thread tracking lives in _graph_threads.py (registry
# class + singleton); the shims/aliases are re-exported near the bottom of this
# module so trw_memory.graph.wait_for_graph_updates et al. keep working.


def _optional_lock(lock: threading.Lock | None) -> contextlib.AbstractContextManager[bool]:
    """Return a context manager that acquires *lock* if provided, else no-op."""
    if lock is not None:
        return lock
    return contextlib.nullcontext(True)


def _run_scheduled_graph_update(
    entry: MemoryEntry,
    config: MemoryConfig,
    embedding: list[float] | None,
) -> None:
    # Runs on the store's persistent worker (PRD-FIX-143), which owns one backend
    # for its lifetime instead of reopening -- and re-running quick_check -- per row.
    with _worker_backend(config, entry.namespace) as backend:
        update_entry_graph(entry, backend, embedding=embedding, config=config)


def schedule_graph_update(
    entry: MemoryEntry,
    backend: StorageBackend,
    *,
    embedding: list[float] | None = None,
    config: MemoryConfig | None = None,
) -> bool:
    """Queue best-effort graph enrichment on the store's worker, off the write critical path."""
    resolved_config = _derive_graph_config(backend, config)
    if resolved_config is None:
        logger.debug("graph_update_skipped", entry_id=entry.id, reason="missing_background_config")
        return False
    # The lambda re-reads the module global, so tests can substitute the job body.
    return _submit_graph_job(
        resolved_config,
        entry.namespace,
        backend,
        entry.id,
        lambda: _run_scheduled_graph_update(entry, resolved_config, embedding),
    )


def schedule_graph_update_many(
    items: list[tuple[MemoryEntry, list[float] | None]],
    backend: StorageBackend,
    *,
    config: MemoryConfig | None = None,
) -> bool:
    """Queue ONE enrichment pass per namespace of a write batch (see ``_graph_batch``)."""
    resolved_config = _derive_graph_config(backend, config)
    if not items or resolved_config is None:
        logger.debug("graph_update_skipped", count=len(items), reason="empty_batch_or_missing_background_config")
        return False
    queued = True
    for namespace in dict.fromkeys(entry.namespace for entry, _embedding in items):
        sub_batch = [item for item in items if item[0].namespace == namespace]

        def run(
            sub_batch: list[tuple[MemoryEntry, list[float] | None]] = sub_batch, namespace: str = namespace
        ) -> None:
            with _worker_backend(resolved_config, namespace) as ns_backend:
                update_entries_graph(sub_batch, ns_backend, config=resolved_config)

        queued = _submit_graph_job(resolved_config, namespace, backend, f"batch-{sub_batch[0][0].id}", run) and queued
    return queued


def update_entry_graph(
    entry: MemoryEntry,
    backend: StorageBackend,
    *,
    embedding: list[float] | None = None,
    config: MemoryConfig | None = None,
) -> dict[str, int]:
    """Best-effort graph enrichment for one freshly written entry (``update_entries_graph`` of one)."""
    return update_entries_graph([(entry, embedding)], backend, config=config)


# Batched enrichment (one pass per write batch) lives in _graph_batch.py.
from trw_memory._graph_batch import update_entries_graph as update_entries_graph  # noqa: E402

# One resumable page of the forced sweep over existing rows lives in _graph_backfill.py.
from trw_memory._graph_backfill import backfill_graph_page as backfill_graph_page  # noqa: E402

# Cross-project validation cluster extracted to _graph_cross_project.py
# (PRD-DIST-245 batch 93). Re-exports preserve back-compat names.
# ``merge_cross_validated_entry`` has no consumer through this facade -- import
# it from `_graph_cross_project` directly (as the tests below do).
from trw_memory._graph_cross_project import (  # noqa: E402
    _ENTRY_UPDATE_LOCKS as _ENTRY_UPDATE_LOCKS,
    _ENTRY_UPDATE_LOCKS_GUARD as _ENTRY_UPDATE_LOCKS_GUARD,
    cross_validate_entries as cross_validate_entries,
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
)

# Cluster detection + impact propagation extracted to _graph_clusters.py
# (PRD-DIST-245 batch 96).
from trw_memory._graph_clusters import (  # noqa: E402
    _propose_domain_name as _propose_domain_name,
    detect_clusters as detect_clusters,
    propagate_impact as propagate_impact,
)

# BFS traversal + derived tag neighbours extracted to _graph_traversal.py
# (PRD-CORE-245 FR07 — the facade had 8 effective LOC of headroom).
from trw_memory._graph_traversal import (  # noqa: E402
    DERIVED_EDGE_TYPE as DERIVED_EDGE_TYPE,
    MAX_TRAVERSAL_DEPTH as MAX_TRAVERSAL_DEPTH,
    graph_query as graph_query,
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
    invocation: RecallInvocation | None = None,
    entry_filter: Callable[[MemoryEntry], bool] | None = None,
    open_backend: StorageBackend | None = None,
) -> list[MemoryEntry]:
    """Acquire authorized sibling entries; native policy precedes every pool cap.

    *open_backend* is the caller's open store, reused instead of reopened.

    Native paging bounds retained page/final references, not database scans or
    total work. No-invocation callers retain the legacy per-namespace 10k cut.
    """
    current_project = _project_scope_key(namespace)
    if current_project is None:
        return []
    from trw_memory.integrations._backend import discover_namespace_backends
    from trw_memory.security.namespace_scope import NamespaceScopeError
    from trw_memory.security.rbac import Permission, require_namespace_permission, within_grant

    seen = set(exclude_keys or set())

    def candidates() -> Iterator[MemoryEntry]:
        with discover_namespace_backends(config, reuse=open_backend) as stores:
            for namespaces, backend in stores:
                for candidate_namespace in namespaces:
                    project_id = _project_scope_key(candidate_namespace)
                    # A sibling outside the request's grant is skipped silently: org recall
                    # visits every namespace in the store, so a refusal here is routine (W27).
                    if project_id is None or project_id == current_project or not within_grant(candidate_namespace):
                        continue
                    try:
                        require_namespace_permission(config, candidate_namespace, Permission.READ, "read")
                    except AuthorizationError:
                        continue
                    policy = replace(invocation, namespace=candidate_namespace) if invocation else None

                    def allows(
                        entry: MemoryEntry,
                        candidate_namespace: str = candidate_namespace,
                        policy: RecallInvocation | None = policy,
                    ) -> bool:
                        if entry.namespace != candidate_namespace:
                            raise NamespaceScopeError("org producer returned an unauthorized namespace")
                        return (
                            entry.cross_validated
                            and (entry.namespace, entry.id) not in seen
                            and (
                                policy is None
                                or (
                                    policy.allows_entry(entry)
                                    and (policy.temporal.eligible(entry) or policy.temporal.include_superseded)
                                )
                            )
                            and (entry_filter is None or entry_filter(entry))
                        )

                    after = None
                    while True:
                        entries = backend.list_entries(
                            status=MemoryStatus.ACTIVE,
                            namespace=candidate_namespace,
                            min_importance=min_importance,
                            limit=256 if policy else 10_000,
                            after=after,
                            entry_filter=allows if policy else None,
                        )
                        for entry in entries:
                            if not allows(entry) or entry.importance < min_importance:
                                continue
                            seen.add((entry.namespace, entry.id))
                            yield entry
                        if policy is None or len(entries) < 256:
                            break
                        after = EntryCursor.from_entry(entries[-1])

    if invocation is not None:
        return nsmallest(
            limit, candidates(), key=lambda entry: invocation.rank_key(entry, entry.importance, source="org")
        )
    return sorted(candidates(), key=lambda entry: (entry.importance, entry.updated_at), reverse=True)[:limit]


# Graph primitives extracted to _graph_primitives.py (PRD-DIST-245 batch 98).
from trw_memory._graph_primitives import (  # noqa: E402
    _safe_cosine_similarity as _safe_cosine_similarity,
    _upsert_edge as _upsert_edge,
)
