"""Cross-project validation helpers for the graph layer.

Belongs to the ``graph.py`` facade. Re-exported there for back-compat.

9 helpers covering project-scoped namespace handling and cross-project
validation:

- ``_project_scope_key`` — extract project_id from namespace string.
- ``_cross_validation_prefix`` — outcome-history event prefix per project.
- ``_entry_has_cross_validation`` — has-this-project-already-validated probe.
- ``_append_cross_validation`` — append validation event + boost cross_validated.
- ``_persist_cross_validated_entry`` — write back with diff guard.
- ``_entry_update_lock`` — per-entry threading lock for in-process races.
- ``_backend_update_guard`` — cross-process file-backed RMW guard.
- ``_merge_cross_validated_entry`` — atomic single-project validation +
  importance boost.
- ``cross_validate_entries`` — top-level orchestrator that scores a written
  batch against every sibling project namespace's cached candidates
  (``_graph_sibling_index``) and applies validations bidirectionally.

Extracted as PRD-DIST-245 Phase 2 batch 93.
"""

from __future__ import annotations

import contextlib
import functools
import os
import threading
import weakref
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from trw_memory._graph_sibling_index import SiblingStoreView
from trw_memory.embeddings._similarity_calibration import calibrated_threshold
from trw_memory.embeddings.provenance import EmbeddingSpace
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import StorageBackend

if TYPE_CHECKING:
    from trw_memory.integrations._backend import NamespaceStoreLocation

CROSS_VALIDATION_THRESHOLD = 0.92
CANDIDATE_LIMIT = 500

# WeakValueDictionary so a per-entry lock is reclaimed once no caller holds a
# reference to it. A plain dict accumulated one Lock per (backend, entry_id)
# pair ever cross-validated, with no eviction/TTL/cap — unbounded RAM growth in
# long-lived processes (e.g. an MCP server). The lock is always returned and
# immediately used inside a ``with`` block, so the caller holds a strong
# reference for the entire critical section; the weak entry can only be
# collected after every holder has released it.
_ENTRY_UPDATE_LOCKS: weakref.WeakValueDictionary[tuple[str, str], threading.Lock] = weakref.WeakValueDictionary()
_ENTRY_UPDATE_LOCKS_GUARD = threading.Lock()


def project_scope_key(namespace: str) -> str | None:
    """Return a stable project key for project-scoped namespaces."""
    if namespace == "default":
        return "default"
    if namespace.startswith("project:"):
        return namespace.split(":", 1)[1]
    return None


def cross_validation_prefix(project_id: str) -> str:
    return f"cross_validated:project_id={project_id}:"


def entry_has_cross_validation(entry: MemoryEntry, project_id: str) -> bool:
    prefix = cross_validation_prefix(project_id)
    return any(event.startswith(prefix) for event in entry.outcome_history)


def append_cross_validation(entry: MemoryEntry, project_id: str, similarity: float) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    outcome = f"cross_validated:project_id={project_id}:similarity={similarity:.4f}:timestamp={now.isoformat()}"
    return entry.model_copy(
        update={
            "cross_validated": True,
            "outcome_history": [*entry.outcome_history, outcome],
            "updated_at": now,
        }
    )


def persist_cross_validated_entry(
    backend: StorageBackend,
    original: MemoryEntry,
    updated: MemoryEntry,
) -> None:
    if updated == original:
        return
    backend.update(
        original.id,
        namespace=original.namespace,
        cross_validated=updated.cross_validated,
        importance=updated.importance,
        outcome_history=updated.outcome_history,
        updated_at=updated.updated_at,
    )


def entry_update_lock(backend: StorageBackend, entry_id: str) -> threading.Lock:
    """Return a stable per-entry lock for in-process cross-validation updates."""
    backend_key = str(getattr(backend, "_db_path", f"backend:{id(backend)}"))
    key = (backend_key, entry_id)
    with _ENTRY_UPDATE_LOCKS_GUARD:
        return _ENTRY_UPDATE_LOCKS.setdefault(key, threading.Lock())


def backend_update_guard(backend: StorageBackend) -> contextlib.AbstractContextManager[Path | None]:
    """Cross-process guard for backend RMW updates when the store has a stable on-disk path."""
    from trw_memory.storage.persistence import lock_for_rmw

    db_path = getattr(backend, "_db_path", None)
    if isinstance(db_path, Path):
        return lock_for_rmw(db_path)

    entries_dir = getattr(backend, "_dir", None)
    if isinstance(entries_dir, Path):
        return lock_for_rmw(entries_dir / ".graph-update")

    return contextlib.nullcontext()


def merge_cross_validated_entry(
    backend: StorageBackend,
    entry_id: str,
    project_id: str,
    similarity: float,
    *,
    namespace: str,
) -> tuple[MemoryEntry | None, bool]:
    """Atomically append a single project's validation and boost once.

    The thread lock prevents same-process races; the file-backed guard closes
    the remaining gap where two separate processes open the same store
    concurrently. Looks up ``apply_importance_boost`` via the parent
    ``graph`` module for the test-monkeypatch indirection pattern.
    """
    from trw_memory import graph as _graph_module

    with entry_update_lock(backend, entry_id), backend_update_guard(backend):
        current = backend.get(entry_id, namespace=namespace)
        if current is None:
            return None, False
        if entry_has_cross_validation(current, project_id):
            return current, False

        updated = append_cross_validation(current, project_id, similarity)
        updated = _graph_module.apply_importance_boost(updated)
        persist_cross_validated_entry(backend, current, updated)
        reloaded = backend.get(entry_id, namespace=namespace)
        return (reloaded or updated), True


def _open_sibling(
    config: MemoryConfig, location: NamespaceStoreLocation, writer_key: str | None, writer: StorageBackend
) -> contextlib.AbstractContextManager[StorageBackend]:
    """Open *location*, or reuse the writer's already-open backend when it IS that store."""
    if writer_key is not None and os.path.realpath(location.db_path) == writer_key:
        return contextlib.nullcontext(writer)
    from trw_memory.integrations._backend import open_namespace_store

    return open_namespace_store(config, location)


def cross_validate_entries(
    items: Sequence[tuple[MemoryEntry, list[float], EmbeddingSpace | None]],
    backend: StorageBackend,
    *,
    config: MemoryConfig | None = None,
) -> dict[str, int]:
    """Cross-validate *items* (entry, its embedding, its recorded space) against sibling project stores.

    Returns the number of projects newly validating each entry, by entry id.
    Only sibling vectors recorded in an entry's own space are compared; an
    entry with no known space compares against nothing (a cross-space cosine is
    noise). Package-local evidence comes from sibling on-disk project
    namespaces, so the feature works without a platform embedding feed.

    Sibling candidates come from the process-wide ``SIBLING_CACHE``
    (``_graph_sibling_index``): a sibling store is opened only when its file
    changed since its candidates were read, or when a match must be written
    back to it. Re-reading every sibling on every write made a single-row
    store cost O(sibling namespaces x ``CANDIDATE_LIMIT``) row decodes.
    """
    matched: dict[str, int] = {entry.id: 0 for entry, _embedding, _space in items}
    live = [(entry, embedding, space) for entry, embedding, space in items if space is not None]
    if not live:
        return matched
    from trw_memory.integrations import _backend as stores

    cfg = config or MemoryConfig()
    writer_path = getattr(backend, "_db_path", None)
    writer_key = os.path.realpath(writer_path) if isinstance(writer_path, Path) else None
    for location in stores.namespace_store_locations(cfg):
        opener = functools.partial(_open_sibling, cfg, location, writer_key, backend)
        with SiblingStoreView(location.db_path, opener, candidate_limit=CANDIDATE_LIMIT) as view:
            for namespace in view.namespaces:
                project_id = project_scope_key(namespace)
                if project_id is None:
                    continue
                for entry, embedding, space in live:
                    current_project = project_scope_key(entry.namespace)
                    if current_project in (None, project_id):
                        continue
                    candidates = view.candidates(namespace, space)
                    if candidates is None:
                        continue
                    threshold = calibrated_threshold(CROSS_VALIDATION_THRESHOLD, space)
                    for remote_id, similarity in candidates.above(embedding, threshold):
                        _merged, applied = merge_cross_validated_entry(
                            backend, entry.id, project_id, similarity, namespace=entry.namespace
                        )
                        if applied:
                            matched[entry.id] += 1
                        merge_cross_validated_entry(
                            view.backend(), remote_id, str(current_project), similarity, namespace=namespace
                        )
    return matched
