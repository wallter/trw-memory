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
- ``apply_cross_project_validation`` — top-level orchestrator that walks
  sibling project stores, computes embeddings similarity, and applies
  validations bidirectionally.

Extracted as PRD-DIST-245 Phase 2 batch 93.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

CROSS_VALIDATION_THRESHOLD = 0.92
CANDIDATE_LIMIT = 500

_ENTRY_UPDATE_LOCKS: dict[tuple[str, str], threading.Lock] = {}
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
) -> tuple[MemoryEntry | None, bool]:
    """Atomically append a single project's validation and boost once.

    The thread lock prevents same-process races; the file-backed guard closes
    the remaining gap where two separate processes open the same store
    concurrently. Looks up ``apply_importance_boost`` via the parent
    ``graph`` module for the test-monkeypatch indirection pattern.
    """
    from trw_memory import graph as _graph_module

    with entry_update_lock(backend, entry_id), backend_update_guard(backend):
        current = backend.get(entry_id)
        if current is None:
            return None, False
        if entry_has_cross_validation(current, project_id):
            return current, False

        updated = append_cross_validation(current, project_id, similarity)
        updated = _graph_module.apply_importance_boost(updated)
        persist_cross_validated_entry(backend, current, updated)
        reloaded = backend.get(entry_id)
        return (reloaded or updated), True


def apply_cross_project_validation(
    entry: MemoryEntry,
    backend: StorageBackend,
    conn: sqlite3.Connection,
    *,
    embedding: list[float] | None = None,
    config: MemoryConfig | None = None,
) -> int:
    """Cross-validate against sibling project stores when embeddings exist.

    Package-local cross-project evidence comes from sibling on-disk project
    namespaces. This keeps the feature usable without waiting on a platform-
    side embedding feed while still failing closed when embeddings are
    unavailable.

    Looks up ``detect_cross_validation`` and ``_safe_cosine_similarity`` via
    the parent ``graph`` module so test monkeypatches propagate.
    """
    if embedding is None:
        return 0

    current_project = project_scope_key(entry.namespace)
    if current_project is None:
        return 0

    from trw_memory import graph as _graph_module
    from trw_memory.integrations._backend import discover_namespace_backends

    cfg = config or MemoryConfig()
    matched_projects = 0
    current_entry = entry

    with discover_namespace_backends(cfg) as stores:
        for namespaces, remote_backend in stores:
            project_namespaces = [
                namespace
                for namespace in namespaces
                if (project_id := project_scope_key(namespace)) is not None and project_id != current_project
            ]
            for namespace in project_namespaces:
                project_id = project_scope_key(namespace)
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
                    (candidate.id, project_id, remote_embedding) for candidate, remote_embedding in remote_candidates
                ]
                if not _graph_module.detect_cross_validation(
                    current_entry,
                    conn,
                    embedding=embedding,
                    remote_entries=remote_payload,
                ):
                    continue

                for remote_entry, remote_embedding in remote_candidates:
                    similarity = _graph_module._safe_cosine_similarity(embedding, remote_embedding)
                    if similarity <= CROSS_VALIDATION_THRESHOLD:
                        continue

                    merged_entry, applied = merge_cross_validated_entry(
                        backend,
                        entry.id,
                        project_id,
                        similarity,
                    )
                    if merged_entry is not None:
                        current_entry = merged_entry
                    if applied:
                        matched_projects += 1

                    merge_cross_validated_entry(
                        remote_backend,
                        remote_entry.id,
                        current_project,
                        similarity,
                    )

    return matched_projects
