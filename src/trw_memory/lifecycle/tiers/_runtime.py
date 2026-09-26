"""Runtime helpers for wiring TierManager into package entry points."""

from __future__ import annotations

import atexit
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import cast, overload

import structlog
from typing_extensions import TypedDict

from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.lifecycle.tiers._manager import TierManager
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.retrieval.recall_selection import LocalCandidate, RecallInvocation
from trw_memory.security.namespace_scope import NamespaceScopeError
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

# Bounded LRU of process-local TierManagers. Each manager holds an open
# SQLite connection (its warm-tier backend), so an unbounded cache leaks one
# connection per distinct (storage_path, backend, namespace) key forever. On
# overflow we close() the evicted (least-recently-used) manager before dropping
# it so the connection is released.
_TIER_MANAGER_CACHE_MAX = 32
_TIER_MANAGER_CACHE: OrderedDict[tuple[str, str, str], TierManager] = OrderedDict()
# PRD-CORE-279 FR04: the served tool bodies now run in a worker pool, so two
# requests can reach this cache at once. The lock is REENTRANT and is held
# across acquisition AND use, not just the dictionary mutation: a manager owns
# an open warm-tier SQLite connection that eviction closes, so a lock released
# at the end of the lookup would let one worker close the very manager another
# worker is mid-search on. Serialising the tier phase is the cheap, provable
# answer; the expensive phases (BM25, dense search, the primary backend) stay
# parallel because they hold no shared object.
_TIER_MANAGER_CACHE_LOCK = threading.RLock()


def namespace_storage_dir(config: MemoryConfig, namespace: str) -> Path:
    """Resolve the on-disk directory that owns a namespace's tier files."""
    return Path(config.storage_path).resolve() / namespace.replace(":", "_")


def supports_tier_runtime(backend: object) -> bool:
    """Return whether this backend is a concrete trw-memory storage backend."""
    return backend.__class__.__module__.startswith("trw_memory.storage.")


def tier_runtime_enabled(config: MemoryConfig) -> bool:
    """Return whether the Hot/Warm/Cold runtime can persist safely for this config."""
    return not config.encryption_enabled


def embedding_has_consumer(config: MemoryConfig, backend: StorageBackend) -> bool:
    """Return whether a freshly computed dense embedding has any live sink.

    Store paths compute an embedding purely to persist/search it. A dense
    vector has exactly three potential sinks on the write path:

    1. the primary backend's vector store (``backend.upsert_vector``);
    2. the warm tier's own ``sqlite-vec`` sidecar, which is an independent
       SQLite store and therefore usable even when the *primary* backend
       cannot persist vectors (e.g. a YAML primary with ``sqlite-vec``
       installed) — gated by :func:`tier_runtime_enabled`;
    3. graph similarity edges, which read candidate vectors back from the
       primary backend and so are already covered by ``supports_vectors``.

    Remote publish is not one: no vector leaves the machine (PRD-CORE-302 FR04).

    When none of these are live, computing the embedding is pure waste —
    every downstream ``upsert_vector`` would no-op. Returning ``True`` is the
    conservative answer: it only ever asks the caller to keep doing the work
    it already does today, so this never regresses existing behaviour.

    Args:
        config: Active memory configuration (tier settings).
        backend: The primary storage backend for this write.

    Returns:
        ``True`` when at least one vector sink can consume the embedding.
    """
    if backend.supports_vectors():
        return True
    return tier_runtime_enabled(config)


def get_tier_manager(config: MemoryConfig, namespace: str) -> TierManager:
    """Return the process-local TierManager for a namespace."""
    key = (str(Path(config.storage_path).resolve()), config.storage_backend, namespace)
    with _TIER_MANAGER_CACHE_LOCK:
        manager = _TIER_MANAGER_CACHE.get(key)
        if manager is None:
            manager = TierManager(base_dir=namespace_storage_dir(config, namespace), config=config, namespace=namespace)
            _TIER_MANAGER_CACHE[key] = manager
            # Evict the least-recently-used managers if we're over the cap,
            # closing each one first so its SQLite connection is released.
            while len(_TIER_MANAGER_CACHE) > _TIER_MANAGER_CACHE_MAX:
                evicted_key, evicted_manager = _TIER_MANAGER_CACHE.popitem(last=False)
                try:
                    evicted_manager.close()
                except Exception:
                    logger.warning("tier_manager_cache_evict_close_failed", cache_key=evicted_key, exc_info=True)
        else:
            manager.update_config(config)
            # Mark as most-recently-used.
            _TIER_MANAGER_CACHE.move_to_end(key)
        return manager


def reset_tier_manager_cache() -> None:
    """Close and drop every cached TierManager.

    The cache is process-lifetime by design -- in production the process is a
    server. A test process has many logical lifetimes in one real one, and the
    cache key is the CONFIGURED storage path, so two tests pointing that path at
    different temporary stores share entries and resolve one store's ids against
    the other's backend. Closing on the way out also releases each manager's
    warm-tier SQLite connection instead of leaking it.
    """
    with _TIER_MANAGER_CACHE_LOCK:
        managers = list(_TIER_MANAGER_CACHE.items())
        _TIER_MANAGER_CACHE.clear()
    for cache_key, manager in managers:
        try:
            manager.close()
        except Exception:  # justified: a best-effort teardown must not mask the caller's failure
            logger.warning("tier_manager_cache_reset_close_failed", cache_key=cache_key, exc_info=True)


# A cached manager holds its warm.db connection for the life of the process; nothing
# else closes it. Close them at exit, as the graph worker pool does its backends, so
# SQLite checkpoints and removes warm.db-wal instead of leaving it for the next open
# (W37: a 7.5 MB warm.db-wal survived every benchmark process).
atexit.register(reset_tier_manager_cache)


def warmup_tier_manager(
    config: MemoryConfig,
    namespace: str,
    backend: StorageBackend,
) -> TierManager:
    """Ensure the namespace tier manager has a usable hot cache."""
    with _TIER_MANAGER_CACHE_LOCK:
        manager = get_tier_manager(config, namespace)
        warmed = manager.warmup_hot_from_warm()
        if warmed > 0 or manager.hot_size > 0:
            return manager

        # Existing stores may predate the warm sidecar entirely. Seeding the
        # hottest current backend rows into both hot and warm gives the tier
        # runtime a migration path without forcing users to rewrite their store
        # first.
        try:
            entries = backend.list_entries(
                namespace=namespace, limit=max(config.hot_max_entries * 8, 200), status=MemoryStatus.ACTIVE
            )
        except Exception:
            logger.warning("tier_warmup_backend_scan_failed", namespace=namespace, exc_info=True)
            return manager

        manager.warmup_hot_from_entries(entries, mirror_to_warm=True)
        return manager


def remember_entry_in_tiers(
    config: MemoryConfig,
    namespace: str,
    entry: MemoryEntry,
    embedding: list[float] | None = None,
    provenance: VectorProvenance | None = None,
) -> None:
    """Mirror a freshly written entry into the runtime tier system.

    *provenance* is the embedding's generation record; a warm vector without it
    is kept for keyword search but never dense-scored.
    """
    if not tier_runtime_enabled(config):
        return
    with _TIER_MANAGER_CACHE_LOCK:
        manager = get_tier_manager(config, namespace)
        manager.hot_put(entry.id, entry)
        try:
            manager.warm_add(entry.id, entry.model_dump(mode="json"), embedding, provenance=provenance)
        except (OSError, ValueError):
            logger.warning("tier_warm_mirror_failed", namespace=namespace, entry_id=entry.id, exc_info=True)


def remember_entries_data_in_tiers(config: MemoryConfig, payloads: list[dict[str, object]]) -> None:
    """Mirror several serialized entry payloads into the tiers, one warm write per namespace.

    Recall calls this with every returned row so the warm sidecar sees a
    fresh ``last_accessed_at``; doing it per entry made recall latency scale
    with ``limit * sidecar_rows``. Hot-tier order is the same as one
    :func:`remember_entry_in_tiers` per entry; the warm sidecar ends up identical.
    """
    if not tier_runtime_enabled(config) or not payloads:
        return
    by_namespace: dict[str, list[MemoryEntry]] = {}
    for entry_data in payloads:
        try:
            entry = MemoryEntry.model_validate(entry_data)
        except Exception:
            logger.warning("tier_entry_validation_failed", namespace=entry_data.get("namespace", ""), exc_info=True)
            continue
        by_namespace.setdefault(entry.namespace, []).append(entry)
    for namespace, entries in by_namespace.items():
        # Held across put, warm write and drop, as remember_entry_in_tiers holds
        # it: a concurrent single-entry write to an evictee id could otherwise be
        # reverted by the deferred hot_drop and the batch's stale warm snapshot.
        with _TIER_MANAGER_CACHE_LOCK:
            manager = get_tier_manager(config, namespace)
            # Evictees stay in hot until warm has them: a failed warm write must not
            # lose them from both tiers (release-verify 2026-09-17 B-1).
            evicted = manager.hot_put_many([(entry.id, entry) for entry in entries], drop_evictees=False)
            # Evictees first, then the recalled rows, so a row that was both
            # evicted and recalled ends up with its fresh payload (last write wins).
            items: list[tuple[str, dict[str, object], list[float] | None]] = [
                (evicted_id, evicted_data, None) for evicted_id, evicted_data in evicted
            ]
            items.extend((entry.id, entry.model_dump(mode="json"), None) for entry in entries)
            try:
                manager.warm_add_many(items)
            except (OSError, ValueError):
                logger.warning("tier_warm_mirror_failed", namespace=namespace, entry_count=len(items), exc_info=True)
                continue
            manager.hot_drop([evicted_id for evicted_id, _ in evicted])


def remove_entry_from_tiers(config: MemoryConfig, namespace: str, entry_id: str) -> None:
    """Delete an entry from EVERY runtime tier (hot, warm, and cold).

    Erasure / GDPR ``forget`` flows rely on this removing the entry from all
    tiers. Cold-tier deletion is required because :func:`cold_archive` moves an
    entry out of the canonical backend into the YAML archive — without scanning
    cold, a forgotten entry could survive there permanently.
    """
    if not tier_runtime_enabled(config):
        return
    with _TIER_MANAGER_CACHE_LOCK:
        manager = get_tier_manager(config, namespace)
        manager.hot_remove(entry_id)
        try:
            manager.warm_remove(entry_id)
        except (OSError, ValueError):
            logger.warning("tier_warm_remove_failed", namespace=namespace, entry_id=entry_id, exc_info=True)
        try:
            manager.cold_remove(entry_id)
        except OSError:
            logger.warning("tier_cold_remove_failed", namespace=namespace, entry_id=entry_id, exc_info=True)


@overload
def tier_candidates(
    config: MemoryConfig,
    namespace: str,
    backend: StorageBackend,
    *,
    query: str,
    tags: list[str] | None,
    limit: int,
    query_embedding: list[float] | None = None,
    query_space: EmbeddingSpace | None = None,
    invocation: None = None,
    covered_ids: frozenset[str] = frozenset(),
) -> list[dict[str, object]]: ...


@overload
def tier_candidates(
    config: MemoryConfig,
    namespace: str,
    backend: StorageBackend,
    *,
    query: str,
    tags: list[str] | None,
    limit: int,
    query_embedding: list[float] | None = None,
    query_space: EmbeddingSpace | None = None,
    invocation: RecallInvocation,
    covered_ids: frozenset[str] = frozenset(),
) -> list[LocalCandidate]: ...


def tier_candidates(
    config: MemoryConfig,
    namespace: str,
    backend: StorageBackend,
    *,
    query: str,
    tags: list[str] | None,
    limit: int,
    query_embedding: list[float] | None = None,
    query_space: EmbeddingSpace | None = None,
    invocation: RecallInvocation | None = None,
    covered_ids: frozenset[str] = frozenset(),
) -> list[dict[str, object]] | list[LocalCandidate]:
    """Collect full-entry candidates from the tier runtime.

    *covered_ids* (discovery mode only) names primary rows the caller already
    ranked for this query; see :meth:`TierManager.search`.
    """
    if not tier_runtime_enabled(config):
        return []
    with _TIER_MANAGER_CACHE_LOCK:
        manager = (
            get_tier_manager(config, namespace)
            if invocation is not None
            else warmup_tier_manager(config, namespace, backend)
        )
        query_tokens = [token for token in query.lower().split() if token]

        found = manager.search(
            query_tokens,
            query_embedding=query_embedding,
            query_space=query_space,
            tags=tags,
            top_k=max(limit * 2, config.hot_max_entries),
            invocation=invocation,
            resolve_entry=lambda entry_id: backend.get(entry_id, namespace=namespace),
            covered_ids=covered_ids,
            **_restoration_callbacks(config, namespace, backend),
        )
    if invocation is not None:
        return found  # discovery admits only active canonical rows (RecallInvocation.allows_entry)
    # The mirror keeps a row as it was when stored or recalled; recall is active-only
    # (PRD-CORE-294 FR03), so a row retired since then must not come back through it.
    rows = cast("list[dict[str, object]]", found)
    return [row for row in rows if row.get("status", MemoryStatus.ACTIVE.value) == MemoryStatus.ACTIVE.value]


class _RestorationCallbacks(TypedDict):
    restore_entry_fn: Callable[[dict[str, object]], None]
    delete_restored_entry_fn: Callable[[str], bool | None]
    force_delete_restored_entry_fn: Callable[[str], bool | None]
    verify_restored_entry_removed_fn: Callable[[str], bool]


def _restoration_callbacks(config: MemoryConfig, namespace: str, backend: StorageBackend) -> _RestorationCallbacks:
    def _restore_entry(entry_data: dict[str, object]) -> None:
        entry = MemoryEntry.model_validate(entry_data)
        if entry.namespace != namespace:
            raise NamespaceScopeError("cold restoration outside authorized namespace")
        backend.store(entry)

    def _delete_restored_entry(entry_id: str) -> bool | None:
        return backend.delete(entry_id, namespace=namespace)

    def _force_delete_restored_entry(entry_id: str) -> bool | None:
        with create_backend_from_config(config, namespace) as rollback_backend:
            return rollback_backend.delete(entry_id, namespace=namespace)

    def _verify_restored_entry_removed(entry_id: str) -> bool:
        with create_backend_from_config(config, namespace) as verification_backend:
            return verification_backend.get(entry_id, namespace=namespace) is None

    return {
        "restore_entry_fn": _restore_entry,
        "delete_restored_entry_fn": _delete_restored_entry,
        "force_delete_restored_entry_fn": _force_delete_restored_entry,
        "verify_restored_entry_removed_fn": _verify_restored_entry_removed,
    }


def restore_selected_cold(
    config: MemoryConfig,
    namespace: str,
    backend: StorageBackend,
    selected: list[LocalCandidate],
) -> set[tuple[str, str]]:
    """Restore returned cold hits only; existing rollback is per entry, not per call."""
    with _TIER_MANAGER_CACHE_LOCK:
        failed: set[tuple[str, str]] = set()
        manager = get_tier_manager(config, namespace)
        for candidate in selected:
            if candidate.cold:
                if candidate.entry.namespace != namespace:
                    raise NamespaceScopeError("cold restoration outside authorized namespace")
                if (
                    manager.cold_promote(candidate.entry.id, **_restoration_callbacks(config, namespace, backend))
                    is None
                ):
                    failed.add((namespace, candidate.entry.id))
        return failed
