"""Runtime helpers for wiring TierManager into package entry points."""

from __future__ import annotations

import threading
from pathlib import Path

import structlog

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.lifecycle.tiers._manager import TierManager
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

_TIER_MANAGER_CACHE: dict[tuple[str, str, str], TierManager] = {}
_TIER_MANAGER_CACHE_LOCK = threading.Lock()


def namespace_storage_dir(config: MemoryConfig, namespace: str) -> Path:
    """Resolve the on-disk directory that owns a namespace's tier files."""
    return Path(config.storage_path).resolve() / namespace.replace(":", "_")


def supports_tier_runtime(backend: object) -> bool:
    """Return whether this backend is a concrete trw-memory storage backend."""
    return backend.__class__.__module__.startswith("trw_memory.storage.")


def get_tier_manager(config: MemoryConfig, namespace: str) -> TierManager:
    """Return the process-local TierManager for a namespace."""
    key = (str(Path(config.storage_path).resolve()), config.storage_backend, namespace)
    with _TIER_MANAGER_CACHE_LOCK:
        manager = _TIER_MANAGER_CACHE.get(key)
        if manager is None:
            manager = TierManager(base_dir=namespace_storage_dir(config, namespace), config=config, namespace=namespace)
            _TIER_MANAGER_CACHE[key] = manager
        else:
            manager.update_config(config)
        return manager


def warmup_tier_manager(
    config: MemoryConfig,
    namespace: str,
    backend: StorageBackend,
) -> TierManager:
    """Ensure the namespace tier manager has a usable hot cache."""
    manager = get_tier_manager(config, namespace)
    warmed = manager.warmup_hot_from_warm()
    if warmed > 0 or manager.hot_size > 0:
        return manager

    # Existing stores may predate the warm sidecar entirely. Seeding the hottest
    # current backend rows into both hot and warm gives the tier runtime a
    # migration path without forcing users to rewrite their store first.
    try:
        entries = backend.list_entries(namespace=namespace, limit=max(config.hot_max_entries * 8, 200))
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
) -> None:
    """Mirror a freshly written entry into the runtime tier system."""
    manager = get_tier_manager(config, namespace)
    manager.hot_put(entry.id, entry)
    try:
        manager.warm_add(entry.id, entry.model_dump(mode="json"), embedding)
    except (OSError, ValueError):
        logger.warning("tier_warm_mirror_failed", namespace=namespace, entry_id=entry.id, exc_info=True)


def remember_entry_data_in_tiers(config: MemoryConfig, entry_data: dict[str, object]) -> None:
    """Mirror a serialized entry payload into the runtime tier system."""
    try:
        entry = MemoryEntry.model_validate(entry_data)
    except Exception:
        logger.warning("tier_entry_validation_failed", namespace=entry_data.get("namespace", ""), exc_info=True)
        return
    remember_entry_in_tiers(config, entry.namespace, entry)


def remove_entry_from_tiers(config: MemoryConfig, namespace: str, entry_id: str) -> None:
    """Delete an entry from the runtime tier system."""
    manager = get_tier_manager(config, namespace)
    manager.hot_remove(entry_id)
    try:
        manager.warm_remove(entry_id)
    except (OSError, ValueError):
        logger.warning("tier_warm_remove_failed", namespace=namespace, entry_id=entry_id, exc_info=True)


def tier_candidates(
    config: MemoryConfig,
    namespace: str,
    backend: StorageBackend,
    *,
    query: str,
    tags: list[str] | None,
    limit: int,
    query_embedding: list[float] | None = None,
) -> list[dict[str, object]]:
    """Collect full-entry candidates from the tier runtime."""
    manager = warmup_tier_manager(config, namespace, backend)
    query_tokens = [token for token in query.lower().split() if token]
    
    def _restore_entry(entry_data: dict[str, object]) -> None:
        backend.store(MemoryEntry.model_validate(entry_data))

    def _delete_restored_entry(entry_id: str) -> bool | None:
        return backend.delete(entry_id)

    def _force_delete_restored_entry(entry_id: str) -> bool | None:
        with create_backend_from_config(config, namespace) as rollback_backend:
            return rollback_backend.delete(entry_id)

    def _verify_restored_entry_removed(entry_id: str) -> bool:
        with create_backend_from_config(config, namespace) as verification_backend:
            return verification_backend.get(entry_id) is None

    return manager.search(
        query_tokens,
        query_embedding=query_embedding,
        tags=tags,
        top_k=max(limit * 2, config.hot_max_entries),
        restore_entry_fn=_restore_entry,
        delete_restored_entry_fn=_delete_restored_entry,
        force_delete_restored_entry_fn=_force_delete_restored_entry,
        verify_restored_entry_removed_fn=_verify_restored_entry_removed,
    )
