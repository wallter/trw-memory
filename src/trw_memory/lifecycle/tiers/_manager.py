"""TierManager: Hot/Warm/Cold tier orchestrator for memory entry lifecycle.

Composes WarmTierStore and ColdTierStore, manages the hot LRU cache directly,
and orchestrates lifecycle sweep transitions via execute_sweep().
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.lifecycle.tiers._cold import ColdTierStore
from trw_memory.lifecycle.tiers._scoring import TierSweepResult, compute_importance_score
from trw_memory.lifecycle.tiers._sweep import execute_sweep
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.encryption import derive_namespace_key
from trw_memory.security.keys import get_master_key
from trw_memory.storage.persistence import read_yaml

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from trw_memory.storage.interface import StorageBackend


def _entry_matches_tokens(entry: dict[str, object], query_tokens: list[str]) -> bool:
    """Return whether any token matches the entry text surface."""
    if not query_tokens:
        return True
    content = str(entry.get("content", "")).lower()
    detail = str(entry.get("detail", "")).lower()
    raw_tags = entry.get("tags", [])
    tag_text = " ".join(str(tag).lower() for tag in raw_tags) if isinstance(raw_tags, list) else ""
    entry_id = str(entry.get("id", "")).lower()
    haystack = f"{entry_id} {content} {detail} {tag_text}"
    return any(token in haystack for token in query_tokens)


class TierManager:
    """Hot/Warm/Cold tier manager for memory entry lifecycle.

    Hot tier: in-memory LRU cache (OrderedDict, O(1) ops).
    Warm tier: sqlite-vec backed persistent index with JSONL sidecar fallback.
    Cold tier: YAML archive partitioned by {YYYY}/{MM}/.

    Usage::

        mgr = TierManager(base_dir=Path(".memory"))
        entry = mgr.hot_get("some-id")
        mgr.hot_put("some-id", memory_entry)
        result = mgr.sweep()
    """

    def __init__(
        self,
        base_dir: Path,
        config: MemoryConfig | None = None,
        entries_dir: Path | None = None,
        namespace: str = "default",
    ) -> None:
        """Initialise TierManager.

        Args:
            base_dir: Base directory for memory storage.
            config: MemoryConfig for capacity/TTL settings.
            entries_dir: Optional explicit entries directory for sweep.
                         Defaults to base_dir / "entries".
        """
        self._base_dir = base_dir
        self._config = config or MemoryConfig()
        self._entries_dir: Path = entries_dir or (base_dir / "entries")
        self._namespace = namespace

        # Hot tier: OrderedDict used as LRU cache
        # LRU invariant: MRU at the end (rightmost), LRU at the front (leftmost)
        self._hot: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._hot_lock = threading.Lock()

        # Composed tier stores
        self._warm_store = WarmTierStore(base_dir)
        self._cold_store = ColdTierStore(base_dir, self._warm_store)

    def update_config(self, config: MemoryConfig) -> None:
        """Refresh the active config for call-time policy overrides."""
        self._config = config

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Release all resources held by this TierManager."""
        self._hot.clear()
        self._warm_store.close()

    def __enter__(self) -> TierManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    # -----------------------------------------------------------------------
    # Hot Tier
    # -----------------------------------------------------------------------

    def hot_get(self, entry_id: str) -> MemoryEntry | None:
        """Return a cached entry, moving it to MRU position on hit.

        Args:
            entry_id: Memory entry identifier.

        Returns:
            MemoryEntry if in cache, None otherwise.
        """
        with self._hot_lock:
            if entry_id not in self._hot:
                return None
            self._hot.move_to_end(entry_id)
            entry = self._hot[entry_id]
            entry.last_accessed_at = datetime.now(timezone.utc)
            return entry

    def hot_put(self, entry_id: str, entry: MemoryEntry) -> None:
        """Add or refresh an entry in the hot cache.

        Evicts the LRU entry when capacity is exceeded.

        Args:
            entry_id: Memory entry identifier.
            entry: MemoryEntry to cache.
        """
        cfg = self._config

        with self._hot_lock:
            if entry_id in self._hot:
                self._hot.move_to_end(entry_id)
                self._hot[entry_id] = entry
                return

            self._hot[entry_id] = entry
            self._hot.move_to_end(entry_id)

            # Evict LRU if over capacity
            if len(self._hot) > cfg.hot_max_entries:
                evicted_id = next(iter(self._hot))
                evicted_entry = self._hot[evicted_id]
                # The hot tier is only an acceleration layer. When capacity forces
                # an eviction, the entry must be demoted into warm storage so the
                # tier transition preserves data instead of silently dropping it.
                try:
                    self.warm_add(evicted_id, evicted_entry.model_dump(mode="json"), None)
                except (OSError, ValueError):
                    self._hot.pop(entry_id, None)
                    logger.warning(
                        "hot_tier_evict_deferred",
                        evicted_id=evicted_id,
                        rejected_entry_id=entry_id,
                        reason="warm_add_failed",
                        capacity=cfg.hot_max_entries,
                        exc_info=True,
                    )
                    return
                self._hot.popitem(last=False)
                logger.debug(
                    "hot_tier_evict",
                    evicted_id=evicted_id,
                    capacity=cfg.hot_max_entries,
                )

    def hot_clear(self) -> None:
        """Evict all entries from the hot cache (for testing / shutdown)."""
        with self._hot_lock:
            self._hot.clear()

    def hot_remove(self, entry_id: str) -> None:
        """Delete an entry from the hot cache without touching lower tiers."""
        with self._hot_lock:
            self._hot.pop(entry_id, None)

    @property
    def hot_size(self) -> int:
        """Number of entries currently in the hot cache."""
        with self._hot_lock:
            return len(self._hot)

    def hot_search(
        self,
        query_tokens: list[str],
        *,
        tags: list[str] | None = None,
        top_k: int = 25,
    ) -> list[dict[str, object]]:
        """Search the in-memory hot tier without touching disk."""
        tag_set = set(tags or [])
        with self._hot_lock:
            entries = [entry.model_dump(mode="json") for entry in self._hot.values()]

        scored: list[dict[str, object]] = []
        for item in entries:
            item_tags = item.get("tags", [])
            if tag_set and (not isinstance(item_tags, list) or not tag_set.issubset({str(tag) for tag in item_tags})):
                continue
            if not _entry_matches_tokens(item, query_tokens):
                continue
            enriched = dict(item)
            enriched["score"] = compute_importance_score(enriched, query_tokens, config=self._config)
            scored.append(enriched)

        scored.sort(key=lambda entry: float(str(entry.get("score", 0.0))), reverse=True)
        return scored[:top_k]

    def warmup_hot_from_warm(self, *, max_entries: int | None = None) -> int:
        """Populate the hot tier from the highest-utility warm-tier entries."""
        if self.hot_size > 0:
            return 0

        target = max_entries or self._config.hot_max_entries
        warm_entries = self._warm_store.warm_entries()
        if not warm_entries:
            return 0

        ranked = sorted(
            warm_entries,
            key=lambda entry: compute_importance_score(entry, [], config=self._config),
            reverse=True,
        )

        loaded = 0
        for item in ranked[:target]:
            try:
                entry = MemoryEntry.model_validate(item)
            except Exception:
                logger.warning("tier_warmup_invalid_sidecar_entry", exc_info=True)
                continue
            self.hot_put(entry.id, entry)
            loaded += 1
        return loaded

    def warmup_hot_from_entries(
        self,
        entries: list[MemoryEntry],
        *,
        mirror_to_warm: bool = False,
        max_entries: int | None = None,
    ) -> int:
        """Fallback warmup path for pre-existing stores that predate the warm sidecar."""
        if self.hot_size > 0:
            return 0

        target = max_entries or self._config.hot_max_entries
        ranked = sorted(
            entries,
            key=lambda entry: compute_importance_score(entry.model_dump(mode="json"), [], config=self._config),
            reverse=True,
        )

        loaded = 0
        for entry in ranked[:target]:
            self.hot_put(entry.id, entry)
            if mirror_to_warm:
                self.warm_add(entry.id, entry.model_dump(mode="json"), None)
            loaded += 1
        return loaded

    # -----------------------------------------------------------------------
    # Warm Tier (delegated)
    # -----------------------------------------------------------------------

    def _warm_db_path(self) -> Path:
        """Resolve path to warm.db (delegates to WarmTierStore)."""
        return self._warm_store._warm_db_path()

    def _warm_sidecar_path(self) -> Path:
        """Path to the warm tier keyword-search sidecar (JSONL)."""
        return self._warm_store._warm_sidecar_path()

    def warm_add(
        self,
        entry_id: str,
        entry_data: dict[str, object],
        embedding: list[float] | None,
    ) -> None:
        """Insert or replace an entry in the warm store."""
        self._warm_store.warm_add(entry_id, entry_data, embedding)

    def warm_remove(self, entry_id: str) -> bool:
        """Delete an entry from the warm store and sidecar."""
        return self._warm_store.warm_remove(entry_id)

    def warm_search(
        self,
        query_tokens: list[str],
        query_embedding: list[float] | None,
        top_k: int = 25,
    ) -> list[dict[str, object]]:
        """Search the warm tier for relevant entries."""
        raw_hits = self._warm_store.warm_search(query_tokens, query_embedding, max(top_k * 2, top_k))
        ranked = sorted(
            (
                dict(
                    entry,
                    score=compute_importance_score(
                        entry,
                        query_tokens,
                        query_embedding=query_embedding,
                        config=self._config,
                        relevance_hint=(
                            float(str(entry.get("_tier_relevance", entry.get("score"))))
                            if entry.get("_tier_relevance") is not None or entry.get("score") is not None
                            else None
                        ),
                    ),
                )
                for entry in raw_hits
            ),
            key=lambda entry: float(str(entry["score"])),
            reverse=True,
        )
        return ranked[:top_k]

    # -----------------------------------------------------------------------
    # Cold Tier (delegated)
    # -----------------------------------------------------------------------

    def _cold_dir(self) -> Path:
        """Base cold archive directory."""
        return self._cold_store._cold_dir()

    def cold_archive(self, entry_id: str, entry_path: Path) -> None:
        """Move a warm-tier YAML entry to the cold archive partition."""
        self._cold_store.cold_archive(entry_id, entry_path)

    def cold_archive_entry(self, entry_id: str, entry_data: dict[str, object]) -> None:
        """Archive an active canonical entry into the cold tier."""
        with self._open_canonical_backend(self._config) as backend:
            self._cold_store.cold_archive_entry(
                entry_id,
                entry_data,
                delete_source_entry_fn=backend.delete,
                verify_source_entry_removed_fn=lambda archived_entry_id: backend.get(archived_entry_id) is None,
            )

    def cold_promote(
        self,
        entry_id: str,
        *,
        restore_entry_fn: Callable[[dict[str, object]], None] | None = None,
        delete_restored_entry_fn: Callable[[str], bool | None] | None = None,
        force_delete_restored_entry_fn: Callable[[str], bool | None] | None = None,
        verify_restored_entry_removed_fn: Callable[[str], bool] | None = None,
    ) -> dict[str, object] | None:
        """Move a cold-tier entry back to warm tier on access."""
        return self._cold_store.cold_promote(
            entry_id,
            restore_entry_fn=restore_entry_fn,
            delete_restored_entry_fn=delete_restored_entry_fn,
            force_delete_restored_entry_fn=force_delete_restored_entry_fn,
            verify_restored_entry_removed_fn=verify_restored_entry_removed_fn,
        )

    def cold_search(
        self,
        query_tokens: list[str],
        *,
        promote: bool = False,
        top_k: int | None = None,
        restore_entry_fn: Callable[[dict[str, object]], None] | None = None,
        delete_restored_entry_fn: Callable[[str], bool | None] | None = None,
        force_delete_restored_entry_fn: Callable[[str], bool | None] | None = None,
        verify_restored_entry_removed_fn: Callable[[str], bool] | None = None,
    ) -> list[dict[str, object]]:
        """Linear scan of the cold archive for keyword matches."""
        return self._cold_store.cold_search(
            query_tokens,
            promote=promote,
            top_k=top_k,
            restore_entry_fn=restore_entry_fn,
            delete_restored_entry_fn=delete_restored_entry_fn,
            force_delete_restored_entry_fn=force_delete_restored_entry_fn,
            verify_restored_entry_removed_fn=verify_restored_entry_removed_fn,
        )

    def search(
        self,
        query_tokens: list[str],
        *,
        query_embedding: list[float] | None = None,
        tags: list[str] | None = None,
        top_k: int = 25,
        restore_entry_fn: Callable[[dict[str, object]], None] | None = None,
        delete_restored_entry_fn: Callable[[str], bool | None] | None = None,
        force_delete_restored_entry_fn: Callable[[str], bool | None] | None = None,
        verify_restored_entry_removed_fn: Callable[[str], bool] | None = None,
    ) -> list[dict[str, object]]:
        """Search hot, warm, and cold tiers as one merged runtime surface."""
        warm_hits = self.warm_search(query_tokens, query_embedding, top_k=max(top_k * 2, top_k))
        cold_hits = self._cold_store.cold_search(
            query_tokens,
            promote=bool(query_tokens),
            top_k=max(top_k * 2, top_k),
            restore_entry_fn=restore_entry_fn,
            delete_restored_entry_fn=delete_restored_entry_fn,
            force_delete_restored_entry_fn=force_delete_restored_entry_fn,
            verify_restored_entry_removed_fn=verify_restored_entry_removed_fn,
        )
        hot_hits = self.hot_search(query_tokens, tags=tags, top_k=max(top_k * 2, top_k))

        tag_set = set(tags or [])
        merged: dict[str, dict[str, object]] = {}
        for source_hits in (hot_hits, warm_hits, cold_hits):
            for item in source_hits:
                entry_id = str(item.get("id", ""))
                if not entry_id:
                    continue
                item_tags = item.get("tags", [])
                if tag_set and (
                    not isinstance(item_tags, list) or not tag_set.issubset({str(tag) for tag in item_tags})
                ):
                    continue
                merged.setdefault(entry_id, item)

        ranked = sorted(
            (
                dict(
                    entry,
                    score=compute_importance_score(
                        entry,
                        query_tokens,
                        query_embedding=query_embedding,
                        config=self._config,
                        relevance_hint=(
                            float(str(entry.get("_tier_relevance")))
                            if entry.get("_tier_relevance") is not None
                            else None
                        ),
                    ),
                )
                for entry in merged.values()
            ),
            key=lambda entry: float(str(entry["score"])),
            reverse=True,
        )
        return ranked[:top_k]

    # -----------------------------------------------------------------------
    # Sweep (delegated)
    # -----------------------------------------------------------------------

    def sweep(self, config: MemoryConfig | None = None) -> TierSweepResult:
        """Execute lifecycle sweep across all tiers.

        Performs three transition checks in order:
        1. Hot -> Warm: entries whose last_accessed_at exceeds hot_ttl_days.
        2. Warm -> Cold: entries idle > cold_threshold_days with importance < 0.22.
        3. Cold -> Purge: entries idle > retention_days with importance < 0.1.

        All thresholds are read from config at call time.
        Per-entry failures are logged and counted in ``errors``.

        Returns:
            TierSweepResult with counts of promoted, demoted, purged, and errors.
        """
        active_config = config or MemoryConfig()
        self._config = active_config
        warm_entries, preload_errors = self._load_warm_entries(active_config)
        result = execute_sweep(
            hot=self._hot,
            config=active_config,
            warm_entries=warm_entries,
            base_dir=self._base_dir,
            warm_add_fn=self.warm_add,
            cold_archive_entry_fn=self.cold_archive_entry,
            cold_dir=self._cold_dir(),
        )
        if preload_errors == 0:
            return result
        return TierSweepResult(
            promoted=result.promoted,
            demoted=result.demoted,
            purged=result.purged,
            errors=result.errors + preload_errors,
        )

    def _open_canonical_backend(self, config: MemoryConfig) -> StorageBackend:
        db_path = self._base_dir / config.sqlite_db_name
        if config.storage_backend == "sqlite" and db_path.exists():
            from trw_memory.storage.sqlite_backend import SQLiteBackend

            sqlcipher_key_hex: str | None = None
            if config.encryption_enabled:
                master_key = get_master_key(config)
                sqlcipher_key_hex = derive_namespace_key(master_key, self._namespace)

            return SQLiteBackend(
                db_path,
                dim=config.embedding_dim,
                sqlcipher_key_hex=sqlcipher_key_hex,
                recovery_policy=config.memory_recovery_policy,
                corrupt_backup_keep=config.memory_corrupt_backup_keep,
                rebuild_from_cold=config.memory_recovery_rebuild_from_cold,
            )

        from trw_memory.storage.yaml_backend import YAMLBackend

        return YAMLBackend(self._entries_dir)

    def _load_warm_entries(self, config: MemoryConfig) -> tuple[list[dict[str, object]], int]:
        db_path = self._base_dir / config.sqlite_db_name
        if config.storage_backend == "sqlite" and db_path.exists():
            try:
                with self._open_canonical_backend(config) as backend:
                    backend_entries = backend.list_entries(limit=max(backend.count(), config.hot_max_entries * 8, 200))
                return [entry.model_dump(mode="json") for entry in backend_entries], 0
            except (OSError, StorageError, ValueError):
                logger.warning("tier_sweep_backend_scan_failed", namespace=self._namespace, exc_info=True)
                return [], 1

        if not self._entries_dir.exists():
            return [], 0

        entries: list[dict[str, object]] = []
        errors = 0
        for yaml_file in sorted(self._entries_dir.glob("*.yaml")):
            if yaml_file.name == "index.yaml":
                continue
            try:
                entries.append(read_yaml(yaml_file))
            except (OSError, StorageError, ValueError):
                logger.warning("tier_sweep_entry_scan_failed", path=str(yaml_file), exc_info=True)
                errors += 1
        return entries, errors
