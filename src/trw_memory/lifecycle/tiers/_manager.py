"""TierManager: Hot/Warm/Cold tier orchestrator for memory entry lifecycle."""

from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from trw_memory.lifecycle.tiers._cold import ColdTierStore
from trw_memory.lifecycle.tiers._manager_io import load_warm_entries, open_canonical_backend
from trw_memory.lifecycle.tiers._manager_search import (
    discover_candidates,
    merge_search_results,
    rank_search_hits,
    search_hot_entries,
    warmup_hot_from_entries,
    warmup_hot_from_warm_entries,
)
from trw_memory.lifecycle.tiers._scoring import TierSweepResult
from trw_memory.lifecycle.tiers._sweep import execute_sweep
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.recall_selection import LocalCandidate, RecallInvocation

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
    from trw_memory.storage.interface import StorageBackend


class TierManager:
    """Hot/Warm/Cold tier manager for memory entry lifecycle."""

    def __init__(
        self,
        base_dir: Path,
        config: MemoryConfig | None = None,
        entries_dir: Path | None = None,
        namespace: str = "default",
    ) -> None:
        self._base_dir = base_dir
        self._config = config or MemoryConfig()
        self._entries_dir: Path = entries_dir or (base_dir / "entries")
        self._namespace = namespace

        self._hot: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._hot_lock = threading.Lock()

        self._warm_store = WarmTierStore(base_dir)
        self._cold_store = ColdTierStore(
            base_dir, self._warm_store, search_cache_max=self._config.cold_search_cache_max
        )

    def update_config(self, config: MemoryConfig) -> None:
        """Refresh the active config for call-time policy overrides."""
        self._config = config

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

    def hot_put(self, entry_id: str, entry: MemoryEntry) -> None:
        """Add or refresh an entry in the hot cache."""
        cfg = self._config

        with self._hot_lock:
            if entry_id in self._hot:
                self._hot.move_to_end(entry_id)
                self._hot[entry_id] = entry
                return

            self._hot[entry_id] = entry
            self._hot.move_to_end(entry_id)

            if len(self._hot) > cfg.hot_max_entries:
                evicted_id = next(iter(self._hot))
                evicted_entry = self._hot[evicted_id]
                try:
                    self.warm_add(evicted_id, evicted_entry.model_dump(mode="json"), None)
                except (OSError, ValueError):
                    # warm_add failed for the LRU evictee. Previously we popped
                    # ``entry_id`` (the just-written MRU entry) — that lost the new
                    # write AND left the LRU evictee in place, so overflow was never
                    # resolved (hot stayed at hot_max_entries + 1). Drop the LRU
                    # evictee instead: it was already selected for demotion, so
                    # removing it from hot resolves the overflow while preserving the
                    # freshly-written entry the caller just stored.
                    self._hot.pop(evicted_id, None)
                    logger.warning(
                        "hot_tier_evict_dropped",
                        evicted_id=evicted_id,
                        kept_entry_id=entry_id,
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

    def hot_put_many(
        self, items: list[tuple[str, MemoryEntry]], *, drop_evictees: bool = True
    ) -> list[tuple[str, dict[str, object]]]:
        """Add or refresh several entries; return the LRU evictees instead of demoting them.

        With ``drop_evictees=False`` the evictees are reported but left in the
        hot tier, so the caller can write them to warm first and then call
        :meth:`hot_drop`; a failed warm write then loses nothing, where popping
        first lost the whole batch from both tiers (release-verify 2026-09-17 B-1).

        ``hot_put`` demotes each evictee with its own ``warm_add`` (a full
        sidecar read-modify-write). A recall that mirrors ``limit`` results
        into a full hot tier evicts up to ``limit`` entries, so the caller
        pays ``limit`` more sidecar rewrites. This variant applies the same
        LRU policy in one pass and hands back ``(evicted_id, entry_data)``
        pairs so the caller can demote them together with its own batch via
        ``warm_add_many``. Hot-tier contents after the call equal sequential
        ``hot_put`` calls in the same order.
        """
        evicted: list[tuple[str, dict[str, object]]] = []
        with self._hot_lock:
            for entry_id, entry in items:
                self._hot[entry_id] = entry
                self._hot.move_to_end(entry_id)
            capacity = self._config.hot_max_entries
            surplus = len(self._hot) - capacity
            for evicted_id, evicted_entry in list(self._hot.items())[: max(surplus, 0)]:
                evicted.append((evicted_id, evicted_entry.model_dump(mode="json")))
                if drop_evictees:
                    del self._hot[evicted_id]
                logger.debug("hot_tier_evict", evicted_id=evicted_id, capacity=capacity)
        return evicted

    def hot_drop(self, entry_ids: list[str]) -> None:
        """Remove entries from the hot tier without demoting them (they are already in warm)."""
        with self._hot_lock:
            for entry_id in entry_ids:
                self._hot.pop(entry_id, None)

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
        with self._hot_lock:
            entries = [entry.model_dump(mode="json") for entry in self._hot.values()]
        return search_hot_entries(entries, query_tokens=query_tokens, tags=tags, top_k=top_k, config=self._config)

    def warmup_hot_from_warm(self, *, max_entries: int | None = None) -> int:
        """Populate the hot tier from the highest-utility warm-tier entries."""
        if self.hot_size > 0:
            return 0

        target = max_entries or self._config.hot_max_entries
        return warmup_hot_from_warm_entries(
            self._warm_store.warm_entries(),
            target=target,
            hot_put_fn=self.hot_put,
            config=self._config,
        )

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
        return warmup_hot_from_entries(
            entries,
            target=target,
            hot_put_fn=self.hot_put,
            config=self._config,
            mirror_to_warm_fn=self.warm_add if mirror_to_warm else None,
        )

    def _warm_db_path(self) -> Path:
        """Resolve path to warm.db."""
        return self._warm_store._warm_db_path()

    def _warm_sidecar_path(self) -> Path:
        """Path to the warm tier keyword-search sidecar (JSONL)."""
        return self._warm_store._warm_sidecar_path()

    def warm_add(
        self,
        entry_id: str,
        entry_data: dict[str, object],
        embedding: list[float] | None,
        *,
        provenance: VectorProvenance | None = None,
    ) -> None:
        """Insert or replace an entry in the warm store."""
        self._warm_store.warm_add(entry_id, entry_data, embedding, provenance=provenance)

    def warm_add_many(self, items: list[tuple[str, dict[str, object], list[float] | None]]) -> None:
        """Insert or replace several entries with one sidecar read-modify-write."""
        self._warm_store.warm_add_many(items)

    def warm_remove(self, entry_id: str) -> bool:
        """Delete an entry from the warm store and sidecar."""
        return self._warm_store.warm_remove(entry_id)

    def warm_search(
        self,
        query_tokens: list[str],
        query_embedding: list[float] | None,
        top_k: int = 25,
        *,
        query_space: EmbeddingSpace | None = None,
    ) -> list[dict[str, object]]:
        """Search the warm tier; only vectors in *query_space* are dense-scored."""
        raw_hits = self._warm_store.warm_search(
            query_tokens, query_embedding, max(top_k * 2, top_k), query_space=query_space
        )
        ranked = rank_search_hits(
            raw_hits,
            query_tokens=query_tokens,
            query_embedding=query_embedding,
            config=self._config,
            relevance_hint_keys=("_tier_relevance", "score"),
        )
        return ranked[:top_k]

    def _cold_dir(self) -> Path:
        """Base cold archive directory."""
        return self._cold_store._cold_dir()

    def cold_remove(self, entry_id: str) -> int:
        """Permanently delete an entry from the cold YAML archive."""
        return self._cold_store.cold_remove(entry_id)

    def cold_archive(self, entry_id: str, entry_path: Path) -> None:
        """Move a warm-tier YAML entry to the cold archive partition."""
        self._cold_store.cold_archive(entry_id, entry_path)

    def cold_archive_entry(self, entry_id: str, entry_data: dict[str, object]) -> None:
        """Archive an active canonical entry into the cold tier."""
        namespace = self._namespace
        with self._open_canonical_backend(self._config) as backend:
            self._cold_store.cold_archive_entry(
                entry_id,
                entry_data,
                delete_source_entry_fn=lambda archived_entry_id: backend.delete(archived_entry_id, namespace=namespace),
                verify_source_entry_removed_fn=lambda archived_entry_id: (
                    backend.get(archived_entry_id, namespace=namespace) is None
                ),
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
        query_space: EmbeddingSpace | None = None,
        tags: list[str] | None = None,
        top_k: int = 25,
        restore_entry_fn: Callable[[dict[str, object]], None] | None = None,
        delete_restored_entry_fn: Callable[[str], bool | None] | None = None,
        force_delete_restored_entry_fn: Callable[[str], bool | None] | None = None,
        verify_restored_entry_removed_fn: Callable[[str], bool] | None = None,
        invocation: RecallInvocation | None = None,
        resolve_entry: Callable[[str], MemoryEntry | None] | None = None,
        covered_ids: frozenset[str] = frozenset(),
    ) -> list[dict[str, object]] | list[LocalCandidate]:
        """Search hot, warm, and cold tiers as one merged runtime surface.

        With an *invocation* (discovery mode), *covered_ids* names primary rows
        the caller already ranked for this query: they are neither resolved nor
        vector-scored, so a caller whose hybrid pool held the whole namespace
        pays only for the rows the pool could not see (cold archives and warm
        rows with no primary copy).
        """
        if invocation is not None:
            if resolve_entry is None:
                raise ValueError("tier discovery requires canonical entry resolution")
            with self._hot_lock:
                hot = [entry.model_dump(mode="json") for entry in self._hot.values()]
            warm = self._warm_store.discovery_entries(
                query_embedding, covered_ids=covered_ids, namespace=invocation.namespace, query_space=query_space
            )
            with closing(self._cold_store.iter_search(query_tokens, promote=False)) as cold:
                return discover_candidates(
                    ((row, is_cold) for group, is_cold in ((hot, False), (warm, False), (cold, True)) for row in group),
                    invocation=invocation,
                    resolve_entry=resolve_entry,
                    query_tokens=query_tokens,
                    query_embedding=query_embedding,
                    config=self._config,
                    top_k=top_k,
                    covered_ids=covered_ids,
                )
        if covered_ids:
            raise ValueError("covered_ids applies only to invocation-based tier discovery")
        warm_hits = self.warm_search(
            query_tokens, query_embedding, top_k=max(top_k * 2, top_k), query_space=query_space
        )
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
        ranked = merge_search_results(
            hot_hits,
            warm_hits,
            cold_hits,
            query_tokens=query_tokens,
            query_embedding=query_embedding,
            tags=tags,
            config=self._config,
        )
        return ranked[:top_k]

    def sweep(self, config: MemoryConfig | None = None) -> TierSweepResult:
        """Execute lifecycle sweep across all tiers."""
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
            hot_lock=self._hot_lock,
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
        return open_canonical_backend(self._base_dir, self._entries_dir, self._namespace, config)

    def _load_warm_entries(self, config: MemoryConfig) -> tuple[list[dict[str, object]], int]:
        return load_warm_entries(self._base_dir, self._entries_dir, self._namespace, config)
