"""TierManager: Hot/Warm/Cold tier orchestrator for memory entry lifecycle.

Composes WarmTierStore and ColdTierStore, manages the hot LRU cache directly,
and orchestrates lifecycle sweep transitions via execute_sweep().
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import structlog

from trw_memory.lifecycle.tiers._cold import ColdTierStore
from trw_memory.lifecycle.tiers._scoring import TierSweepResult
from trw_memory.lifecycle.tiers._sweep import execute_sweep
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger(__name__)


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

        # Hot tier: OrderedDict used as LRU cache
        # LRU invariant: MRU at the end (rightmost), LRU at the front (leftmost)
        self._hot: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._hot_lock = threading.Lock()

        # Composed tier stores
        self._warm_store = WarmTierStore(base_dir)
        self._cold_store = ColdTierStore(base_dir, self._warm_store)

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
            return self._hot[entry_id]

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
                evicted_id, _ = self._hot.popitem(last=False)
                logger.debug(
                    "hot_tier_evict",
                    evicted_id=evicted_id,
                    capacity=cfg.hot_max_entries,
                )

    def hot_clear(self) -> None:
        """Evict all entries from the hot cache (for testing / shutdown)."""
        with self._hot_lock:
            self._hot.clear()

    @property
    def hot_size(self) -> int:
        """Number of entries currently in the hot cache."""
        with self._hot_lock:
            return len(self._hot)

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

    def warm_remove(self, entry_id: str) -> None:
        """Delete an entry from the warm store and sidecar."""
        self._warm_store.warm_remove(entry_id)

    def warm_search(
        self,
        query_tokens: list[str],
        query_embedding: list[float] | None,
        top_k: int = 25,
    ) -> list[dict[str, object]]:
        """Search the warm tier for relevant entries."""
        return self._warm_store.warm_search(query_tokens, query_embedding, top_k)

    # -----------------------------------------------------------------------
    # Cold Tier (delegated)
    # -----------------------------------------------------------------------

    def _cold_dir(self) -> Path:
        """Base cold archive directory."""
        return self._cold_store._cold_dir()

    def cold_archive(self, entry_id: str, entry_path: Path) -> None:
        """Move a warm-tier YAML entry to the cold archive partition."""
        self._cold_store.cold_archive(entry_id, entry_path)

    def cold_promote(self, entry_id: str) -> dict[str, object] | None:
        """Move a cold-tier entry back to warm tier on access."""
        return self._cold_store.cold_promote(entry_id)

    def cold_search(self, query_tokens: list[str]) -> list[dict[str, object]]:
        """Linear scan of the cold archive for keyword matches."""
        return self._cold_store.cold_search(query_tokens)

    # -----------------------------------------------------------------------
    # Sweep (delegated)
    # -----------------------------------------------------------------------

    def sweep(self) -> TierSweepResult:
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
        return execute_sweep(
            hot=self._hot,
            config=self._config,
            entries_dir=self._entries_dir,
            base_dir=self._base_dir,
            warm_add_fn=self.warm_add,
            cold_archive_fn=self.cold_archive,
            cold_dir=self._cold_dir(),
        )
