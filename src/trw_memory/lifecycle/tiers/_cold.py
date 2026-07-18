"""Cold tier operations for tiered memory lifecycle.

Manages the YAML archive partitioned by {YYYY}/{MM}/ for long-term storage
of infrequently accessed memory entries.
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.lifecycle.tiers._cold_partition import entry_partition_timestamp as _entry_partition_timestamp
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.storage.persistence import read_yaml, write_yaml

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger(__name__)
_WARM_EMBEDDING_KEY = "_warm_embedding"


@dataclass(frozen=True)
class _ColdSearchCacheEntry:
    mtime_ns: int
    data: dict[str, object]
    search_text: str


class ColdTierStore:
    """Cold tier: YAML archive partitioned by {YYYY}/{MM}/.

    Args:
        base_dir: Base directory for memory storage.
        warm_store: WarmTierStore instance for cross-tier operations.
        search_cache_max: Maximum number of cold YAML files retained in the
            in-memory search cache before the least-recently-used entry is
            evicted (trw-memory-15). Bounds RAM on long-lived processes with
            large cold archives; the cache previously grew without limit.
    """

    def __init__(self, base_dir: Path, warm_store: WarmTierStore, search_cache_max: int = 1000) -> None:
        self._base_dir = base_dir
        self._warm_store = warm_store
        self._search_cache_max = max(1, search_cache_max)
        self._search_cache: OrderedDict[str, _ColdSearchCacheEntry] = OrderedDict()

    def _cold_dir(self) -> Path:
        """Base cold archive directory (base_dir/memory/cold/)."""
        return self._base_dir / "memory" / "cold"

    def _cold_partition(self, ts: datetime | None = None) -> Path:
        """Return cold partition directory for a given datetime.

        Args:
            ts: Datetime to use for partitioning. Defaults to now (UTC).

        Returns:
            Path like base_dir/memory/cold/2026/02/
        """
        if ts is None:
            ts = datetime.now(timezone.utc)
        return self._cold_dir() / str(ts.year) / f"{ts.month:02d}"

    _entry_partition_timestamp = staticmethod(_entry_partition_timestamp)

    def _assert_within_cold_dir(self, path: Path) -> None:
        """Guard against path traversal attacks on cold archive operations.

        Args:
            path: Path to validate.

        Raises:
            ValueError: If path is outside the cold archive directory.
        """
        cold_base = self._cold_dir().resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(cold_base):
            raise ValueError(f"Path traversal guard: {path} is not under cold dir {cold_base}")

    def _assert_within_base_dir(self, path: Path) -> None:
        """Guard against path traversal attacks on any base dir operations.

        Source entry_path for cold_archive must be inside base_dir.

        Args:
            path: Path to validate.

        Raises:
            ValueError: If path is outside the base directory.
        """
        base = self._base_dir.resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(base):
            raise ValueError(f"Path traversal guard: {path} is not under base_dir {base}")

    def cold_archive(self, entry_id: str, entry_path: Path) -> None:
        """Move a warm-tier YAML entry to the cold archive partition.

        Writes the entry to base_dir/memory/cold/{YYYY}/{MM}/{filename} atomically,
        then removes the original file.

        Args:
            entry_id: Memory entry identifier.
            entry_path: Absolute path to the source YAML file.

        Raises:
            ValueError: If entry_path is outside base_dir (path traversal guard).
            Exception: Re-raises any read/write failure.
        """
        # Path traversal guard
        self._assert_within_base_dir(entry_path)

        data = read_yaml(entry_path)

        def _delete_source(_entry_id: str) -> bool | None:
            entry_path.unlink(missing_ok=True)
            return None

        def _verify_source_removed(_entry_id: str) -> bool:
            return not entry_path.exists()

        self.cold_archive_entry(
            entry_id,
            data,
            delete_source_entry_fn=_delete_source,
            verify_source_entry_removed_fn=_verify_source_removed,
            source_path=entry_path,
        )

    def cold_archive_entry(
        self,
        entry_id: str,
        entry_data: dict[str, object],
        *,
        delete_source_entry_fn: Callable[[str], bool | None] | None = None,
        verify_source_entry_removed_fn: Callable[[str], bool] | None = None,
        source_path: Path | None = None,
    ) -> None:
        """Archive an active entry payload into the cold tier."""
        dest: Path | None = None
        try:
            archive_data, embedding = self._archive_payload(entry_id, entry_data)
            partition = self._cold_partition(self._entry_partition_timestamp(archive_data))
            partition.mkdir(parents=True, exist_ok=True)
            dest_name = source_path.name if source_path is not None else f"{entry_id}.yaml"
            dest = partition / dest_name
            write_yaml(dest, archive_data)
            warm_cleanup_confirmed = False
            try:
                warm_cleanup_confirmed = self._warm_store.warm_remove(entry_id)
            except (OSError, ValueError):
                warm_cleanup_confirmed = self._warm_store.purge_sidecar_entry(entry_id)
            if not warm_cleanup_confirmed:
                dest.unlink(missing_ok=True)
                raise StorageError(f"warm cleanup failed for {entry_id}", path=str(dest))
            if delete_source_entry_fn is not None:
                source_cleanup_confirmed = False
                try:
                    delete_result = delete_source_entry_fn(entry_id)
                    source_cleanup_confirmed = delete_result is not False
                except (OSError, StorageError, ValueError):
                    source_cleanup_confirmed = False
                if verify_source_entry_removed_fn is not None:
                    with contextlib.suppress(OSError, StorageError, ValueError):
                        source_cleanup_confirmed = verify_source_entry_removed_fn(entry_id)
                if not source_cleanup_confirmed:
                    self._rollback_cold_archive_failure(entry_id, dest, entry_data, embedding)
                    raise StorageError(f"source cleanup failed for {entry_id}", path=str(dest))
            logger.debug("cold_archive", entry_id=entry_id, dest=str(dest))
            self._search_cache.pop(str(dest), None)
        except (OSError, StorageError):
            logger.warning(
                "cold_archive_failed",
                entry_id=entry_id,
                src=str(source_path) if source_path is not None else "",
                dest=str(dest) if dest is not None else "",
                exc_info=True,
            )
            raise

    def cold_promote(
        self,
        entry_id: str,
        *,
        restore_entry_fn: Callable[[dict[str, object]], None] | None = None,
        delete_restored_entry_fn: Callable[[str], bool | None] | None = None,
        force_delete_restored_entry_fn: Callable[[str], bool | None] | None = None,
        verify_restored_entry_removed_fn: Callable[[str], bool] | None = None,
    ) -> dict[str, object] | None:
        """Move a cold-tier entry back to warm tier on access.

        Locates the YAML in the cold archive by scanning for a file containing
        the entry_id, updates last_accessed_at, adds to warm tier, and removes
        the cold file.

        Args:
            entry_id: Memory entry identifier to promote.

        Returns:
            Entry data dict if found and promoted, None otherwise.
        """
        cold_base = self._cold_dir()
        if not cold_base.exists():
            return None

        for yaml_file in cold_base.rglob("*.yaml"):
            try:
                data = read_yaml(yaml_file)
            except (OSError, StorageError):
                continue
            if str(data.get("id", "")) != entry_id:
                continue

            # Stage the promoted payload in memory so a failed promotion does not
            # mutate the cold archive's retention metadata.
            promoted_data = dict(data)
            embedding = self._extract_archived_embedding(promoted_data)
            promoted_data["last_accessed_at"] = datetime.now(timezone.utc).isoformat()
            canonical_restore = restore_entry_fn or self._restore_to_entries_dir
            canonical_delete = delete_restored_entry_fn or self._delete_from_entries_dir
            canonical_force_delete = force_delete_restored_entry_fn or self._delete_from_entries_dir
            restored = False
            warm_added = False
            try:
                canonical_restore(promoted_data)
                restored = True
                self._warm_store.warm_add(entry_id, promoted_data, embedding)
                warm_added = True
                yaml_file.unlink(missing_ok=True)
                self._search_cache.pop(str(yaml_file), None)
                logger.debug("cold_promote", entry_id=entry_id, src=str(yaml_file))
                return promoted_data
            except (OSError, StorageError):
                if warm_added:
                    warm_cleanup_confirmed = False
                    try:
                        warm_cleanup_confirmed = self._warm_store.warm_remove(entry_id)
                    except (OSError, ValueError):
                        warm_cleanup_confirmed = self._warm_store.purge_sidecar_entry(entry_id)
                    if not warm_cleanup_confirmed:
                        logger.warning("cold_promote_warm_rollback_incomplete", entry_id=entry_id, path=str(yaml_file))
                if restored:
                    rollback_confirmed = self._rollback_restored_entry(
                        entry_id,
                        canonical_delete=canonical_delete,
                        canonical_force_delete=canonical_force_delete,
                        verify_restored_entry_removed_fn=verify_restored_entry_removed_fn,
                    )
                    if not rollback_confirmed:
                        logger.warning(
                            "cold_promote_rollback_incomplete",
                            entry_id=entry_id,
                            path=str(yaml_file),
                        )
                logger.warning("cold_promote_failed", entry_id=entry_id, path=str(yaml_file), exc_info=True)
                return None

        return None

    def cold_remove(self, entry_id: str) -> int:
        """Permanently delete an entry from the cold YAML archive.

        Scans the cold partition tree for any archived YAML whose ``id`` field
        matches ``entry_id`` and unlinks it. Used by erasure / GDPR
        ``forget`` flows so a deleted entry cannot survive in the cold tier.

        Failures to stat/read a file are skipped (the file may be mid-write or
        belong to another entry); unlink failures are logged WARN and the file
        is counted as *not* removed so the caller can detect incomplete erasure.

        Args:
            entry_id: Memory entry identifier to erase from cold storage.

        Returns:
            Count of cold YAML files removed (0 if the entry was not archived).
        """
        cold_base = self._cold_dir()
        if not cold_base.exists():
            return 0

        removed = 0
        for yaml_file in sorted(cold_base.rglob("*.yaml")):
            try:
                data = read_yaml(yaml_file)
            except (OSError, StorageError):
                continue
            if str(data.get("id", "")) != entry_id:
                continue
            try:
                yaml_file.unlink(missing_ok=True)
            except OSError:
                logger.warning("cold_remove_unlink_failed", entry_id=entry_id, path=str(yaml_file), exc_info=True)
                continue
            self._search_cache.pop(str(yaml_file), None)
            removed += 1
            logger.debug("cold_remove", entry_id=entry_id, path=str(yaml_file))
        return removed

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
        """Linear scan of the cold archive for keyword matches.

        Args:
            query_tokens: Tokens to match (case-insensitive).
            promote: If True, promote matched entries back to warm storage before
                returning them so recall-time cold hits repair the active tier set.
            top_k: Optional maximum number of matches to return.

        Returns:
            List of matching entry dicts (includes all YAML fields).
        """
        cold_base = self._cold_dir()
        if not cold_base.exists() or not query_tokens:
            return []

        lower_tokens = {t.lower() for t in query_tokens}
        results: list[dict[str, object]] = []
        live_paths: set[str] = set()

        for yaml_file in sorted(cold_base.rglob("*.yaml")):
            live_paths.add(str(yaml_file))
            cached = self._cached_search_entry(yaml_file)
            if cached is None:
                continue
            data = cached.data
            text = cached.search_text

            if any(tok in text for tok in lower_tokens):
                if promote:
                    entry_id = str(data.get("id", ""))
                    promoted_entry = (
                        self.cold_promote(
                            entry_id,
                            restore_entry_fn=restore_entry_fn,
                            delete_restored_entry_fn=delete_restored_entry_fn,
                            force_delete_restored_entry_fn=force_delete_restored_entry_fn,
                            verify_restored_entry_removed_fn=verify_restored_entry_removed_fn,
                        )
                        if entry_id
                        else None
                    )
                    if promoted_entry is None:
                        continue
                    results.append(promoted_entry)
                else:
                    results.append(self._sanitize_archived_entry(data))
                if top_k is not None and len(results) >= top_k:
                    break

        stale_paths = set(self._search_cache) - live_paths
        for stale_path in stale_paths:
            self._search_cache.pop(stale_path, None)

        return results

    def _cached_search_entry(self, yaml_file: Path) -> _ColdSearchCacheEntry | None:
        cache_key = str(yaml_file)
        try:
            mtime_ns = yaml_file.stat().st_mtime_ns
        except OSError:
            self._search_cache.pop(cache_key, None)
            return None

        cached = self._search_cache.get(cache_key)
        if cached is not None and cached.mtime_ns == mtime_ns:
            self._search_cache.move_to_end(cache_key)  # LRU touch on hit
            return cached

        try:
            data = read_yaml(yaml_file)
        except (OSError, StorageError):
            self._search_cache.pop(cache_key, None)
            return None

        text = str(data.get("content", data.get("summary", ""))).lower()
        text += " " + str(data.get("detail", "")).lower()
        tags = [str(t).lower() for t in cast("list[object]", data.get("tags") or [])]
        text += " " + " ".join(tags)
        entry = _ColdSearchCacheEntry(mtime_ns=mtime_ns, data=data, search_text=text)
        self._search_cache[cache_key] = entry
        self._search_cache.move_to_end(cache_key)  # mark most-recently-used
        self._evict_search_cache_overflow()
        return entry

    def _evict_search_cache_overflow(self) -> None:
        """Bound the cold-tier search cache to ``search_cache_max`` (LRU).

        ``_cached_search_entry`` inserts one entry per distinct cold YAML file
        searched. Without this bound the cache grew without limit, leaking RAM
        on long-lived processes that search across a large cold archive
        (trw-memory-15). Eviction drops the least-recently-used key first.
        """
        while len(self._search_cache) > self._search_cache_max:
            self._search_cache.popitem(last=False)

    def _archive_payload(
        self,
        entry_id: str,
        entry_data: dict[str, object],
    ) -> tuple[dict[str, object], list[float] | None]:
        archive_data = dict(entry_data)
        embedding = self._warm_store.get_embedding(entry_id)
        if embedding is not None:
            archive_data[_WARM_EMBEDDING_KEY] = embedding
        return archive_data, embedding

    def _extract_archived_embedding(self, entry_data: dict[str, object]) -> list[float] | None:
        raw_embedding = entry_data.pop(_WARM_EMBEDDING_KEY, None)
        if not isinstance(raw_embedding, list):
            return None
        values: list[float] = []
        for value in raw_embedding:
            if not isinstance(value, (int, float)):
                return None
            values.append(float(value))
        return values

    def _sanitize_archived_entry(self, entry_data: dict[str, object]) -> dict[str, object]:
        sanitized = dict(entry_data)
        sanitized.pop(_WARM_EMBEDDING_KEY, None)
        return sanitized

    def _rollback_cold_archive_failure(
        self,
        entry_id: str,
        dest: Path,
        entry_data: dict[str, object],
        embedding: list[float] | None,
    ) -> None:
        cold_removed = False
        with contextlib.suppress(OSError):
            dest.unlink(missing_ok=True)
            cold_removed = True
        if not cold_removed:
            logger.warning("cold_archive_rollback_incomplete", entry_id=entry_id, path=str(dest))
        try:
            self._warm_store.warm_add(entry_id, entry_data, embedding)
        except (OSError, ValueError):
            logger.warning("cold_archive_warm_restore_failed", entry_id=entry_id, exc_info=True)

    def _rollback_restored_entry(
        self,
        entry_id: str,
        *,
        canonical_delete: Callable[[str], bool | None],
        canonical_force_delete: Callable[[str], bool | None],
        verify_restored_entry_removed_fn: Callable[[str], bool] | None,
    ) -> bool:
        with contextlib.suppress(OSError, StorageError, ValueError, RuntimeError):
            canonical_delete(entry_id)

        if self._restored_entry_removed(entry_id, verify_restored_entry_removed_fn):
            return True

        with contextlib.suppress(OSError, StorageError, ValueError, RuntimeError):
            canonical_force_delete(entry_id)

        return self._restored_entry_removed(entry_id, verify_restored_entry_removed_fn)

    def _restored_entry_removed(
        self,
        entry_id: str,
        verify_restored_entry_removed_fn: Callable[[str], bool] | None,
    ) -> bool:
        if verify_restored_entry_removed_fn is not None:
            with contextlib.suppress(OSError, StorageError, ValueError):
                return verify_restored_entry_removed_fn(entry_id)
            return False
        return not (self._base_dir / "entries" / f"{entry_id}.yaml").exists()

    def _restore_to_entries_dir(self, entry_data: dict[str, object]) -> None:
        """Default canonical restore for direct TierManager usage."""
        entry_id = str(entry_data.get("id", ""))
        if not entry_id:
            raise ValueError("entry id required for canonical restore")
        entries_dir = self._base_dir / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(entries_dir / f"{entry_id}.yaml", entry_data)

    def _delete_from_entries_dir(self, entry_id: str) -> bool | None:
        """Rollback helper for the default canonical restore path."""
        (self._base_dir / "entries" / f"{entry_id}.yaml").unlink(missing_ok=True)
        return None
