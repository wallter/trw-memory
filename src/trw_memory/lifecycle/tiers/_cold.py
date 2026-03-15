"""Cold tier operations for tiered memory lifecycle.

Manages the YAML archive partitioned by {YYYY}/{MM}/ for long-term storage
of infrequently accessed memory entries.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.storage.persistence import read_yaml, write_yaml

logger = structlog.get_logger()


class ColdTierStore:
    """Cold tier: YAML archive partitioned by {YYYY}/{MM}/.

    Args:
        base_dir: Base directory for memory storage.
        warm_store: WarmTierStore instance for cross-tier operations.
    """

    def __init__(self, base_dir: Path, warm_store: WarmTierStore) -> None:
        self._base_dir = base_dir
        self._warm_store = warm_store

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

        partition = self._cold_partition()
        partition.mkdir(parents=True, exist_ok=True)
        dest = partition / entry_path.name

        try:
            data = read_yaml(entry_path)
            write_yaml(dest, data)
            # Remove from warm sidecar (best-effort)
            with contextlib.suppress(OSError, ValueError):
                self._warm_store.warm_remove(entry_id)
            # Delete original
            entry_path.unlink(missing_ok=True)
            logger.debug("cold_archive", entry_id=entry_id, dest=str(dest))
        except (OSError, StorageError):
            logger.warning(
                "cold_archive_failed",
                entry_id=entry_id,
                src=str(entry_path),
                dest=str(dest),
                exc_info=True,
            )
            raise

    def cold_promote(self, entry_id: str) -> dict[str, object] | None:
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

            # Found -- update last_accessed_at and move to warm
            data["last_accessed_at"] = datetime.now(timezone.utc).isoformat()
            try:
                write_yaml(yaml_file, data)
                self._warm_store.warm_add(entry_id, data, None)
                yaml_file.unlink(missing_ok=True)
                logger.debug("cold_promote", entry_id=entry_id, src=str(yaml_file))
                return data
            except (OSError, StorageError):
                logger.warning(
                    "cold_promote_failed",
                    entry_id=entry_id,
                    path=str(yaml_file),
                    exc_info=True,
                )
                return None

        return None

    def cold_search(self, query_tokens: list[str]) -> list[dict[str, object]]:
        """Linear scan of the cold archive for keyword matches.

        Args:
            query_tokens: Tokens to match (case-insensitive).

        Returns:
            List of matching entry dicts (includes all YAML fields).
        """
        cold_base = self._cold_dir()
        if not cold_base.exists() or not query_tokens:
            return []

        lower_tokens = {t.lower() for t in query_tokens}
        results: list[dict[str, object]] = []

        for yaml_file in sorted(cold_base.rglob("*.yaml")):
            try:
                data = read_yaml(yaml_file)
            except (OSError, StorageError):
                continue

            text = str(data.get("content", data.get("summary", ""))).lower()
            tags = [str(t).lower() for t in cast("list[object]", data.get("tags") or [])]
            text += " " + " ".join(tags)

            if any(tok in text for tok in lower_tokens):
                results.append(data)

        return results
