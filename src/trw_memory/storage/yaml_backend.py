"""YAML file-based storage backend.

Portable fallback implementation of :class:`StorageBackend`.  Each
:class:`MemoryEntry` is stored as a separate YAML file named ``{id}.yaml``
inside a configurable directory.

Trade-offs vs :class:`~trw_memory.storage.sqlite_backend.SQLiteBackend`:

- Pro: human-readable, version-control friendly, zero binary dependencies
- Con: O(N) for all queries — use for small stores (<10k entries) only
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage._parsing import parse_dt, parse_json_dict_str, parse_json_list
from trw_memory.storage._shared import (
    serialize_update_value,
    validate_update_fields,
)
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.persistence import lock_for_rmw, read_yaml, write_yaml

logger = structlog.get_logger(__name__)

_VALID_UPDATE_FIELDS: frozenset[str] = frozenset(
    {
        "content",
        "detail",
        "tags",
        "evidence",
        "importance",
        "status",
        "recurrence",
        "namespace",
        "updated_at",
        "last_accessed_at",
        "access_count",
        "q_value",
        "q_observations",
        "source",
        "source_identity",
        "merged_from",
        "consolidated_from",
        "consolidated_into",
        "metadata",
    }
)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _entry_to_dict(entry: MemoryEntry) -> dict[str, object]:
    """Serialise a :class:`MemoryEntry` to a plain dict suitable for YAML."""
    status_val = str(entry.status)
    return {
        "id": entry.id,
        "content": entry.content,
        "detail": entry.detail,
        "tags": list(entry.tags),
        "evidence": list(entry.evidence),
        "importance": entry.importance,
        "status": status_val,
        "recurrence": entry.recurrence,
        "namespace": entry.namespace,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "last_accessed_at": (entry.last_accessed_at.isoformat() if entry.last_accessed_at else None),
        "access_count": entry.access_count,
        "q_value": entry.q_value,
        "q_observations": entry.q_observations,
        "source": entry.source,
        "source_identity": entry.source_identity,
        "merged_from": list(entry.merged_from),
        "consolidated_from": list(entry.consolidated_from),
        "consolidated_into": entry.consolidated_into,
        "metadata": dict(entry.metadata),
    }


def _dict_to_entry(data: dict[str, object]) -> MemoryEntry:
    """Deserialise a YAML dict back into a :class:`MemoryEntry`.

    All fields are cast explicitly to satisfy Pydantic strict mode.
    """

    def _str(key: str, default: str = "") -> str:
        val = data.get(key, default)
        return str(val) if val is not None else default

    def _int(key: str, default: int = 0) -> int:
        val = data.get(key, default)
        try:
            return int(str(val))
        except (TypeError, ValueError):
            return default

    def _float(key: str, default: float = 0.5) -> float:
        val = data.get(key, default)
        try:
            return float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    def _str_list(key: str) -> list[str]:
        return parse_json_list(data.get(key, []))

    def _str_dict(key: str) -> dict[str, str]:
        return parse_json_dict_str(data.get(key, {}))

    created_at_raw = data.get("created_at")
    updated_at_raw = data.get("updated_at")
    created_at = parse_dt(created_at_raw) if created_at_raw else datetime.now(timezone.utc)
    updated_at = parse_dt(updated_at_raw) if updated_at_raw else datetime.now(timezone.utc)

    last_accessed_raw = data.get("last_accessed_at")
    last_accessed_at: datetime | None = parse_dt(last_accessed_raw) if last_accessed_raw else None

    status_raw = _str("status", "active")
    try:
        status = MemoryStatus(status_raw)
    except ValueError:
        status = MemoryStatus.ACTIVE

    consolidated_into_raw = data.get("consolidated_into")
    consolidated_into: str | None = str(consolidated_into_raw) if consolidated_into_raw else None

    return MemoryEntry(
        id=_str("id"),
        content=_str("content"),
        detail=_str("detail"),
        tags=_str_list("tags"),
        evidence=_str_list("evidence"),
        importance=_float("importance", 0.5),
        status=status,
        recurrence=_int("recurrence", 1),
        namespace=_str("namespace", "default"),
        created_at=created_at,
        updated_at=updated_at,
        last_accessed_at=last_accessed_at,
        access_count=_int("access_count", 0),
        q_value=_float("q_value", 0.5),
        q_observations=_int("q_observations", 0),
        source=_str("source", "agent"),
        source_identity=_str("source_identity"),
        merged_from=_str_list("merged_from"),
        consolidated_from=_str_list("consolidated_from"),
        consolidated_into=consolidated_into,
        metadata=_str_dict("metadata"),
    )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class YAMLBackend(StorageBackend):
    """YAML file-based storage backend.

    Each :class:`MemoryEntry` is stored as ``{entries_dir}/{id}.yaml``.
    All query operations are O(N) scans.

    Args:
        entries_dir: Root directory where entry YAML files are written.
            Created automatically if it does not exist.
    """

    def __init__(self, entries_dir: Path) -> None:
        self._dir = entries_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.debug("yaml_backend_init", entries_dir=str(entries_dir))

    def _path(self, entry_id: str) -> Path:
        candidate = (self._dir / f"{entry_id}.yaml").resolve()
        if not candidate.is_relative_to(self._dir.resolve()):
            raise StorageError(
                f"Invalid entry_id: path traversal detected in {entry_id!r}",
                path=str(candidate),
            )
        return candidate

    def _load_all(self) -> list[MemoryEntry]:
        """Load every YAML file in the directory.  Silently skips corrupt files."""
        entries: list[MemoryEntry] = []
        for yaml_file in self._dir.glob("*.yaml"):
            try:
                data = read_yaml(yaml_file)
                entries.append(_dict_to_entry(data))
            except (OSError, StorageError, ValueError, KeyError):  # per-item error handling: skip corrupt YAML files, load the rest  # noqa: PERF203
                logger.warning("yaml_backend_skip_corrupt", path=str(yaml_file))
        return entries

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> None:
        """Atomically write the entry to ``{id}.yaml``.

        Args:
            entry: Memory entry to persist.

        Raises:
            StorageError: If the write fails.
        """
        path = self._path(entry.id)
        write_yaml(path, _entry_to_dict(entry))
        logger.debug("yaml_entry_stored", entry_id=entry.id)

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Read ``{id}.yaml`` and deserialise.

        Args:
            entry_id: Target entry id.

        Returns:
            :class:`MemoryEntry` or ``None`` if the file does not exist.

        Raises:
            StorageError: If the file exists but cannot be parsed.
        """
        path = self._path(entry_id)
        if not path.exists():
            return None
        try:
            data = read_yaml(path)
            return _dict_to_entry(data)
        except StorageError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            raise StorageError(
                f"Failed to deserialise entry {entry_id}: {exc}",
                path=str(path),
            ) from exc

    def update(self, entry_id: str, **fields: object) -> MemoryEntry | None:
        """Read-modify-write the entry with an exclusive lock.

        Args:
            entry_id: Target entry id.
            **fields: Fields to update.

        Returns:
            Updated :class:`MemoryEntry` or ``None`` if not found.

        Raises:
            StorageError: If the update fails.
        """
        path = self._path(entry_id)
        if not path.exists():
            return None

        with lock_for_rmw(path):
            try:
                data = read_yaml(path)
            except StorageError:
                raise
            except (OSError, ValueError, KeyError) as exc:
                raise StorageError(
                    f"Failed to read entry {entry_id} for update: {exc}",
                    path=str(path),
                ) from exc

            # Apply updates — serialise complex types to YAML-friendly forms
            field_dict: dict[str, object] = dict(fields)
            try:
                validate_update_fields(field_dict, _VALID_UPDATE_FIELDS)
            except ValueError as ve:
                raise StorageError(
                    f"Invalid update field: {ve.args[0]!r}",
                    path=str(path),
                ) from None
            if "updated_at" not in field_dict:
                field_dict["updated_at"] = datetime.now(timezone.utc)
            for key, val in field_dict.items():
                data[key] = serialize_update_value(key, val)

            write_yaml(path, data)

        try:
            return _dict_to_entry(data)
        except (ValueError, KeyError, TypeError) as exc:
            raise StorageError(
                f"Failed to deserialise updated entry {entry_id}: {exc}",
                path=str(path),
            ) from exc

    def delete(self, entry_id: str) -> bool:
        """Remove the ``{id}.yaml`` file.

        Args:
            entry_id: Target entry id.

        Returns:
            ``True`` if deleted, ``False`` if not found.

        Raises:
            StorageError: If deletion fails for a reason other than not-found.
        """
        path = self._path(entry_id)
        if not path.exists():
            return False
        try:
            path.unlink()
            logger.debug("yaml_entry_deleted", entry_id=entry_id)
            return True
        except OSError as exc:
            raise StorageError(
                f"Failed to delete entry {entry_id}: {exc}",
                path=str(path),
            ) from exc

    def search(
        self,
        query: str,
        *,
        top_k: int = 25,
        tags: list[str] | None = None,
        status: MemoryStatus | None = None,
        min_importance: float = 0.0,
        namespace: str | None = None,
    ) -> list[MemoryEntry]:
        """O(N) keyword scan with optional filters.

        Matches entries where *query* appears (case-insensitive) in
        ``content``, ``detail``, or any tag string.

        Args:
            query: Free-text search term.
            top_k: Maximum results.
            tags: If provided, entries must contain ALL of these tags.
            status: If provided, restrict to this status.
            min_importance: Lower bound on importance (inclusive).
            namespace: If provided, restrict to this namespace.

        Returns:
            Up to *top_k* matching entries sorted by importance desc.
        """
        needle = query.lower()
        status_val: str | None = None
        if status is not None:
            status_val = status.value

        results: list[MemoryEntry] = []
        for entry in self._load_all():
            # Keyword match
            entry_status = str(entry.status)
            tag_text = " ".join(entry.tags).lower()
            if needle not in entry.content.lower() and needle not in entry.detail.lower() and needle not in tag_text:
                continue

            # Status filter
            if status_val is not None and entry_status != status_val:
                continue

            # Importance filter
            if entry.importance < min_importance:
                continue

            # Namespace filter
            if namespace is not None and entry.namespace != namespace:
                continue

            # Tag all-of filter
            if tags and not set(tags).issubset(set(entry.tags)):
                continue

            results.append(entry)

        # Sort by importance desc, then by updated_at desc
        results.sort(key=lambda e: (e.importance, e.updated_at.isoformat()), reverse=True)
        return results[:top_k]

    def count(self, namespace: str | None = None) -> int:
        """Count YAML files matching the optional namespace filter.

        Args:
            namespace: If provided, count only entries in this namespace.

        Returns:
            Entry count.
        """
        if namespace is None:
            # Fast path: count files without deserialising
            return sum(1 for _ in self._dir.glob("*.yaml"))
        # Need to deserialise for namespace filtering
        return sum(1 for e in self._load_all() if e.namespace == namespace)

    def list_entries(
        self,
        *,
        status: MemoryStatus | None = None,
        namespace: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Return entries with optional filters.

        Args:
            status: If provided, filter by this status.
            namespace: If provided, filter by this namespace.
            limit: Maximum entries to return.

        Returns:
            Up to *limit* matching entries ordered by updated_at descending.
        """
        status_val: str | None = None
        if status is not None:
            status_val = status.value

        results: list[MemoryEntry] = []
        for entry in self._load_all():
            entry_status = str(entry.status)
            if status_val is not None and entry_status != status_val:
                continue
            if namespace is not None and entry.namespace != namespace:
                continue
            results.append(entry)

        results.sort(key=lambda e: e.updated_at.isoformat(), reverse=True)
        return results[:limit]

    def close(self) -> None:
        """No-op for the YAML backend."""
