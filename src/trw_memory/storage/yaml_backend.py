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
from trw_memory.storage._shared import (
    ENTRY_COLUMNS,
    IMMUTABLE_FIELDS,
    serialize_update_value,
    validate_update_fields,
)
from trw_memory.storage._yaml_row_mapper import (
    ParsedRow,
)
from trw_memory.storage._yaml_row_mapper import (
    dict_to_entry as _dict_to_entry,
)
from trw_memory.storage._yaml_row_mapper import (
    entry_to_dict as _entry_to_dict,
)
from trw_memory.storage.interface import EntryCursor, StorageBackend
from trw_memory.storage.persistence import lock_for_rmw, read_yaml, write_yaml
from trw_memory.sync.delta import DeltaTracker

logger = structlog.get_logger(__name__)

# Allowlist for UPDATE: all columns except immutable ones (derived from shared constants).
_VALID_UPDATE_FIELDS: frozenset[str] = (frozenset(ENTRY_COLUMNS) - IMMUTABLE_FIELDS) | frozenset({"expires"})


def _read_row(path: Path, data: dict[str, object]) -> ParsedRow:
    """Deserialise *data* and log any evidence the mapper had to drop.

    A row whose assertions or anchors did not parse still reads back as a usable
    entry -- availability beats a hard failure on a corrupt file -- but it is
    logged at WARNING rather than left indistinguishable from a row that simply
    carries no verification evidence.
    """
    row = _dict_to_entry(data)
    if row.partial:
        logger.warning(
            "yaml_row_partial_parse",
            path=str(path),
            entry_id=row.entry.id,
            dropped_assertions=row.dropped_assertions,
            dropped_anchors=row.dropped_anchors,
        )
    return row


# ---------------------------------------------------------------------------
# Serialisation helpers
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

    def __repr__(self) -> str:
        return f"YAMLBackend(entries_dir={self._dir!r})"

    def _path(self, entry_id: str) -> Path:
        if not entry_id or "\x00" in entry_id or "/" in entry_id or "\\" in entry_id:
            raise StorageError(
                f"Invalid entry_id: path traversal or nested component in {entry_id!r}",
                path=str(self._dir),
            )
        filename = f"{entry_id}.yaml"
        if Path(filename).name != filename:
            raise StorageError(f"Invalid entry_id: path traversal detected in {entry_id!r}", path=str(self._dir))
        candidate = (self._dir / filename).resolve()
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
                entries.append(_read_row(yaml_file, data).entry)
            except (
                OSError,
                StorageError,
                ValueError,
                KeyError,
            ):  # per-item error handling: skip corrupt YAML files, load the rest
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
        # Auto dirty-mark for sync pipeline (PRD-INFRA-051)
        entry.sync_seq = (entry.sync_seq or 0) + 1
        entry.sync_hash = DeltaTracker.compute_sync_hash(entry)
        entry.last_synced_at = None

        path = self._path(entry.id)
        write_yaml(path, _entry_to_dict(entry))
        logger.debug("yaml_entry_stored", entry_id=entry.id)

    def get(self, entry_id: str, *, namespace: str) -> MemoryEntry | None:
        """Read ``{id}.yaml`` and deserialise, enforcing namespace containment.

        PRD-CORE-245 FR03: a row is identified by ``(namespace, id)``. This
        backend stores one file per id, so containment is enforced on the
        deserialised value — a row belonging to another namespace reads back as
        ``None``, exactly as it would from the composite-keyed SQLite backend.

        Args:
            entry_id: Target entry id.
            namespace: The namespace that must own the entry.

        Returns:
            :class:`MemoryEntry` or ``None`` if absent or foreign.

        Raises:
            StorageError: If the file exists but cannot be parsed.
        """
        entry = self._locate_unscoped(entry_id)
        return entry if entry is not None and entry.namespace == namespace else None

    def _locate_unscoped(self, entry_id: str) -> MemoryEntry | None:
        """Read ``{id}.yaml`` without applying the namespace predicate."""
        path = self._path(entry_id)
        if not path.exists():
            return None
        try:
            data = read_yaml(path)
            return _read_row(path, data).entry
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
            StorageError: If the update fails, or if the stored row carries
                verification evidence the mapper could not parse -- rewriting it
                would drop that evidence from disk.
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
            row = _read_row(path, data)
            if row.partial:
                # Read-modify-write: everything below serialises the PARSED
                # entry back over the file. Rewriting a row whose assertions or
                # anchors did not parse would delete that evidence permanently,
                # so the update is refused while the file is still intact.
                raise StorageError(
                    f"Refusing to rewrite entry {entry_id}: "
                    f"{row.dropped_assertions} assertion(s) and {row.dropped_anchors} anchor(s) "
                    "did not parse, and this update would drop them from the stored row",
                    path=str(path),
                )
            entry = row.entry
            for key, val in field_dict.items():
                setattr(entry, key, val)

            if not {"sync_seq", "sync_hash", "last_synced_at"} & field_dict.keys():
                entry.sync_seq = (entry.sync_seq or 0) + 1
                entry.sync_hash = DeltaTracker.compute_sync_hash(entry)
                entry.last_synced_at = None

            for key, val in _entry_to_dict(entry).items():
                data[key] = serialize_update_value(key, val)

            write_yaml(path, data)

        try:
            return _read_row(path, data).entry
        except (ValueError, KeyError, TypeError) as exc:
            raise StorageError(
                f"Failed to deserialise updated entry {entry_id}: {exc}",
                path=str(path),
            ) from exc

    def delete(self, entry_id: str, *, namespace: str) -> bool:
        """Remove the ``{id}.yaml`` file owned by *namespace*.

        PRD-CORE-245 FR03: a delete addressed at another namespace's row is a
        miss, not a deletion.

        Args:
            entry_id: Target entry id.
            namespace: The namespace that must own the entry.

        Returns:
            ``True`` if deleted, ``False`` if not found or foreign.

        Raises:
            StorageError: If deletion fails for a reason other than not-found.
        """
        if self.get(entry_id, namespace=namespace) is None:
            return False
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
        if top_k <= 0:
            return []
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
        min_importance: float = 0.0,
        limit: int = 100,
        exclude_superseded: bool = False,
        tags: list[str] | None = None,
        after: EntryCursor | None = None,
    ) -> list[MemoryEntry]:
        """Return entries with optional filters.

        Args:
            status: If provided, filter by this status.
            namespace: If provided, filter by this namespace.
            min_importance: If > 0.0, only return entries whose importance is
                >= this value (parity with the SQLite backend's storage-layer
                pre-filter). Default 0.0 disables the importance filter.
            limit: Maximum entries to return.
            exclude_superseded: When True, exclude entries with a non-null
                ``invalid_from`` (bi-temporal superseded entries).
            tags: If provided, only return entries containing ALL of these tags.
                Applied BEFORE the limit so tagged entries past the row limit
                are not truncated away (parity with the SQLite backend).
            after: Keyset position from a previous page; only entries ranked
                strictly below it are returned (parity with the SQLite
                backend, which a namespace merge across two YAML stores
                depends on).

        Returns:
            Up to *limit* matching entries ordered by updated_at descending,
            id descending. The id tiebreak makes the order total, which is what
            makes consecutive ``after=`` pages disjoint and complete.
        """
        if limit <= 0:
            return []
        status_val: str | None = None
        if status is not None:
            status_val = status.value
        required_tags = set(tags) if tags else None

        results: list[MemoryEntry] = []
        for entry in self._load_all():
            entry_status = str(entry.status)
            if status_val is not None and entry_status != status_val:
                continue
            if namespace is not None and entry.namespace != namespace:
                continue
            if min_importance > 0.0 and entry.importance < min_importance:
                continue
            if exclude_superseded and entry.invalid_from is not None:
                continue
            if required_tags is not None and not required_tags.issubset(set(entry.tags)):
                continue
            if after is not None and (entry.updated_at.isoformat(), entry.id) >= (after.updated_at, after.entry_id):
                continue
            results.append(entry)

        results.sort(key=lambda e: (e.updated_at.isoformat(), e.id), reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Namespace operations (override ABC defaults)
    # ------------------------------------------------------------------

    def list_namespaces(self, required_namespaces: list[str] | None = None) -> list[str]:
        """Return distinct namespaces across stored YAML entries.

        Performs an O(N) scan of all YAML files, extracting the ``namespace``
        field from each.  Silently skips corrupt files.

        Args:
            required_namespaces: When provided, scope the result to this
                authorized set so enumeration never leaks other tenants'
                namespaces (trw-memory-11). ``None`` returns every namespace.

        Returns:
            Sorted list of unique namespace strings.
        """
        allowed = set(required_namespaces) if required_namespaces is not None else None
        namespaces: set[str] = set()
        for entry in self._load_all():
            if allowed is not None and entry.namespace not in allowed:
                continue
            namespaces.add(entry.namespace)
        return sorted(namespaces)

    def delete_by_namespace(self, namespace: str) -> int:
        """Delete all YAML entry files belonging to *namespace*.

        Performs an O(N) scan to identify matching entries, then removes
        their files from disk.

        Args:
            namespace: The namespace whose entries should be removed.

        Returns:
            Number of entry files deleted.

        Raises:
            StorageError: If a file deletion fails for a reason other than
                the file already being absent.
        """
        deleted = 0
        for entry in self._load_all():
            if entry.namespace == namespace:
                path = self._path(entry.id)
                try:
                    path.unlink()
                    deleted += 1
                except FileNotFoundError:
                    pass  # already gone — race-safe
                except OSError as exc:
                    raise StorageError(
                        f"Failed to delete entry {entry.id!r} during namespace cleanup: {exc}",
                        path=str(path),
                    ) from exc
        if deleted > 0:
            logger.debug(
                "namespace_deleted",
                namespace=namespace,
                entries_deleted=deleted,
            )
        return deleted

    def close(self) -> None:
        """No-op for the YAML backend — no resources to release."""
