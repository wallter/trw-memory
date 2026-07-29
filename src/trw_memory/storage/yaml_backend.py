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
from typing import Literal, cast

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import Anchor, Assertion, MemoryEntry, MemoryStatus
from trw_memory.storage._parsing import (
    parse_dt_safe,
    parse_json_dict_int,
    parse_json_dict_str,
    parse_json_list,
)
from trw_memory.storage._row_mapper import parse_verification_status
from trw_memory.storage._shared import (
    ENTRY_COLUMNS,
    IMMUTABLE_FIELDS,
    serialize_update_value,
    validate_update_fields,
)
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.persistence import lock_for_rmw, read_yaml, write_yaml
from trw_memory.sync.delta import DeltaTracker

logger = structlog.get_logger(__name__)

# Allowlist for UPDATE: all columns except immutable ones (derived from shared constants).
_VALID_UPDATE_FIELDS: frozenset[str] = (frozenset(ENTRY_COLUMNS) - IMMUTABLE_FIELDS) | frozenset({"expires"})


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _parse_assertions(raw: object) -> list[Assertion]:
    """Deserialise assertions from YAML data.

    Skips malformed items with a debug log rather than failing the entire entry.
    """
    if not raw or not isinstance(raw, list):
        return []
    result: list[Assertion] = []
    for item in raw:
        if isinstance(item, dict):
            try:
                result.append(Assertion.model_validate(item, strict=False))
            except (ValueError, KeyError):
                logger.debug("yaml_assertion_parse_skipped", item=item)
                continue
    return result


def _entry_to_dict(entry: MemoryEntry) -> dict[str, object]:
    """Serialise a :class:`MemoryEntry` to a plain dict suitable for YAML."""
    return entry.to_dict()


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
            return float(str(val))
        except (TypeError, ValueError):
            return default

    def _str_list(key: str) -> list[str]:
        return parse_json_list(data.get(key, []))

    def _str_dict(key: str) -> dict[str, str]:
        return parse_json_dict_str(data.get(key, {}))

    # Timestamps are parsed fail-open: a malformed value (e.g. the WAL-reset
    # byte-shift '026-04-13T00:00:00+00:002') degrades the single field to a
    # usable default instead of raising and collapsing the whole listing —
    # matching how this mapper already fail-opens status / anchors / floats.
    now = datetime.now(timezone.utc)
    created_at_raw = data.get("created_at")
    updated_at_raw = data.get("updated_at")
    created_at = parse_dt_safe(created_at_raw, default=now) or now if created_at_raw else now
    updated_at = parse_dt_safe(updated_at_raw, default=now) or now if updated_at_raw else now

    last_accessed_raw = data.get("last_accessed_at")
    last_accessed_at: datetime | None = parse_dt_safe(last_accessed_raw, default=None) if last_accessed_raw else None

    status_raw = _str("status", "active")
    try:
        status = MemoryStatus(status_raw)
    except ValueError:
        status = MemoryStatus.ACTIVE

    consolidated_into_raw = data.get("consolidated_into")
    consolidated_into: str | None = str(consolidated_into_raw) if consolidated_into_raw else None

    anchors_raw = data.get("anchors")
    anchors: list[Anchor] = []
    if anchors_raw:
        try:
            anchors = [Anchor.model_validate(a) for a in anchors_raw] if isinstance(anchors_raw, list) else []
        except (ValueError, KeyError):
            logger.debug("yaml_anchor_parse_skipped", anchors=anchors_raw)
            anchors = []

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
        session_count=_int("session_count", 0),
        q_value=_float("q_value", 0.5),
        q_observations=_int("q_observations", 0),
        source=cast("Literal['human', 'agent', 'tool', 'consolidated']", _str("source", "agent")),
        source_identity=_str("source_identity"),
        client_profile=_str("client_profile"),
        model_id=_str("model_id"),
        merged_from=_str_list("merged_from"),
        consolidated_from=_str_list("consolidated_from"),
        consolidated_into=consolidated_into,
        metadata=_str_dict("metadata"),
        vector_clock=parse_json_dict_int(data.get("vector_clock", {})),
        remote_id=str(remote_id_raw) if (remote_id_raw := data.get("remote_id")) else None,
        published_to_platform=bool(data.get("published_to_platform", False)),
        pending_delete=bool(data.get("pending_delete", False)),
        cross_validated=bool(data.get("cross_validated", False)),
        outcome_history=_str_list("outcome_history"),
        assertions=_parse_assertions(data.get("assertions", [])),
        anchors=anchors,
        valid_from=parse_dt_safe(data.get("valid_from"), default=created_at) or created_at,
        invalid_from=parse_dt_safe(data.get("invalid_from"), default=None) if data.get("invalid_from") else None,
        invalidated_by=str(data["invalidated_by"]) if data.get("invalidated_by") else None,
        anchor_validity=_float("anchor_validity", 1.0),
        verification_status=parse_verification_status(data.get("verification_status")),
        type=_str("type", "pattern"),
        nudge_line=_str("nudge_line", ""),
        expires=_str("expires", ""),
        confidence=_str("confidence", "unverified"),
        task_type=_str("task_type", ""),
        domain=_str_list("domain"),
        phase_origin=_str("phase_origin", ""),
        phase_affinity=_str_list("phase_affinity"),
        team_origin=_str("team_origin", ""),
        protection_tier=_str("protection_tier", "normal"),
        sync_hash=_str("sync_hash", ""),
        sync_seq=_int("sync_seq", 0),
        last_synced_at=parse_dt_safe(data.get("last_synced_at"), default=None) if data.get("last_synced_at") else None,
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
                entries.append(_dict_to_entry(data))
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
            entry = _dict_to_entry(data)
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

        Returns:
            Up to *limit* matching entries ordered by updated_at descending.
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
            results.append(entry)

        results.sort(key=lambda e: e.updated_at.isoformat(), reverse=True)
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
