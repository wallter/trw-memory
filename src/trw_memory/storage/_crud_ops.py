"""SQLite CRUD operations.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — ``SQLiteBackend.store`` etc. become 1-line delegators
that pass the backend handle.

6 helpers covering the persistence operations:

- ``store`` — INSERT OR REPLACE with UTF-8 validation + sync-pipeline
  bookkeeping (sync_seq increment, sync_hash compute, last_synced_at
  reset).
- ``get`` — single-row SELECT by id.
- ``update`` — partial UPDATE with sync-pipeline auto-dirty +
  whitelisted column validation + JSON serialization for list/dict
  fields. Suppresses commit when within a ``transaction()`` block
  via ``backend._skip_commit_depth``.
- ``increment_session_counts`` — bulk session_count++.
- ``increment_access_counts`` — bulk access_count++ + last_accessed_at.
- ``delete`` — DELETE WHERE id, with vector cleanup when available.

Each helper takes a ``backend`` argument exposing the instance state
(_conn, _lock, _db_path, _skip_commit_depth, _vec_available, _delete_vector).

Extracted as PRD-DIST-245 Phase 1 batch 87.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._row_mapper import entry_to_row, row_to_entry
from trw_memory.storage._shared import (
    DICT_FIELDS,
    LIST_FIELDS,
    serialize_update_value,
    validate_update_fields,
)
from trw_memory.storage._utf8_validator import validate_utf8_fields
from trw_memory.sync.delta import DeltaTracker

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)


def store(
    backend: SQLiteBackend,
    insert_columns_sql: str,
    columns: tuple[str, ...],
    entry: MemoryEntry,
) -> None:
    """INSERT OR REPLACE the entry into the memories table."""
    validate_utf8_fields(
        {
            "id": entry.id,
            "content": entry.content,
            "detail": entry.detail,
            "nudge_line": entry.nudge_line,
            "type": entry.type,
            "namespace": entry.namespace,
            "source": entry.source,
            "source_identity": entry.source_identity,
            "client_profile": entry.client_profile,
            "model_id": entry.model_id,
            "consolidated_into": entry.consolidated_into,
            "remote_id": entry.remote_id,
            "expires_at": entry.expires,
            "task_type": entry.task_type,
            "phase_origin": entry.phase_origin,
            "team_origin": entry.team_origin,
            "outcome_correlation": entry.outcome_correlation,
            "sync_hash": entry.sync_hash,
        }
    )

    entry.sync_seq = (entry.sync_seq or 0) + 1
    entry.sync_hash = DeltaTracker.compute_sync_hash(entry)
    entry.last_synced_at = None

    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT OR REPLACE INTO memories ({insert_columns_sql}) VALUES ({placeholders})"  # noqa: S608
    try:
        with backend._lock:
            backend._conn.execute(sql, entry_to_row(entry))
            backend._conn.commit()
        logger.debug("memory_stored", entry_id=entry.id)
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        raise StorageError(
            f"Failed to store entry {entry.id}: {exc}",
            path=str(backend._db_path),
        ) from exc


def get(
    backend: SQLiteBackend,
    select_columns_sql: str,
    entry_id: str,
) -> MemoryEntry | None:
    """Retrieve an entry by id."""
    sql = f"SELECT {select_columns_sql} FROM memories WHERE id = ?"  # noqa: S608
    try:
        with backend._lock:
            row = backend._conn.execute(sql, (entry_id,)).fetchone()
        if row is None:
            return None
        return row_to_entry(tuple(row))
    except (sqlite3.Error, ValueError, KeyError) as exc:
        raise StorageError(
            f"Failed to get entry {entry_id}: {exc}",
            path=str(backend._db_path),
        ) from exc


def update(
    backend: SQLiteBackend,
    select_columns_sql: str,
    valid_update_columns: frozenset[str],
    entry_id: str,
    **fields: object,
) -> MemoryEntry | None:
    """Apply a partial update to an existing entry."""
    if not fields:
        return get(backend, select_columns_sql, entry_id)

    existing = get(backend, select_columns_sql, entry_id)
    if existing is None:
        return None

    try:
        set_parts: list[str] = []
        values: list[object] = []

        field_dict: dict[str, object] = dict(fields)
        if "updated_at" not in field_dict:
            field_dict["updated_at"] = datetime.now(timezone.utc)

        try:
            validate_update_fields(field_dict, valid_update_columns)
        except ValueError as ve:
            raise StorageError(
                f"Invalid update field: {ve.args[0]!r}",
                path=str(backend._db_path),
            ) from None

        # Mark dirty for sync pipeline (PRD-INFRA-051)
        if not {"sync_seq", "sync_hash", "last_synced_at"} & field_dict.keys():
            updated_entry = existing.model_copy(deep=True)
            for key, val in field_dict.items():
                setattr(updated_entry, key, val)
            next_sync_seq = (existing.sync_seq or 0) + 1
            field_dict["sync_seq"] = next_sync_seq
            updated_entry.sync_seq = next_sync_seq
            field_dict["sync_hash"] = DeltaTracker.compute_sync_hash(updated_entry)
            field_dict["last_synced_at"] = None

        for key, val in field_dict.items():
            sql_key = "expires_at" if key == "expires" else key
            set_parts.append(f"{sql_key} = ?")
            normalised = serialize_update_value(key, val)
            if (key in LIST_FIELDS and isinstance(normalised, list)) or (
                key in DICT_FIELDS and isinstance(normalised, dict)
            ):
                values.append(json.dumps(normalised))
            else:
                values.append(normalised)

        values.append(entry_id)
        sql = f"UPDATE memories SET {', '.join(set_parts)} WHERE id = ?"  # noqa: S608
        with backend._lock:
            backend._conn.execute(sql, values)
            if backend._skip_commit_depth == 0:
                backend._conn.commit()
        return get(backend, select_columns_sql, entry_id)
    except StorageError:
        raise
    except (sqlite3.Error, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise StorageError(
            f"Failed to update entry {entry_id}: {exc}",
            path=str(backend._db_path),
        ) from exc


def increment_session_counts(
    backend: SQLiteBackend,
    entry_ids: list[str],
    *,
    updated_at: datetime | None = None,
) -> int:
    """Increment session_count for multiple entries in one transaction."""
    if not entry_ids:
        return 0

    now = updated_at or datetime.now(timezone.utc)
    values = [(now.isoformat(), entry_id) for entry_id in entry_ids]

    try:
        sql = """
            UPDATE memories
            SET session_count = COALESCE(session_count, 0) + 1,
                updated_at = ?,
                sync_seq = COALESCE(sync_seq, 0) + 1,
                last_synced_at = NULL
            WHERE id = ?
        """
        with backend._lock:
            before = backend._conn.total_changes
            backend._conn.executemany(sql, values)
            backend._conn.commit()
            return int(backend._conn.total_changes - before)
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to increment session counts: {exc}",
            path=str(backend._db_path),
        ) from exc


def increment_access_counts(
    backend: SQLiteBackend,
    entry_ids: list[str],
    *,
    accessed_at: datetime | None = None,
) -> int:
    """Increment access_count and last_accessed_at for entries in one transaction."""
    if not entry_ids:
        return 0

    now = accessed_at or datetime.now(timezone.utc)
    values = [(now.isoformat(), now.isoformat(), entry_id) for entry_id in entry_ids]

    try:
        sql = """
            UPDATE memories
            SET access_count = COALESCE(access_count, 0) + 1,
                last_accessed_at = ?,
                updated_at = ?,
                sync_seq = COALESCE(sync_seq, 0) + 1,
                last_synced_at = NULL
            WHERE id = ?
        """
        with backend._lock:
            before = backend._conn.total_changes
            backend._conn.executemany(sql, values)
            backend._conn.commit()
            return int(backend._conn.total_changes - before)
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to increment access counts: {exc}",
            path=str(backend._db_path),
        ) from exc


def delete(backend: SQLiteBackend, entry_id: str) -> bool:
    """Remove an entry from memories (and vec_index when available)."""
    try:
        with backend._lock:
            cursor = backend._conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
            deleted = cursor.rowcount > 0
            if deleted and backend._vec_available:
                backend._delete_vector(entry_id)
            backend._conn.commit()
        logger.debug("memory_deleted", entry_id=entry_id, existed=deleted)
        return bool(deleted)
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to delete entry {entry_id}: {exc}",
            path=str(backend._db_path),
        ) from exc
