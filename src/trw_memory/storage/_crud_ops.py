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

import contextlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
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

# Upper bound for the monotonic recall/session/access counters. These only
# ever increment (one per recall/session), so an adversary replaying access
# could otherwise grow them without limit and skew utility/decay scoring. The
# cap is far above any legitimate usage and keeps the values comfortably within
# SQLite's signed-64-bit integer range.
_MAX_COUNTER = 1_000_000_000

# F11: statuses that retire an entry from active recall. When an update
# transitions an entry into one of these, its dense vector is removed from
# the KNN index so stale vectors stop polluting dense recall. The entry row
# itself is preserved for audit (no hard delete).
_TERMINAL_STATUS_VALUES: frozenset[str] = frozenset(
    {
        MemoryStatus.OBSOLETE.value,
        MemoryStatus.OBSOLETE_POISONED.value,
        MemoryStatus.ARCHIVED.value,
        MemoryStatus.RESOLVED.value,
    }
)


def _normalise_status_value(value: object) -> str | None:
    """Return the string status value for an update field, or None if absent/odd."""
    if isinstance(value, MemoryStatus):
        return value.value
    if isinstance(value, str):
        return value
    return None


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
            if getattr(backend, "_fts_available", False):
                tags_json = json.dumps(entry.tags) if isinstance(entry.tags, list) else (entry.tags or "[]")
                # FTS5 virtual tables don't enforce uniqueness — delete first to
                # avoid stale rows when store() is called as INSERT OR REPLACE.
                backend._conn.execute("DELETE FROM memories_fts WHERE id = ?", (entry.id,))
                backend._conn.execute(
                    "INSERT INTO memories_fts(id, content, detail, tags) VALUES (?, ?, ?, ?)",
                    (entry.id, entry.content, entry.detail or "", tags_json),
                )
            # S9 fix: suppress the commit when inside a ``transaction()`` block
            # so a store() batched with other writes commits exactly once at
            # the outermost COMMIT — matching update()/increment_recall_access.
            if backend._skip_commit_depth == 0:
                backend._conn.commit()
        logger.debug("memory_stored", entry_id=entry.id)
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        raise StorageError(
            f"Failed to store entry {entry.id}: {exc}",
            path=str(backend._db_path),
        ) from exc


def store_many(
    backend: SQLiteBackend,
    insert_columns_sql: str,
    columns: tuple[str, ...],
    entries: list[MemoryEntry],
) -> int:
    """Bulk-insert a list of entries in one SQLite transaction using executemany.

    Returns the number of entries successfully stored. Uses ``INSERT OR REPLACE``
    so duplicate ids overwrite existing rows — same semantics as :func:`store`
    but batched for 90x throughput gain vs per-row stores (99K/sec vs 1.1K/sec
    at 5K entries on WAL-mode SQLite).

    Sync bookkeeping (sync_seq + sync_hash) is applied per-entry before the
    batch insert. FTS5 dual-write is done in two extra ``executemany`` calls so
    the full batch lands in one ``BEGIN IMMEDIATE / COMMIT``.
    """
    if not entries:
        return 0

    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT OR REPLACE INTO memories ({insert_columns_sql}) VALUES ({placeholders})"  # noqa: S608

    now = datetime.now(timezone.utc)
    for entry in entries:
        if not entry.created_at:
            entry.created_at = now
        if not entry.updated_at:
            entry.updated_at = now
        entry.sync_seq = (entry.sync_seq or 0) + 1
        entry.sync_hash = DeltaTracker.compute_sync_hash(entry)
        entry.last_synced_at = None

    rows = [entry_to_row(e) for e in entries]

    try:
        with backend._lock:
            backend._conn.execute("BEGIN IMMEDIATE")
            backend._conn.executemany(sql, rows)
            if getattr(backend, "_fts_available", False):
                backend._conn.executemany(
                    "DELETE FROM memories_fts WHERE id = ?",
                    [(e.id,) for e in entries],
                )
                backend._conn.executemany(
                    "INSERT INTO memories_fts(id, content, detail, tags) VALUES (?, ?, ?, ?)",
                    [
                        (e.id, e.content, e.detail or "", json.dumps(e.tags) if isinstance(e.tags, list) else (e.tags or "[]"))
                        for e in entries
                    ],
                )
            backend._conn.commit()
        logger.debug("memory_batch_stored", count=len(entries))
        return len(entries)
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        with contextlib.suppress(sqlite3.Error):
            backend._conn.execute("ROLLBACK")
        raise StorageError(
            f"Failed to batch-store {len(entries)} entries: {exc}",
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

        # Derive FTS content BEFORE the serialize loop converts lists to JSON strings.
        _fts_content: str = str(field_dict.get("content", existing.content))
        _fts_detail: str = str(field_dict.get("detail", existing.detail or ""))
        _raw_tags = field_dict.get("tags", existing.tags or [])
        _fts_tags: str = json.dumps(_raw_tags) if isinstance(_raw_tags, list) else str(_raw_tags)

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

        # F11: detect a transition INTO a terminal (non-active) status so we
        # can prune the now-stale dense vector after the row update lands.
        new_status = _normalise_status_value(fields["status"]) if "status" in fields else None
        prune_vector = (
            new_status is not None and new_status in _TERMINAL_STATUS_VALUES and existing.status != new_status
        )

        values.append(entry_id)
        sql = f"UPDATE memories SET {', '.join(set_parts)} WHERE id = ?"  # noqa: S608
        with backend._lock:
            backend._conn.execute(sql, values)
            if prune_vector and backend._vec_available:
                # Keep the entry row for audit; drop only the KNN vector so
                # dense recall stops returning the retired entry.
                backend._delete_vector(entry_id)
            if getattr(backend, "_fts_available", False):
                backend._conn.execute("DELETE FROM memories_fts WHERE id = ?", (entry_id,))
                backend._conn.execute(
                    "INSERT INTO memories_fts(id, content, detail, tags) VALUES (?, ?, ?, ?)",
                    (entry_id, _fts_content, _fts_detail, _fts_tags),
                )
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
        sql = f"""
            UPDATE memories
            SET session_count = MIN(COALESCE(session_count, 0) + 1, {_MAX_COUNTER}),
                updated_at = ?,
                sync_seq = COALESCE(sync_seq, 0) + 1,
                last_synced_at = NULL
            WHERE id = ?
        """  # noqa: S608 — _MAX_COUNTER is a module-level int constant, not user input.
        with backend._lock:
            before = backend._conn.total_changes
            backend._conn.executemany(sql, values)
            # Suppress the commit inside a ``transaction()`` block so this
            # batches into the caller's outermost COMMIT instead of prematurely
            # committing their open transaction (matches store()/update()).
            if backend._skip_commit_depth == 0:
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
        sql = f"""
            UPDATE memories
            SET access_count = MIN(COALESCE(access_count, 0) + 1, {_MAX_COUNTER}),
                last_accessed_at = ?,
                updated_at = ?,
                sync_seq = COALESCE(sync_seq, 0) + 1,
                last_synced_at = NULL
            WHERE id = ?
        """  # noqa: S608 — _MAX_COUNTER is a module-level int constant, not user input.
        with backend._lock:
            before = backend._conn.total_changes
            backend._conn.executemany(sql, values)
            # Defer the commit inside a ``transaction()`` block (see
            # increment_session_counts) to preserve caller transaction atomicity.
            if backend._skip_commit_depth == 0:
                backend._conn.commit()
            return int(backend._conn.total_changes - before)
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to increment access counts: {exc}",
            path=str(backend._db_path),
        ) from exc


def increment_recall_access(
    backend: SQLiteBackend,
    entry_ids: list[str],
    *,
    accessed_at: datetime | None = None,
) -> int:
    """F-008: batch recall bookkeeping in ONE UPDATE / one commit.

    Increments ``access_count`` AND ``recall_count`` and stamps
    ``last_accessed_at`` for every id in ``entry_ids`` using a single
    ``WHERE id IN (...)`` statement — replacing the per-entry get+update loop
    (2 statements + 1 WAL append per entry) that amplified WAL writes on every
    recall. De-duplicates ids so each entry is incremented at most once per
    call (matching the prior loop's per-id semantics). Returns rows updated.
    """
    if not entry_ids:
        return 0

    # Preserve order while de-duplicating (each entry counted once per call).
    unique_ids = list(dict.fromkeys(entry_ids))
    now = accessed_at or datetime.now(timezone.utc)
    now_iso = now.isoformat()
    placeholders = ", ".join(["?"] * len(unique_ids))
    sql = f"""
        UPDATE memories
        SET access_count = MIN(COALESCE(access_count, 0) + 1, {_MAX_COUNTER}),
            recall_count = MIN(COALESCE(recall_count, 0) + 1, {_MAX_COUNTER}),
            last_accessed_at = ?,
            updated_at = ?,
            sync_seq = COALESCE(sync_seq, 0) + 1,
            last_synced_at = NULL
        WHERE id IN ({placeholders})
    """  # noqa: S608 — placeholders are positional binds; _MAX_COUNTER is an int constant.
    params: list[object] = [now_iso, now_iso, *unique_ids]
    try:
        with backend._lock:
            before = backend._conn.total_changes
            backend._conn.execute(sql, params)
            if backend._skip_commit_depth == 0:
                backend._conn.commit()
            return int(backend._conn.total_changes - before)
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to increment recall access: {exc}",
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
            if deleted and getattr(backend, "_fts_available", False):
                backend._conn.execute("DELETE FROM memories_fts WHERE id = ?", (entry_id,))
            # Defer the commit inside a ``transaction()`` block so the row +
            # vector deletes batch into the caller's outermost COMMIT rather
            # than prematurely committing their open transaction.
            if backend._skip_commit_depth == 0:
                backend._conn.commit()
        logger.debug("memory_deleted", entry_id=entry_id, existed=deleted)
        return bool(deleted)
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to delete entry {entry_id}: {exc}",
            path=str(backend._db_path),
        ) from exc
