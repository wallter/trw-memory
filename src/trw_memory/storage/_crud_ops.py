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
from collections.abc import Sequence
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
from trw_memory.storage._sql_utils import iter_bind_chunks
from trw_memory.storage._utf8_validator import validate_entry_utf8, validate_utf8_fields
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


def _fts_row(entry: MemoryEntry) -> tuple[str, str, str, str, str]:
    """Return the ``memories_fts`` row tuple for *entry* (namespace-qualified)."""
    tags_json = json.dumps(entry.tags) if isinstance(entry.tags, list) else (entry.tags or "[]")
    return (entry.id, entry.namespace, entry.content, entry.detail or "", tags_json)


def _replace_tag_postings(backend: SQLiteBackend, namespace: str, entry_id: str, tags: Sequence[str]) -> None:
    """Re-point the ``memory_tags`` inverted index at *entry_id*'s current tags.

    PRD-CORE-245 FR07: this index is what the bounded tag derivation queries in
    place of the 98,288 materialised ``tag_cooccurrence`` edges the schema-5
    migration deleted. It is maintained beside the FTS row on every write and
    pruned beside the edges on every delete, so it can never drift from the
    ``tags`` column it mirrors.
    """
    backend._conn.execute(
        "DELETE FROM memory_tags WHERE namespace = ? AND entry_id = ?",
        (namespace, entry_id),
    )
    rows = [(namespace, tag, entry_id) for tag in dict.fromkeys(tags) if str(tag).strip()]
    if rows:
        backend._conn.executemany("INSERT OR IGNORE INTO memory_tags(namespace, tag, entry_id) VALUES(?, ?, ?)", rows)


def purge_tag_postings_for(backend: SQLiteBackend, namespace: str, entry_ids: Sequence[str]) -> None:
    """Drop every ``memory_tags`` posting for *entry_ids* within *namespace*.

    CONTRACT: mirrors :func:`purge_edges_for` — the caller already holds
    ``backend._lock`` and owns the commit.
    """
    if not entry_ids:
        return
    for chunk in iter_bind_chunks(list(entry_ids), reserved_bindings=1):
        placeholders = ",".join("?" for _ in chunk)
        backend._conn.execute(
            f"DELETE FROM memory_tags WHERE namespace = ? AND entry_id IN ({placeholders})",  # noqa: S608 — placeholders is ? repeated; ids are parameterized
            (namespace, *chunk),
        )


def store(
    backend: SQLiteBackend,
    insert_columns_sql: str,
    columns: tuple[str, ...],
    entry: MemoryEntry,
) -> None:
    """INSERT OR REPLACE the entry into the memories table."""
    validate_entry_utf8(entry)

    entry.sync_seq = (entry.sync_seq or 0) + 1
    entry.sync_hash = DeltaTracker.compute_sync_hash(entry)
    entry.last_synced_at = None

    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT OR REPLACE INTO memories ({insert_columns_sql}) VALUES ({placeholders})"  # noqa: S608
    try:
        with backend._lock:
            backend._conn.execute(sql, entry_to_row(entry))
            if getattr(backend, "_fts_available", False):
                # FTS5 virtual tables don't enforce uniqueness — delete first to
                # avoid stale rows when store() is called as INSERT OR REPLACE.
                # PRD-CORE-245 FR02: the delete is namespace-qualified, or
                # storing (nsA, X) would destroy the FTS row of (nsB, X).
                backend._conn.execute(
                    "DELETE FROM memories_fts WHERE id = ? AND namespace = ?",
                    (entry.id, entry.namespace),
                )
                backend._conn.execute(
                    "INSERT INTO memories_fts(id, namespace, content, detail, tags) VALUES (?, ?, ?, ?, ?)",
                    _fts_row(entry),
                )
            _replace_tag_postings(backend, entry.namespace, entry.id, entry.tags)
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
    so duplicate ids overwrite existing rows, matching :func:`store` semantics.

    Sync bookkeeping (sync_seq + sync_hash) is applied per-entry before the
    batch insert. FTS5 dual-write is done in two extra ``executemany`` calls so
    the full batch lands in one ``BEGIN IMMEDIATE / COMMIT``.
    """
    if not entries:
        return 0

    for entry in entries:
        validate_entry_utf8(entry)

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
    # INSERT OR REPLACE makes the last occurrence authoritative. Mirror that
    # result in FTS5, whose virtual table does not enforce id uniqueness.
    final_entries = list({(entry.namespace, entry.id): entry for entry in entries}.values())

    try:
        with backend._lock:
            # Only open our own transaction when not already inside a
            # ``transaction()`` block.  When _skip_commit_depth > 0 the
            # outer context manager owns the BEGIN/COMMIT; we must not
            # issue a nested BEGIN IMMEDIATE (which would raise
            # "cannot start a transaction within a transaction") or
            # commit prematurely (which would close the outer transaction).
            _own_txn = backend._skip_commit_depth == 0
            if _own_txn:
                backend._conn.execute("BEGIN IMMEDIATE")
            backend._conn.executemany(sql, rows)
            fts_batch = getattr(backend, "_fts_available", False)
            if fts_batch:
                backend._conn.executemany(
                    "DELETE FROM memories_fts WHERE id = ? AND namespace = ?",
                    [(entry.id, entry.namespace) for entry in final_entries],
                )
                backend._conn.executemany(
                    "INSERT INTO memories_fts(id, namespace, content, detail, tags) VALUES (?, ?, ?, ?, ?)",
                    [_fts_row(e) for e in final_entries],
                )
            for entry in final_entries:
                _replace_tag_postings(backend, entry.namespace, entry.id, entry.tags)
            # Merge FTS5 index segments after bulk load to prevent fragmentation.
            # Only worthwhile for large batches — optimize() walks the whole index,
            # so the merge cost only pays off when many segments were just appended.
            if fts_batch and len(entries) >= 100:
                backend._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('optimize')")
            if _own_txn:
                backend._conn.commit()
        logger.debug("memory_batch_stored", count=len(entries))
        return len(entries)
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        # The ``with backend._lock`` above has already released by the time we
        # reach here, so the rollback must RE-ACQUIRE the lock — otherwise a
        # concurrent writer can interleave on the single shared connection
        # between our failure and the ROLLBACK (mirrors _transaction.py's
        # locked rollback). The backend's RLock is no longer held at this point,
        # so re-acquiring it keeps rollback serialized with other connection use.
        if backend._skip_commit_depth == 0:
            with backend._lock, contextlib.suppress(sqlite3.Error):
                backend._conn.rollback()
        raise StorageError(
            f"Failed to batch-store {len(entries)} entries: {exc}",
            path=str(backend._db_path),
        ) from exc


def get(
    backend: SQLiteBackend,
    select_columns_sql: str,
    entry_id: str,
    namespace: str,
) -> MemoryEntry | None:
    """Retrieve the ``(namespace, id)``-identified entry.

    PRD-CORE-245 FR03: identity is composite under schema 5, so a bare id is
    ambiguous and the namespace is required rather than defaulted — a default
    would silently reinstate exactly the ambiguity the composite key removes.
    """
    sql = f"SELECT {select_columns_sql} FROM memories WHERE namespace = ? AND id = ?"  # noqa: S608
    try:
        with backend._lock:
            row = backend._conn.execute(sql, (namespace, entry_id)).fetchone()
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
    namespace: str,
    **fields: object,
) -> MemoryEntry | None:
    """Apply a partial update to the ``(namespace, entry_id)``-identified entry.

    PRD-CORE-245 FR03: required, for the same reason ``get`` and ``delete``
    require it. Resolving the row by bare id and taking whichever one came back
    first was nondeterministic the moment a legitimate cross-namespace id
    collision existed -- which is exactly the state the composite key makes
    reachable.
    """
    if not fields:
        return get(backend, select_columns_sql, entry_id, namespace)

    existing = get(backend, select_columns_sql, entry_id, namespace)
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

        utf8_fields = dict(field_dict)
        if "expires" in utf8_fields:
            utf8_fields["expires_at"] = utf8_fields.pop("expires")
        utf8_fields["id"] = entry_id
        validate_utf8_fields(utf8_fields)

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

        values.extend([namespace, entry_id])
        sql = f"UPDATE memories SET {', '.join(set_parts)} WHERE namespace = ? AND id = ?"  # noqa: S608
        with backend._lock:
            backend._conn.execute(sql, values)
            if prune_vector and backend._vec_available:
                # Keep the entry row for audit; drop only the KNN vector so
                # dense recall stops returning the retired entry.
                backend._delete_vector(entry_id, namespace)
            if getattr(backend, "_fts_available", False):
                backend._conn.execute(
                    "DELETE FROM memories_fts WHERE id = ? AND namespace = ?",
                    (entry_id, namespace),
                )
                backend._conn.execute(
                    "INSERT INTO memories_fts(id, namespace, content, detail, tags) VALUES (?, ?, ?, ?, ?)",
                    (entry_id, namespace, _fts_content, _fts_detail, _fts_tags),
                )
            _replace_tag_postings(backend, namespace, entry_id, json.loads(_fts_tags) if _fts_tags else [])
            if backend._skip_commit_depth == 0:
                backend._conn.commit()
        return get(backend, select_columns_sql, entry_id, namespace)
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
        if backend._skip_commit_depth == 0:
            with backend._lock, contextlib.suppress(sqlite3.Error):
                backend._conn.rollback()
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
        if backend._skip_commit_depth == 0:
            with backend._lock, contextlib.suppress(sqlite3.Error):
                backend._conn.rollback()
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
    try:
        with backend.transaction(), backend._lock:
            before = backend._conn.total_changes
            for chunk in iter_bind_chunks(unique_ids, reserved_bindings=2):
                placeholders = ", ".join(["?"] * len(chunk))
                sql = f"""
                    UPDATE memories
                    SET access_count = MIN(COALESCE(access_count, 0) + 1, {_MAX_COUNTER}),
                        recall_count = MIN(COALESCE(recall_count, 0) + 1, {_MAX_COUNTER}),
                        last_accessed_at = ?,
                        updated_at = ?,
                        sync_seq = COALESCE(sync_seq, 0) + 1,
                        last_synced_at = NULL
                    WHERE id IN ({placeholders})
                """  # noqa: S608
                backend._conn.execute(sql, [now_iso, now_iso, *chunk])
            return int(backend._conn.total_changes - before)
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to increment recall access: {exc}",
            path=str(backend._db_path),
        ) from exc


# memory_graph_edges binds each purged id twice (source + target), so chunking
# keeps each statement below SQLite's conservative bind ceiling. All deletes
# run inside the caller's held lock and transaction, so the net effect remains
# atomic.


def purge_edges_for(backend: SQLiteBackend, entry_ids: Sequence[str], namespace: str) -> None:
    """Delete knowledge-graph edges referencing any of *entry_ids* in *namespace*.

    ``memory_graph_edges`` declares no FK cascade (``PRAGMA foreign_keys`` is
    off process-wide), so orphan edges must be pruned explicitly whenever their
    endpoint rows are deleted. A single ``DELETE`` with ``OR`` covers both sides
    of a directed edge. Shared by the per-row :func:`delete` and the bulk
    ``delete_by_namespace`` paths — one source of truth for edge cleanup.

    CONTRACT: the caller MUST already hold ``backend._lock`` and own the commit.
    This helper neither acquires the lock nor commits, so the edge purge batches
    into the caller's outermost ``COMMIT`` and preserves transaction atomicity
    (test_storage_transaction_atomicity.py).
    """
    if not entry_ids:
        return
    ids = list(entry_ids)
    for chunk in iter_bind_chunks(ids, bindings_per_item=2, reserved_bindings=1):
        placeholders = ",".join("?" for _ in chunk)
        backend._conn.execute(
            f"DELETE FROM memory_graph_edges WHERE namespace = ? AND (source_id IN ({placeholders}) "  # noqa: S608 — placeholders is ? repeated; ids are parameterized values
            f"OR target_id IN ({placeholders}))",
            (namespace, *chunk, *chunk),
        )


def purge_orphan_edges(backend: SQLiteBackend) -> None:
    """Delete every edge whose source or target row no longer exists.

    :func:`purge_edges_for` is namespace-qualified (PRD-CORE-245 FR02) because a
    per-row delete knows exactly which namespace it is addressing. A bulk
    ``delete_by_namespace`` needs the complementary guarantee: an edge that
    names a row which is now gone is dangling regardless of which namespace the
    edge itself is filed under, and a BFS that follows it lands on a ghost node.
    This is the edge-table twin of the ``memories_fts`` ghost-row anti-join.

    CONTRACT: caller holds ``backend._lock`` and owns the commit.
    """
    backend._conn.execute(
        "DELETE FROM memory_graph_edges WHERE NOT EXISTS ("
        "  SELECT 1 FROM memories m WHERE m.namespace = memory_graph_edges.namespace"
        "    AND m.id = memory_graph_edges.source_id"
        ") OR NOT EXISTS ("
        "  SELECT 1 FROM memories m WHERE m.namespace = memory_graph_edges.namespace"
        "    AND m.id = memory_graph_edges.target_id"
        ")"
    )


def delete(backend: SQLiteBackend, entry_id: str, namespace: str) -> bool:
    """Remove the ``(namespace, id)``-identified entry and every sidecar row.

    PRD-CORE-245 FR03: the namespace is required. Under the composite key an
    unqualified delete would take the row out of whichever namespace happened
    to hold that id, and FR02's sidecar deletes below would follow it.
    """
    try:
        with backend._lock:
            cursor = backend._conn.execute("DELETE FROM memories WHERE namespace = ? AND id = ?", (namespace, entry_id))
            deleted = cursor.rowcount > 0
            if deleted and backend._vec_available:
                backend._delete_vector(entry_id, namespace)
            if deleted and getattr(backend, "_fts_available", False):
                backend._conn.execute("DELETE FROM memories_fts WHERE id = ? AND namespace = ?", (entry_id, namespace))
            # Remove knowledge-graph edges and tag postings that reference the
            # deleted entry (see purge_edges_for — one source of truth shared
            # with delete_by_namespace's bulk cleanup).
            if deleted:
                purge_edges_for(backend, (entry_id,), namespace)
                purge_tag_postings_for(backend, namespace, (entry_id,))
            # Defer the commit inside a ``transaction()`` block so the row +
            # vector deletes batch into the caller's outermost COMMIT rather
            # than prematurely committing their open transaction.
            if backend._skip_commit_depth == 0:
                backend._conn.commit()
        logger.debug("memory_deleted", entry_id=entry_id, existed=deleted)
        return bool(deleted)
    except sqlite3.Error as exc:
        # delete() runs FOUR statements (memories + vector + FTS + graph edges)
        # before commit. If commit (or any later statement) raises, the lock has
        # already released, so a concurrent writer's commit could persist this
        # partially-applied delete — leaving e.g. the memories row gone but
        # FTS/vec/graph rows behind. Roll back under a re-acquired lock, mirroring
        # store_many() (trw-memory-13). Suppressed only inside an outer
        # transaction() block, which owns the rollback.
        if backend._skip_commit_depth == 0:
            with backend._lock, contextlib.suppress(sqlite3.Error):
                backend._conn.rollback()
        raise StorageError(
            f"Failed to delete entry {entry_id}: {exc}",
            path=str(backend._db_path),
        ) from exc
