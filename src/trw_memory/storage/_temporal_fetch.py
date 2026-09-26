"""Bounded temporal selection with the existing row-quarantine decoder.

The caller supplies an authorized, ordered query without a pre-eligibility
LIMIT and holds the owning backend lock for the operation. This helper owns
its cursors, not the primary connection. No SQL date-policy copy or UDF needed.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Generator
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.temporal_selection import TemporalSelection
from trw_memory.storage._resilient_fetch import (
    FetchQuery,
    _ConnectionLike,
    _DBAPILike,
    _decode_bytes_rows,
    is_utf8_decode_error,
)
from trw_memory.storage._temporal_decode import TemporalMaterialization

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend


class _EntryFilterFailure(Exception):
    """Private callback boundary: never confuse predicate errors with decode failures."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        super().__init__(str(error))


class _StreamingCursor(Protocol):
    description: tuple[tuple[object, ...], ...] | None

    def fetchmany(self, size: int) -> list[tuple[object, ...]]: ...

    def close(self) -> None: ...


def _select_stream(
    connection: _ConnectionLike,
    query: FetchQuery,
    selection: TemporalSelection | None,
    limit: int,
    db_path: Path,
    batch_size: int,
    entry_filter: Callable[[MemoryEntry], bool] | None = None,
) -> tuple[list[MemoryEntry], int]:
    sql, params = query.build()
    cursor = cast("_StreamingCursor", connection.execute(sql, params))
    delta = 0
    materialization = TemporalMaterialization(selection, limit) if selection is not None else None

    def filter_entry(entry: MemoryEntry) -> bool:
        try:
            return entry_filter(entry) if entry_filter is not None else True
        except Exception as exc:
            raise _EntryFilterFailure(exc) from exc

    def entries() -> Generator[MemoryEntry, None, None]:
        nonlocal delta
        columns = tuple(str(column[0]) for column in (cursor.description or ()))
        while rows := cursor.fetchmany(batch_size):
            decoded, quarantined = _decode_bytes_rows(
                rows,
                column_names=columns,
                db_path=db_path,
                table=query.table,
                retain_row=materialization.retain if materialization else None,
                on_entry=materialization.accepted if materialization else None,
                entry_filter=filter_entry if entry_filter is not None else None,
                reference_time=selection.reference_time if selection else None,
            )
            delta += quarantined
            yield from decoded

    source = entries()
    try:
        selected = selection.select(source, limit=limit) if selection else list(islice(source, limit))
        return selected, delta
    finally:
        source.close()
        cursor.close()


def _fetch_selection(
    connection: _ConnectionLike,
    *,
    db_path: Path,
    dbapi: _DBAPILike,
    query: FetchQuery,
    selection: TemporalSelection | None,
    limit: int,
    batch_size: int = 256,
    entry_filter: Callable[[MemoryEntry], bool] | None = None,
) -> tuple[list[MemoryEntry], int]:
    """Select before limiting with bounded batches and exact recovery replay.

    A decode failure discards partial selection and restarts on a bytes-mode
    connection. Secondary failures propagate, never become a successful empty
    result. Quarantine delta describes the completed selection attempt. Schema
    quarantine logs emitted before a replay may repeat for those same rows.
    """
    if query.limit is not None:
        raise ValueError("Temporal selection query must not have a pre-eligibility limit")
    if limit <= 0 or batch_size <= 0:
        raise ValueError("Temporal selection limit and batch_size must be positive")
    try:
        return _select_stream(connection, query, selection, limit, db_path, batch_size, entry_filter)
    except (sqlite3.OperationalError, UnicodeDecodeError) as exc:
        if not is_utf8_decode_error(exc):
            raise
    # PRD-SEC-016 round-4 finding 3: routed through the identity-checked
    # helper, same as the sibling fallback in _resilient_fetch.py.
    from trw_memory.storage._connection import connect as _checked_connect

    secondary = _checked_connect(db_path, dbapi=dbapi, timeout=5.0, check_same_thread=True)
    try:
        secondary.text_factory = bytes
        return _select_stream(secondary, query, selection, limit, db_path, batch_size, entry_filter)
    finally:
        secondary.close()


def execute_temporal_query(
    backend: SQLiteBackend,
    query: FetchQuery,
    selection: TemporalSelection | None,
    *,
    limit: int,
    entry_filter: Callable[[MemoryEntry], bool] | None = None,
) -> list[MemoryEntry]:
    """Run on the backend's current owning connection and record quarantine delta."""
    try:
        with backend._lock:
            results, delta = _fetch_selection(
                backend._conn,
                db_path=backend._db_path,
                dbapi=backend._dbapi,
                query=query,
                selection=selection,
                limit=limit,
                entry_filter=entry_filter,
            )
            backend.quarantine_count_utf8 += delta
            return results
    except _EntryFilterFailure as exc:
        raise exc.error from None
    except (sqlite3.Error, ValueError, KeyError) as exc:
        raise StorageError("Failed temporal selection: " + str(exc), path=str(backend._db_path)) from exc
