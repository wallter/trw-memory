"""SQLite query / list / namespace operations.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — ``SQLiteBackend.search`` etc. become 1-line delegators
that pass the backend handle.

7 helpers covering keyword search + count + assertion-bearing entries
+ list + namespace operations:

- ``search`` — keyword LIKE on id/content/detail/tags + filter clause +
  resilient row materialisation.
- ``count`` — namespace-scoped or global COUNT(*).
- ``entries_with_assertions`` — PRD-CORE-086 FR07 query for
  ``trw_session_start`` assertion-health summary.
- ``count_with_assertions`` — backward-compat alias.
- ``list_entries`` — filter-clause + ORDER BY updated_at DESC.
- ``list_namespaces`` — distinct namespace query.
- ``delete_by_namespace`` — DELETE WHERE namespace = ?.

Each helper takes a ``backend`` argument exposing the instance state
(_conn, _lock, _db_path, _build_filter_clause, _ensure_connection_fresh,
_fetch_rows_resilient).

Extracted as PRD-DIST-245 Phase 1 batch 86.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage._resilient_fetch import is_utf8_decode_error

if TYPE_CHECKING:
    from collections.abc import Sequence

    from trw_memory.storage._resilient_fetch import FetchQuery
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)


def _execute_resilient(
    backend: SQLiteBackend,
    sql: str,
    params: Sequence[object],
    *,
    fetch_query: FetchQuery,
) -> list[MemoryEntry]:
    """Execute *sql* and materialise rows with UTF-8 quarantine resilience.

    Must be called while holding ``backend._lock``. On SQLite >= 3.51 the
    driver decodes TEXT during ``execute()``, so a UTF-8 decode error can
    surface here rather than during fetch — both paths route to the
    bytes-mode fallback, which re-executes ``fetch_query`` (preserving the
    WHERE filter, ORDER BY, and LIMIT). Non-decode errors propagate.
    """
    try:
        cursor = backend._conn.execute(sql, params)
    except (sqlite3.OperationalError, UnicodeDecodeError) as exc:
        if not is_utf8_decode_error(exc):
            raise
        return backend._fetch_rows_via_bytes_fallback(query=fetch_query)
    return backend._fetch_rows_resilient(cursor, query=fetch_query)


def search(
    backend: SQLiteBackend,
    select_columns_sql: str,
    *,
    query: str,
    top_k: int = 25,
    tags: list[str] | None = None,
    status: MemoryStatus | None = None,
    min_importance: float = 0.0,
    namespace: str | None = None,
) -> list[MemoryEntry]:
    """Keyword LIKE search on content + detail + tags with filters."""
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    like_term = f"%{escaped}%"
    like_clause = (
        "(id LIKE ? ESCAPE '\\' OR content LIKE ? ESCAPE '\\' OR detail LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\')"
    )
    like_params: list[object] = [like_term, like_term, like_term, like_term]
    filter_sql, filter_params = backend._build_filter_clause(
        status=status, namespace=namespace, min_importance=min_importance
    )
    where_sql = like_clause if filter_sql == "1" else f"{like_clause} AND {filter_sql}"
    params: list[object] = like_params + filter_params
    order_by = "importance DESC, updated_at DESC"
    sql = (
        f"SELECT {select_columns_sql} FROM memories WHERE {where_sql} "  # noqa: S608
        f"ORDER BY {order_by} LIMIT ?"
    )
    params.append(top_k)
    fetch_query = backend._fetch_query(where_sql=where_sql, params=params[:-1], order_by=order_by, limit=top_k)

    backend._ensure_connection_fresh()

    try:
        with backend._lock:
            results = _execute_resilient(backend, sql, params, fetch_query=fetch_query)
        if tags:
            required = set(tags)
            results = [e for e in results if required.issubset(set(e.tags))]
        return results[:top_k]
    except (sqlite3.Error, ValueError, KeyError) as exc:
        raise StorageError(
            f"Failed to search memories: {exc}",
            path=str(backend._db_path),
        ) from exc


def count(backend: SQLiteBackend, namespace: str | None = None) -> int:
    """Return the number of stored entries."""
    try:
        with backend._lock:
            if namespace is not None:
                row = backend._conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE namespace = ?", (namespace,)
                ).fetchone()
            else:
                row = backend._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to count memories: {exc}",
            path=str(backend._db_path),
        ) from exc


def entries_with_assertions(
    backend: SQLiteBackend,
    select_columns_sql: str,
    *,
    status: MemoryStatus | None = MemoryStatus.ACTIVE,
) -> list[MemoryEntry]:
    """PRD-CORE-086 FR07 query for assertion-health summary.

    F7: defaults to ``status='active'`` so that stale assertions on
    OBSOLETE/ARCHIVED entries don't pollute the session-start assertion-health
    summary with false failures. Pass ``status=None`` to include every status.
    """
    where_sql = "assertions IS NOT NULL AND assertions != '[]'"
    params: tuple[object, ...] = ()
    if status is not None:
        where_sql = f"{where_sql} AND status = ?"
        params = (status.value,)
    sql = f"SELECT {select_columns_sql} FROM memories WHERE {where_sql}"  # noqa: S608
    fetch_query = backend._fetch_query(where_sql=where_sql, params=params, order_by="updated_at DESC")
    backend._ensure_connection_fresh()
    try:
        with backend._lock:
            return _execute_resilient(backend, sql, params, fetch_query=fetch_query)
    except sqlite3.Error as exc:
        logger.debug("entries_with_assertions_query_failed", exc_info=True)
        raise StorageError(
            f"Failed to query entries with assertions: {exc}",
            path=str(backend._db_path),
        ) from exc


def list_entries(
    backend: SQLiteBackend,
    select_columns_sql: str,
    *,
    status: MemoryStatus | None = None,
    namespace: str | None = None,
    limit: int = 100,
) -> list[MemoryEntry]:
    """Return entries with optional filters, ordered by updated_at desc."""
    where_sql, params = backend._build_filter_clause(status=status, namespace=namespace)
    order_by = "updated_at DESC"
    sql = (
        f"SELECT {select_columns_sql} FROM memories WHERE {where_sql} "  # noqa: S608
        f"ORDER BY {order_by} LIMIT ?"
    )
    filter_params = list(params)
    params.append(limit)
    fetch_query = backend._fetch_query(where_sql=where_sql, params=filter_params, order_by=order_by, limit=limit)
    backend._ensure_connection_fresh()
    try:
        with backend._lock:
            return _execute_resilient(backend, sql, params, fetch_query=fetch_query)
    except (sqlite3.Error, ValueError, KeyError) as exc:
        raise StorageError(
            f"Failed to list entries: {exc}",
            path=str(backend._db_path),
        ) from exc


def list_namespaces(backend: SQLiteBackend) -> list[str]:
    """Return all distinct namespaces that have stored entries."""
    try:
        with backend._lock:
            rows = backend._conn.execute("SELECT DISTINCT namespace FROM memories ORDER BY namespace").fetchall()
        return [str(row[0]) for row in rows]
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to list namespaces: {exc}",
            path=str(backend._db_path),
        ) from exc


def delete_by_namespace(backend: SQLiteBackend, namespace: str) -> int:
    """Delete all entries in a namespace."""
    try:
        with backend._lock:
            cursor = backend._conn.execute("DELETE FROM memories WHERE namespace = ?", (namespace,))
            deleted = cursor.rowcount
            backend._conn.commit()
        logger.debug("namespace_deleted", namespace=namespace, entries_deleted=deleted)
        return int(deleted)
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to delete namespace {namespace!r}: {exc}",
            path=str(backend._db_path),
        ) from exc
