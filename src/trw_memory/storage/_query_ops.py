"""SQLite query / list / namespace operations.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — ``SQLiteBackend.search`` etc. become 1-line delegators
that pass the backend handle.

7 helpers covering keyword search + count + assertion-bearing entries
+ list + namespace operations:

- ``search`` — keyword LIKE on id/content/detail/tags + filter clause +
  resilient row materialisation.
- ``find_active_by_content`` — embedding-independent exact-content dedup
  lookup (equality on content + detail, active + namespace scoped).
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

    # Push the tag filter into SQL so the LIMIT is applied AFTER tag filtering,
    # not before. Previously the SQL LIMIT truncated rows first and the
    # in-memory tag filter pruned that truncated set, so tag-scoped searches
    # under-delivered (returned fewer than top_k matching entries even when
    # more existed). Tags are stored as a JSON array (e.g. ["foo","bar"]); we
    # match each required tag as its JSON-quoted token to avoid substring
    # collisions ("foo" must not match ["foobar"]). The in-memory issubset
    # check below remains as the authoritative exact filter.
    tag_clauses: list[str] = []
    for tag in tags or []:
        tag_clauses.append("tags LIKE ? ESCAPE '\\'")
        escaped_tag = tag.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params.append(f'%"{escaped_tag}"%')
    if tag_clauses:
        where_sql = f"{where_sql} AND " + " AND ".join(tag_clauses)
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


def search_fts(
    backend: SQLiteBackend,
    select_columns_sql: str,
    *,
    query: str,
    top_k: int = 25,
    status: MemoryStatus | None = None,
    min_importance: float = 0.0,
    namespace: str | None = None,
) -> list[MemoryEntry]:
    """FTS5 full-text search — O(log N) inverted index candidate retrieval.

    Replaces the LIKE '%term%' table scan in :func:`search` for callers that
    have confirmed ``backend._fts_available``.  Results are ranked by FTS5
    BM25 for candidate retrieval; the caller's hybrid pipeline may re-rank.
    Falls back to an empty list when no FTS candidates match.
    """
    # Sanitize: strip whitespace, enforce max length, escape for FTS5 phrase query.
    # Phrase-quoting (wrapping in "...") makes FTS5 operators (AND, OR, NOT, NEAR)
    # and colon prefix operators literal; the empty guard and length cap add
    # DoS protection (empty/whitespace-only or pathologically long queries).
    query = query.strip()
    if not query:
        return []
    if len(query) > 1000:
        query = query[:1000]
    sanitized = query.replace('"', '""')
    fts_query = f'"{sanitized}"'
    filter_sql, filter_params = backend._build_filter_clause(
        status=status, namespace=namespace, min_importance=min_importance
    )
    over_fetch = min(top_k * 4, 500)
    backend._ensure_connection_fresh()
    try:
        with backend._lock:
            fts_rows = backend._conn.execute(
                "SELECT id FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, over_fetch),
            ).fetchall()
            if not fts_rows:
                return []
            ids = [str(row[0]) for row in fts_rows]
            placeholders = ", ".join(["?"] * len(ids))
            id_filter = f"id IN ({placeholders})"
            where_sql = id_filter if filter_sql == "1" else f"{id_filter} AND {filter_sql}"
            sql = (
                f"SELECT {select_columns_sql} FROM memories "  # noqa: S608
                f"WHERE {where_sql} ORDER BY importance DESC, updated_at DESC LIMIT ?"
            )
            params: list[object] = [*ids, *filter_params, top_k]
            fetch_query = backend._fetch_query(
                where_sql=where_sql,
                params=[*ids, *filter_params],
                order_by="importance DESC, updated_at DESC",
                limit=top_k,
            )
            return _execute_resilient(backend, sql, params, fetch_query=fetch_query)
    except (sqlite3.Error, ValueError, KeyError) as exc:
        raise StorageError(
            f"Failed FTS5 search: {exc}",
            path=str(backend._db_path),
        ) from exc


def find_active_by_content(
    backend: SQLiteBackend,
    content: str,
    detail: str,
    *,
    namespace: str = "default",
) -> str | None:
    """Return the id of an ACTIVE entry whose content + detail match exactly.

    Embedding-independent exact-content dedup (PRD-CORE-042): equality match
    on ``content`` and ``COALESCE(detail,'')`` within a namespace, scoped to
    ``status='active'``. Sub-millisecond at current scale; a ``content_hash``
    index is a future optimization (not added here to avoid a migration).

    Returns the first matching id, or None when no exact active duplicate
    exists. Read-only: never mutates.
    """
    sql = (
        "SELECT id FROM memories "
        "WHERE content = ? AND COALESCE(detail, '') = ? "
        "AND status = ? AND namespace = ? LIMIT 1"
    )
    params: tuple[object, ...] = (content, detail, MemoryStatus.ACTIVE.value, namespace)
    try:
        with backend._lock:
            row = backend._conn.execute(sql, params).fetchone()
        return str(row[0]) if row else None
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to look up active entry by content: {exc}",
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
    namespace: str | None = None,
    limit: int = 500,
) -> list[MemoryEntry]:
    """PRD-CORE-086 FR07 query for assertion-health summary.

    F7: defaults to ``status='active'`` so that stale assertions on
    OBSOLETE/ARCHIVED entries don't pollute the session-start assertion-health
    summary with false failures. Pass ``status=None`` to include every status.

    ``namespace`` scopes the query to a single namespace when provided. Without
    it the query spanned every namespace, leaking cross-namespace assertion
    rows into a session's assertion-health summary (memory-storage-1).

    ``limit`` caps the row scan (default 500). The summary only needs enough
    rows for aggregate stats, so an unbounded full-table scan on a large store
    is avoided (memory-storage-5).
    """
    where_sql = "assertions IS NOT NULL AND assertions != '[]'"
    params: tuple[object, ...] = ()
    if status is not None:
        where_sql = f"{where_sql} AND status = ?"
        params = (*params, status.value)
    if namespace is not None:
        where_sql = f"{where_sql} AND namespace = ?"
        params = (*params, namespace)
    order_by = "updated_at DESC"
    sql = (
        f"SELECT {select_columns_sql} FROM memories WHERE {where_sql} "  # noqa: S608
        f"ORDER BY {order_by} LIMIT ?"
    )
    exec_params: tuple[object, ...] = (*params, limit)
    fetch_query = backend._fetch_query(where_sql=where_sql, params=params, order_by=order_by, limit=limit)
    backend._ensure_connection_fresh()
    try:
        with backend._lock:
            return _execute_resilient(backend, sql, exec_params, fetch_query=fetch_query)
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
    min_importance: float = 0.0,
    limit: int = 100,
    exclude_superseded: bool = False,
    tags: list[str] | None = None,
) -> list[MemoryEntry]:
    """Return entries with optional filters, ordered by updated_at desc.

    When *exclude_superseded* is True, entries with a non-null ``invalid_from``
    value are excluded at the SQL layer rather than post-hoc.  This prevents
    superseded candidates from consuming slots in the BM25/dense candidate pool
    during hybrid retrieval.  Pass ``True`` when the caller has already decided
    that superseded entries are unwanted (e.g. ``include_superseded=False`` on
    the hybrid recall path without an ``as_of`` anchor).

    When *tags* is provided every listed entry must contain ALL of them. The
    predicate is pushed into SQL so the LIMIT applies AFTER tag filtering — a
    tagged entry past the row limit (older ``updated_at``) is still returned,
    rather than being truncated away before the filter runs (the recall-path
    silent-drop bug). Tags are stored as a JSON array (e.g. ``["foo","bar"]``);
    each required tag matches its JSON-quoted token so ``"foo"`` does not match
    ``["foobar"]``. An exact ``issubset`` re-check below remains authoritative.
    """
    where_sql, params = backend._build_filter_clause(status=status, namespace=namespace, min_importance=min_importance)
    if exclude_superseded:
        where_sql = f"({where_sql}) AND (invalid_from IS NULL OR invalid_from = '')"
    tag_clauses: list[str] = []
    for tag in tags or []:
        tag_clauses.append("tags LIKE ? ESCAPE '\\'")
        escaped_tag = tag.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params.append(f'%"{escaped_tag}"%')
    if tag_clauses:
        where_sql = f"({where_sql}) AND " + " AND ".join(tag_clauses)
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
            results = _execute_resilient(backend, sql, params, fetch_query=fetch_query)
    except (sqlite3.Error, ValueError, KeyError) as exc:
        raise StorageError(
            f"Failed to list entries: {exc}",
            path=str(backend._db_path),
        ) from exc
    if tags:
        # Authoritative exact filter: the SQL LIKE on the JSON token narrows the
        # candidate set (so the LIMIT bites the filtered rows) but issubset is
        # the source of truth for "entry has ALL required tags".
        required = set(tags)
        results = [e for e in results if required.issubset(set(e.tags))]
    return results


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
    """Delete all entries in a namespace.

    Commit is suppressed when called inside a ``transaction()`` block
    (``_skip_commit_depth > 0``) so the memories DELETE batches with the
    companion wiki_refs / vector cleanup into the outer COMMIT — see
    ``SQLiteBackend.delete_by_namespace`` for the atomic wrapper.
    """
    try:
        with backend._lock:
            cursor = backend._conn.execute("DELETE FROM memories WHERE namespace = ?", (namespace,))
            deleted = cursor.rowcount
            # Remove FTS5 ghost rows: FTS5 does not cascade from the memories
            # DELETE, so orphan rows accumulate and inflate search results.
            # Anti-join against the remaining memories table removes exactly
            # the rows that were just deleted, regardless of namespace.
            if getattr(backend, "_fts_available", False) and deleted > 0:
                backend._conn.execute(
                    "DELETE FROM memories_fts WHERE id NOT IN (SELECT id FROM memories)"
                )
            if backend._skip_commit_depth == 0:
                backend._conn.commit()
        logger.debug("namespace_deleted", namespace=namespace, entries_deleted=deleted)
        return int(deleted)
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to delete namespace {namespace!r}: {exc}",
            path=str(backend._db_path),
        ) from exc
