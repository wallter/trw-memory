"""SQLite FTS acquisition: public bound facade and legacy/policy query paths."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage._query_ops import _execute_resilient

if TYPE_CHECKING:
    from trw_memory.retrieval.temporal_selection import TemporalSelection
    from trw_memory.storage.sqlite_backend import SQLiteBackend


def search_fts_method(
    self: SQLiteBackend,
    query: str,
    *,
    top_k: int = 25,
    status: MemoryStatus | None = None,
    min_importance: float = 0.0,
    namespace: str | None = None,
    temporal_selection: TemporalSelection | None = None,
    entry_filter: Callable[[MemoryEntry], bool] | None = None,
) -> list[MemoryEntry]:
    """FTS5 full-text search — O(log N) inverted-index candidate retrieval.

    Raises :class:`~trw_memory.exceptions.StorageError` on SQLite failure.
    Returns an empty list when FTS5 is unavailable or no entries match.
    """
    from trw_memory.storage.sqlite_backend import _SELECT_COLUMNS_SQL

    if not self._fts_available:
        return []
    with self._fresh_connection():
        return search_fts(
            self,
            _SELECT_COLUMNS_SQL,
            query=query,
            top_k=top_k,
            status=status,
            min_importance=min_importance,
            namespace=namespace,
            temporal_selection=temporal_selection,
            entry_filter=entry_filter,
        )


def search_fts(
    backend: SQLiteBackend,
    select_columns_sql: str,
    *,
    query: str,
    top_k: int = 25,
    status: MemoryStatus | None = None,
    min_importance: float = 0.0,
    namespace: str | None = None,
    temporal_selection: TemporalSelection | None = None,
    entry_filter: Callable[[MemoryEntry], bool] | None = None,
) -> list[MemoryEntry]:
    """FTS5 full-text search — O(log N) inverted index candidate retrieval.

    Replaces the LIKE '%term%' table scan in :func:`search` for callers that
    have confirmed ``backend._fts_available``.  Results are ranked by FTS5
    BM25 for candidate retrieval; the caller's hybrid pipeline may re-rank.
    Falls back to an empty list when no FTS candidates match.
    """
    if top_k <= 0:
        return []
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
    # The candidate query joins memories to memories_fts, and both now declare a
    # ``namespace`` column, so its copy of the filter must be table-qualified.
    memories_filter_sql, _ = backend._build_filter_clause(
        status=status, namespace=namespace, min_importance=min_importance, column_prefix="memories."
    )
    if temporal_selection is not None or entry_filter is not None:
        from trw_memory.storage._temporal_fetch import execute_temporal_query

        membership = "(id, namespace) IN (SELECT id, namespace FROM memories_fts WHERE memories_fts MATCH ?)"
        where_sql = f"{membership} AND {filter_sql}"
        return execute_temporal_query(
            backend,
            backend._fetch_query(
                where_sql=where_sql, params=[fts_query, *filter_params], order_by="importance DESC, updated_at DESC"
            ),
            temporal_selection,
            limit=top_k,
            entry_filter=entry_filter,
        )
    over_fetch = min(top_k * 4, 500)
    try:
        with backend._lock:
            candidate_sql = f"""
                SELECT memories_fts.id FROM memories_fts
                JOIN memories ON memories.id = memories_fts.id
                    AND memories.namespace = memories_fts.namespace
                WHERE memories_fts MATCH ? AND {memories_filter_sql}
                ORDER BY rank LIMIT ?
            """  # noqa: S608 - filter_sql is built only from fixed internal clauses.
            fts_rows = backend._conn.execute(candidate_sql, (fts_query, *filter_params, over_fetch)).fetchall()
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
