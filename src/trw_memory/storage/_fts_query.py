"""SQLite FTS acquisition: public bound facade and legacy/policy query paths."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.retrieval.lexical import MAX_QUERY_CHARS, MAX_QUERY_TERMS
from trw_memory.storage._query_ops import _append_exact_tag_filters, _execute_resilient

if TYPE_CHECKING:
    from trw_memory.retrieval.temporal_selection import TemporalSelection
    from trw_memory.storage.sqlite_backend import SQLiteBackend


# A chunk is searchable when it holds at least one character the ``unicode61`` tokenizer
# (``_schema.py``) indexes; a punctuation-only chunk would be an empty phrase.
_TERM_RE = re.compile(r"[^\W_]")


def _quote(chunk: str) -> str:
    return '"' + chunk.replace('"', '""') + '"'


def build_match_query(query: str) -> str | None:
    """The FTS5 MATCH expression for ``query``, or ``None`` when it has no searchable term.

    PRD-FIX-148: a natural-language query is an OR of its whitespace-separated chunks,
    each quoted, so a row sharing any chunk is a candidate and BM25 ranks by overlap.
    FTS5 tokenizes inside the quotes, so an identifier or path (``session_store``,
    ``src/auth/session.py``) stays one ordered unit rather than an OR of its common
    parts. Quoting keeps ``AND``/``OR``/``NOT``/``NEAR``, ``-``, ``:`` and ``*`` literal. The previous
    form quoted the WHOLE query, which FTS5 reads as one phrase: a multi-word question
    matched only a row containing that exact word sequence, i.e. almost never.

    A query the caller wraps in double quotes keeps phrase semantics (FR03).
    """
    if len(query) >= 2 and query[0] == '"' and query[-1] == '"':
        inner = query[1:-1].strip()
        return _quote(inner) if _TERM_RE.search(inner) else None
    chunks = [chunk for chunk in dict.fromkeys(query.lower().split()) if _TERM_RE.search(chunk)]
    return " OR ".join(_quote(chunk) for chunk in chunks[:MAX_QUERY_TERMS]) or None


def search_fts_method(
    self: SQLiteBackend,
    query: str,
    *,
    top_k: int = 25,
    status: MemoryStatus | None = None,
    min_importance: float = 0.0,
    namespace: str | None = None,
    tags: list[str] | None = None,
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
            tags=tags,
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
    tags: list[str] | None = None,
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
    # The empty guard, the length cap and the term cap bound the MATCH (DoS);
    # build_match_query keeps every operator literal.
    fts_query = build_match_query(query.strip()[:MAX_QUERY_CHARS])
    if fts_query is None:
        return []
    filter_sql, filter_params = backend._build_filter_clause(
        status=status, namespace=namespace, min_importance=min_importance
    )
    # The candidate query joins memories to memories_fts, and both now declare a
    # ``namespace`` column, so its copy of the filter must be table-qualified.
    memories_filter_sql, _ = backend._build_filter_clause(
        status=status, namespace=namespace, min_importance=min_importance, column_prefix="memories."
    )
    # Required tags filter the MATCH itself, before its LIMIT, as list_entries does,
    # so a wrong-tag row can neither be returned nor use up a page.
    memories_filter_sql = _append_exact_tag_filters(memories_filter_sql, [], tags, column_prefix="memories.")
    filter_sql = _append_exact_tag_filters(filter_sql, filter_params, tags)
    # Candidates are taken in BM25 rank order, a page at a time, for BOTH paths, and
    # returned in that order. Re-sorting by importance/recency (as this function once
    # did) was harmless while the phrase MATCH found almost nothing, but over an OR of
    # terms it returns the most important of many loose matches instead of the most
    # relevant (EngMem-Synth 5,000 rows: 75% vs 100% gold in the top 200; PRD-FIX-148).
    # Paging keeps the filtered contract: a match ranked past the first page is still
    # reached when the caller's filter rejects everything above it. Each page re-runs
    # the MATCH, so pages double: a filter that rejects all N matches costs about
    # log2(N / 500) evaluations rather than N / 500.
    page = min(top_k * 4, 500)
    candidate_sql = f"""
        SELECT memories_fts.id, memories_fts.namespace FROM memories_fts
        JOIN memories ON memories.id = memories_fts.id
            AND memories.namespace = memories_fts.namespace
        WHERE memories_fts MATCH ? AND {memories_filter_sql}
        ORDER BY rank, memories_fts.rowid LIMIT ? OFFSET ?
    """  # noqa: S608 - filter_sql is built only from fixed internal clauses.
    results: list[MemoryEntry] = []
    seen: set[tuple[str, str]] = set()
    offset = 0
    while len(results) < top_k:
        try:
            with backend._lock:
                rows = backend._conn.execute(candidate_sql, (fts_query, *filter_params, page, offset)).fetchall()
        except (sqlite3.Error, ValueError, KeyError) as exc:
            raise StorageError(f"Failed FTS5 search: {exc}", path=str(backend._db_path)) from exc
        if not rows:
            break
        keys = [key for key in dict.fromkeys((str(row[0]), str(row[1])) for row in rows) if key not in seen]
        seen.update(keys)
        rank = {key: position for position, key in enumerate(keys)}
        batch = _fetch_keys(
            backend, select_columns_sql, keys, filter_sql, filter_params, temporal_selection, entry_filter
        )
        results.extend(sorted(batch, key=lambda entry: rank.get((entry.id, entry.namespace), len(keys))))
        if len(rows) < page:
            break
        offset += page
        page *= 2
    return results[:top_k]


def _fetch_keys(
    backend: SQLiteBackend,
    select_columns_sql: str,
    keys: list[tuple[str, str]],
    filter_sql: str,
    filter_params: list[object],
    temporal_selection: TemporalSelection | None,
    entry_filter: Callable[[MemoryEntry], bool] | None,
) -> list[MemoryEntry]:
    """The entries for one page of (id, namespace) keys, through the temporal path when asked.

    Keyed on both columns: the table's key is (namespace, id), so an id alone would pull
    a same-id row from another namespace when the caller passed no namespace filter.
    """
    if not keys:
        return []
    id_filter = f"(id, namespace) IN (VALUES {', '.join(['(?, ?)'] * len(keys))})"
    key_params = [value for key in keys for value in key]
    where_sql = id_filter if filter_sql == "1" else f"{id_filter} AND {filter_sql}"
    # No LIMIT: the id filter bounds it, and temporal selection refuses a pre-eligibility limit.
    fetch_query = backend._fetch_query(
        where_sql=where_sql, params=[*key_params, *filter_params], order_by="importance DESC, updated_at DESC"
    )
    if temporal_selection is not None or entry_filter is not None:
        from trw_memory.storage._temporal_fetch import execute_temporal_query

        return execute_temporal_query(
            backend, fetch_query, temporal_selection, limit=len(keys), entry_filter=entry_filter
        )
    sql = (
        f"SELECT {select_columns_sql} FROM memories "  # noqa: S608
        f"WHERE {where_sql} ORDER BY importance DESC, updated_at DESC LIMIT ?"
    )
    try:
        with backend._lock:
            return _execute_resilient(backend, sql, [*key_params, *filter_params, len(keys)], fetch_query=fetch_query)
    except (sqlite3.Error, ValueError, KeyError) as exc:
        raise StorageError(f"Failed FTS5 search: {exc}", path=str(backend._db_path)) from exc
