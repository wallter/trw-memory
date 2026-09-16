"""Assertion-bearing entry queries — the PRD-CORE-086 FR07 read path.

Belongs to the ``sqlite_backend.py`` facade; ``SQLiteBackend.entries_with_assertions``
(and its ``count_with_assertions`` alias) delegate here. Split out of
``_query_ops.py``, which measured 382 effective LOC against the 350 gate.

This pair travels together and nothing else calls either: ``entries_with_assertions``
is the only caller of ``_refill_verification_entries``, and the split page-refill
traversal exists solely for the maintenance mode that ``include_anchors``/``after``
select. ``_query_ops`` keeps the generic keyword/list/namespace surface.

``_execute_resilient`` stays in ``_query_ops`` — ``search`` and ``list_entries``
share it — and is imported from there rather than copied.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage._query_ops import _execute_resilient

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)

__all__ = ["entries_with_assertions"]


def _refill_verification_entries(
    backend: SQLiteBackend,
    select_columns_sql: str,
    where_sql: str,
    params: tuple[object, ...],
    limit: int,
) -> list[MemoryEntry]:
    """Fill one maintenance batch, advancing by raw keys rather than decoded rows.

    Malformed payloads can disappear during resilient materialization. Only raw
    key exhaustion ends this scan; otherwise a corrupt first page would hide all
    later claims. Each key/payload query and the returned list remain bounded.
    Caller holds the backend lock; concurrent external mutation is not a snapshot.
    """
    result: list[MemoryEntry] = []
    after: tuple[str, str] | None = None
    while len(result) < limit:
        page_where = where_sql
        page_params = params
        if after is not None:
            page_where += " AND (namespace, id) > (?, ?)"
            page_params = (*page_params, *after)
        remaining = limit - len(result)
        keys = backend._conn.execute(
            f"SELECT namespace, id FROM memories WHERE {page_where} "  # noqa: S608
            "ORDER BY namespace, id LIMIT ?",
            (*page_params, remaining),
        ).fetchall()
        if not keys:
            break
        after = (str(keys[-1][0]), str(keys[-1][1]))
        page_where += " AND (namespace, id) <= (?, ?)"
        page_params = (*page_params, *after)
        query = backend._fetch_query(
            where_sql=page_where, params=page_params, order_by="namespace, id", limit=remaining
        )
        result.extend(
            _execute_resilient(
                backend,
                f"SELECT {select_columns_sql} FROM memories WHERE {page_where} "  # noqa: S608
                "ORDER BY namespace, id LIMIT ?",
                (*page_params, remaining),
                fetch_query=query,
            )
        )
    return result


def entries_with_assertions(
    backend: SQLiteBackend,
    select_columns_sql: str,
    *,
    status: MemoryStatus | None = MemoryStatus.ACTIVE,
    namespace: str | None = None,
    limit: int = 500,
    include_anchors: bool = False,
    after: tuple[str, str] | None = None,
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
    # Maintenance opts into stable namespace/id traversal; legacy summaries
    # retain assertions-only eligibility and updated-at ordering.
    if limit <= 0:
        return []
    where_sql = "assertions IS NOT NULL AND assertions != '[]'"
    if include_anchors:
        where_sql = f"({where_sql} OR (anchors IS NOT NULL AND anchors != '[]'))"
    params: tuple[object, ...] = ()
    if status is not None:
        where_sql = f"{where_sql} AND status = ?"
        params = (*params, status.value)
    if namespace is not None:
        where_sql = f"{where_sql} AND namespace = ?"
        params = (*params, namespace)
    if after is not None:
        where_sql += " AND (namespace, id) > (?, ?)"
        params = (*params, *after)
    order_by = "namespace, id" if include_anchors or after is not None else "updated_at DESC"
    sql = (
        f"SELECT {select_columns_sql} FROM memories WHERE {where_sql} "  # noqa: S608
        f"ORDER BY {order_by} LIMIT ?"
    )
    exec_params: tuple[object, ...] = (*params, limit)
    fetch_query = backend._fetch_query(where_sql=where_sql, params=params, order_by=order_by, limit=limit)
    try:
        with backend._lock:
            if include_anchors or after is not None:
                return _refill_verification_entries(backend, select_columns_sql, where_sql, params, limit)
            return _execute_resilient(backend, sql, exec_params, fetch_query=fetch_query)
    except sqlite3.Error as exc:
        logger.debug("entries_with_assertions_query_failed", exc_info=True)
        raise StorageError(
            f"Failed to query entries with assertions: {exc}",
            path=str(backend._db_path),
        ) from exc
