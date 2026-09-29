"""Batched id reads for :class:`SQLiteBackend` (PRD-CORE-318 FR02).

Split from ``_crud_ops`` to keep that module under the effective-LOC ratchet.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import TYPE_CHECKING

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._query_ops import _execute_resilient
from trw_memory.storage._sql_utils import iter_bind_chunks

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend


def get_many(
    backend: SQLiteBackend,
    select_columns_sql: str,
    entry_ids: Sequence[str],
    namespace: str,
) -> dict[str, MemoryEntry]:
    """``_crud_ops.get`` for many ids: one ``IN`` read per bind chunk (PRD-CORE-318 FR02).

    Rows are materialised through the resilient multi-row path every other batched
    read uses, so a row that cannot decode is quarantined (counted, not returned)
    rather than failing the batch.
    """
    found: dict[str, MemoryEntry] = {}
    try:
        with backend._lock:
            for chunk in iter_bind_chunks(list(dict.fromkeys(entry_ids)), reserved_bindings=1):
                where = f"namespace = ? AND id IN ({','.join('?' for _ in chunk)})"
                params = (namespace, *chunk)
                sql = f"SELECT {select_columns_sql} FROM memories WHERE {where}"  # noqa: S608
                fetch_query = backend._fetch_query(where_sql=where, params=params, order_by="id")
                found.update((e.id, e) for e in _execute_resilient(backend, sql, params, fetch_query=fetch_query))
    except (sqlite3.Error, ValueError, KeyError) as exc:
        raise StorageError(f"Failed to get {len(entry_ids)} entries: {exc}", path=str(backend._db_path)) from exc
    return found
