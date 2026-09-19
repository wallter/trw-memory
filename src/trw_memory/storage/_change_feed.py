"""Namespace change tokens and change feeds for ``SQLiteBackend``.

A caller that keeps a derived view of a namespace (the runtime anomaly
reference window, ``security/_anomaly_reference.py``) needs to know, cheaply,
whether the namespace changed since it last looked and, if so, which rows. A
row count or a re-read of the recent rows is O(namespace) or O(window) per
call. The token here costs two index seeks:

- ``insert_seq``: ``MAX(rowid)`` of the namespace. ``store``/``store_many``
  are ``INSERT OR REPLACE``, so every insert AND every upsert gets a new, larger
  rowid, whatever ``updated_at`` the caller supplied.
- ``top_updated_at``: ``MAX(updated_at)`` of the namespace. ``update`` keeps the
  rowid but stamps ``updated_at`` (a status change, a metadata edit, ...).
- ``delete_epoch``: deletes run in this process. A delete leaves both maxima
  alone unless it removed the newest row, so this process counts its own.

What the token cannot see: a DELETE run by another process, unless it lowers
one of the two maxima and no later write raises it again; an ``update`` that
passes an explicit ``updated_at`` no newer than the current maximum, when it is
the only write in between; and an ``update`` limited to the bookkeeping fields
that deliberately do not stamp ``updated_at`` (access and session counters).
Consumers bound the first two by re-reading on a timer; the third touches no
field a consumer of this feed reads.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import TYPE_CHECKING

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._query_ops import _execute_resilient
from trw_memory.storage.interface import NamespaceChangeToken

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

_DELETE_EPOCHS: dict[str, int] = {}
_DELETE_EPOCHS_LOCK = threading.Lock()

_NAMESPACE_MAXIMA_SQL = (
    "SELECT (SELECT MAX(rowid) FROM memories WHERE namespace = ?), "
    "(SELECT MAX(updated_at) FROM memories WHERE namespace = ?)"
)
# Two index-bounded arms (idx_memories_namespace carries the rowid,
# idx_memories_ns_updated the timestamp); a single ``rowid > ? OR updated_at >= ?``
# predicate scans the whole namespace instead. ``>=`` re-reads the previous top
# row too, so an update stamped with the same microsecond is not missed.
_CHANGED_WHERE_SQL = (
    "rowid IN (SELECT rowid FROM (SELECT rowid FROM memories WHERE namespace = ? AND rowid > ? LIMIT ?) "
    "UNION SELECT rowid FROM (SELECT rowid FROM memories WHERE namespace = ? AND updated_at >= ? LIMIT ?))"
)


def store_identity(backend: SQLiteBackend) -> str:
    """Return the key under which instances of one database file share state."""
    path = str(backend._db_path)
    if ":memory:" in path or path.startswith("file:"):
        return f"{path}#{id(backend)}"  # an in-memory database is private to its connection
    return path


def note_delete(backend: SQLiteBackend) -> None:
    """Record that this process deleted rows from *backend*'s database."""
    identity = store_identity(backend)
    with _DELETE_EPOCHS_LOCK:
        _DELETE_EPOCHS[identity] = _DELETE_EPOCHS.get(identity, 0) + 1


def change_token(backend: SQLiteBackend, namespace: str) -> NamespaceChangeToken:
    """Return the namespace's current :class:`NamespaceChangeToken` (two index seeks)."""
    identity = store_identity(backend)
    try:
        with backend._fresh_connection(), backend._lock:
            row = backend._conn.execute(_NAMESPACE_MAXIMA_SQL, (namespace, namespace)).fetchone()
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to read the namespace change token: {exc}", path=str(backend._db_path)) from exc
    with _DELETE_EPOCHS_LOCK:
        epoch = _DELETE_EPOCHS.get(identity, 0)
    return NamespaceChangeToken(
        store=identity,
        insert_seq=int(row[0] or 0) if row else 0,
        top_updated_at=str(row[1] or "") if row else "",
        delete_epoch=epoch,
    )


def entries_changed_since(
    backend: SQLiteBackend, namespace: str, token: NamespaceChangeToken, *, limit: int
) -> list[MemoryEntry] | None:
    """Rows of *namespace*, any status, inserted or stamped since *token*.

    Newest first (``updated_at DESC, id DESC``). ``None`` when more than *limit*
    rows changed, so the caller re-reads instead of trusting a truncated feed.
    """
    params = (namespace, token.insert_seq, limit + 1, namespace, token.top_updated_at, limit + 1)
    fetch_query = backend._fetch_query(where_sql=_CHANGED_WHERE_SQL, params=params, order_by="updated_at DESC, id DESC")
    sql, _ = fetch_query.build()
    try:
        with backend._fresh_connection(), backend._lock:
            rows = _execute_resilient(backend, sql, params, fetch_query=fetch_query)
    except (sqlite3.Error, ValueError, KeyError) as exc:
        raise StorageError(f"Failed to read changed entries: {exc}", path=str(backend._db_path)) from exc
    return None if len(rows) > limit else rows
