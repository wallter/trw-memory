"""Transaction context implementation for :mod:`sqlite_backend`.

The SQLite backend keeps the public ``SQLiteBackend.transaction`` interface on
the facade, but the concurrency-sensitive implementation lives here so the
transaction seam has one focused module and the storage facade stays smaller.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)


@contextmanager
def transaction(backend: SQLiteBackend) -> Iterator[SQLiteBackend]:
    """Batch writes into one BEGIN IMMEDIATE / COMMIT.

    Re-entrant by depth — only the outermost ``transaction()`` issues
    BEGIN/COMMIT; inner exceptions propagate; outermost issues ROLLBACK.

    Concurrency: the re-entrant ``_txn_serializer`` is held for the whole body
    so two threads cannot both open a BEGIN IMMEDIATE on the single shared
    connection. Same-thread nested ``transaction()`` calls re-enter the RLock
    and are classified inner by depth, so they never re-issue BEGIN.
    """
    backend._txn_serializer.acquire()
    try:
        is_outer = backend._skip_commit_depth == 0
        if is_outer:
            with backend._lock:
                backend._conn.execute("BEGIN IMMEDIATE")
                backend._skip_commit_depth += 1
        else:
            with backend._lock:
                backend._skip_commit_depth += 1
        try:
            yield backend
            if is_outer:
                with backend._lock:
                    backend._conn.commit()
        except BaseException:
            if is_outer:
                try:
                    with backend._lock:
                        backend._conn.rollback()
                except sqlite3.Error:
                    logger.exception("transaction_rollback_failed", db=str(backend._db_path))
            raise
        finally:
            with backend._lock:
                backend._skip_commit_depth -= 1
    finally:
        backend._txn_serializer.release()
