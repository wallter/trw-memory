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

    Concurrency: the backend's re-entrant connection lock is held for the whole
    body. Same-thread nested calls re-enter it; other threads cannot join or
    observe the open transaction. This synchronous context must not span an
    ``await`` because a thread-reentrant lock cannot distinguish asyncio tasks
    running on the same thread.
    """
    # Hold the connection lock for the entire body. Public operations re-enter
    # it on this thread; other threads cannot join the transaction, skip their
    # own commit, or observe uncommitted rows.
    with backend._lock:
        is_outer = backend._skip_commit_depth == 0
        # Direct compatibility writes can leave SQLite in an implicit
        # transaction without passing through this depth tracker. Adopt that
        # transaction instead of issuing a nested BEGIN.
        if is_outer and not backend._conn.in_transaction:
            backend._conn.execute("BEGIN IMMEDIATE")
        backend._skip_commit_depth += 1
        try:
            yield backend
            if is_outer:
                backend._conn.commit()
        except BaseException:
            if is_outer:
                try:
                    backend._conn.rollback()
                except sqlite3.Error:
                    logger.exception("transaction_rollback_failed", db=str(backend._db_path))
            raise
        finally:
            backend._skip_commit_depth -= 1
