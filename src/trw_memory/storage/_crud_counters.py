"""Bulk ``session_count`` bookkeeping, split off ``_crud_ops.py``.

Belongs to the ``sqlite_backend.py`` facade (via ``_crud_ops.py``'s
re-export) — moved out of ``_crud_ops.py`` (PRD-CORE-291 slice 3) when that
module crossed the effective-LOC ceiling. ``increment_session_counts`` bumps
one namespace's rows under the ``_MAX_COUNTER`` cap and defers its commit
inside ``transaction()``; its recall-time sibling is
``_crud_ops.increment_recall_access``.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

from trw_memory.exceptions import StorageError
from trw_memory.storage._shared import _MAX_COUNTER

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend


def increment_session_counts(
    backend: SQLiteBackend,
    entry_ids: list[str],
    *,
    namespace: str,
    updated_at: datetime | None = None,
) -> int:
    """Increment session_count for *namespace*'s rows among *entry_ids* in one transaction.

    PRD-CORE-245 FR03: a bare id does not identify a row, so the namespace is
    required; an id's twin in another namespace is left alone.
    """
    if not entry_ids:
        return 0

    # PRD-CORE-278 FR06: session bookkeeping no longer stamps ``updated_at``.
    # ``updated_at`` is kept as the parameter name because callers pass it, and
    # because a future content-bearing use of this path would want it.
    _ = updated_at
    values = [(namespace, entry_id) for entry_id in entry_ids]

    try:
        sql = f"""
            UPDATE memories
            SET session_count = MIN(COALESCE(session_count, 0) + 1, {_MAX_COUNTER}),
                sync_seq = COALESCE(sync_seq, 0) + 1,
                last_synced_at = NULL
            WHERE namespace = ? AND id = ?
        """  # noqa: S608 — _MAX_COUNTER is a module-level int constant, not user input.
        with backend._lock:
            before = backend._conn.total_changes
            backend._conn.executemany(sql, values)
            # Suppress the commit inside a ``transaction()`` block so this
            # batches into the caller's outermost COMMIT instead of prematurely
            # committing their open transaction (matches store()/update()).
            if backend._skip_commit_depth == 0:
                backend._conn.commit()
            return int(backend._conn.total_changes - before)
    except sqlite3.Error as exc:
        if backend._skip_commit_depth == 0:
            with backend._lock, contextlib.suppress(sqlite3.Error):
                backend._conn.rollback()
        raise StorageError(
            f"Failed to increment session counts: {exc}",
            path=str(backend._db_path),
        ) from exc
