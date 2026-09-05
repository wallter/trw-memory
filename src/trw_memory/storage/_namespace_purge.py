"""Atomic whole-namespace purge for the SQLite backend.

Belongs to the ``sqlite_backend.py`` facade; the facade delegates
``delete_by_namespace`` here. Extracted by PRD-CORE-245 when the namespace
predicates it added to the sidecar cleanups pushed the facade past its
grandfathered effective-LOC ceiling — the facade is a delegator, and this was
the one method on it still carrying a body.

The invariant it protects (S8): the entry-row DELETE, the ``wiki_refs`` cleanup,
the graph-edge purge, the ``memory_tags`` purge and the vector purge
are ONE transaction, so a crash can never leave a sidecar row pointing at an
entry that is gone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)

__all__ = ["delete_namespace"]


def delete_namespace(backend: SQLiteBackend, namespace: str) -> int:
    """Delete every entry in *namespace*, with all its sidecars, atomically."""
    from trw_memory.storage._crud_ops import (
        purge_edges_for,
        purge_orphan_edges,
        purge_tag_postings_for,
    )
    from trw_memory.storage._query_ops import delete_by_namespace as _delete_rows
    from trw_memory.storage._vector_ops import purge_vectors_for

    with backend.transaction():
        # Snapshot the namespace's entry IDs INSIDE the BEGIN IMMEDIATE txn so the
        # sidecar rows are cleaned up for exactly the rows the DELETE removes.
        # Reading the IDs before the txn left a TOCTOU: a concurrent INSERT between
        # the SELECT and BEGIN got deleted from ``memories`` (DELETE WHERE
        # namespace) but its vec rows were missed (stale id list) -> orphan vec
        # hits. The write lock here blocks concurrent writers, so the snapshot
        # matches the delete.
        with backend._lock:
            rows = backend._conn.execute(
                "SELECT id FROM memories WHERE namespace = ?",
                (namespace,),
            ).fetchall()
        entry_ids = [str(row[0]) for row in rows]
        if not entry_ids:
            return 0
        deleted = _delete_rows(backend, namespace)
        with backend._lock:
            backend._conn.execute("DELETE FROM wiki_refs WHERE namespace = ?", (namespace,))
            # SQLite enforces no FK cascade here, so the bulk delete cleans the
            # graph explicitly: first the namespace-qualified purge the per-row
            # delete() also uses, then the orphan sweep that catches any edge left
            # naming a row this delete removed.
            purge_edges_for(backend, entry_ids, namespace)
            purge_orphan_edges(backend)
            purge_tag_postings_for(backend, namespace, entry_ids)
            if backend._vec_available:
                purge_vectors_for(backend._conn, namespace, entry_ids)
    return deleted
