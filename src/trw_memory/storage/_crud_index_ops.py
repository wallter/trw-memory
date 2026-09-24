"""Tag-posting and graph-edge sidecar-index maintenance for ``_crud_ops.py``.

Split out of ``_crud_ops.py`` (PRD-CORE-291 slice 3) when the parent module
crossed the effective-LOC ceiling. These four helpers are the ``memory_tags``
and ``memory_graph_edges`` sidecar-index bookkeeping shared by
``_crud_ops.store``/``update``/``delete`` and by ``_namespace_purge.py``'s
bulk ``delete_by_namespace`` — one source of truth for keeping those two
inverted indexes in sync with the ``memories`` table, whichever caller is
mutating it.

- ``_replace_tag_postings`` — re-point the ``memory_tags`` inverted index at
  one entry's current tags.
- ``purge_tag_postings_for`` — chunked bulk delete of ``memory_tags`` rows.
- ``purge_edges_for`` — chunked bulk delete of ``memory_graph_edges`` rows
  referencing any of a set of entry ids.
- ``purge_orphan_edges`` — delete every edge whose source or target row no
  longer exists.

``_crud_ops.py`` re-exports all four (imported there) so every existing
``from trw_memory.storage._crud_ops import ...`` call site keeps working.

CONTRACT (all four): the caller MUST already hold ``backend._lock`` and own
the commit — none of these acquire the lock or commit themselves, so their
writes batch into the caller's outermost ``COMMIT``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from trw_memory.storage._sql_utils import iter_bind_chunks
from trw_memory.storage.interface import GraphEdge

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend


def _replace_tag_postings(backend: SQLiteBackend, namespace: str, entry_id: str, tags: Sequence[str]) -> None:
    """Re-point the ``memory_tags`` inverted index at *entry_id*'s current tags.

    PRD-CORE-245 FR07: this index is what the bounded tag derivation queries in
    place of the 98,288 materialised ``tag_cooccurrence`` edges the schema-5
    migration deleted. It is maintained beside the FTS row on every write and
    pruned beside the edges on every delete, so it can never drift from the
    ``tags`` column it mirrors.
    """
    backend._conn.execute(
        "DELETE FROM memory_tags WHERE namespace = ? AND entry_id = ?",
        (namespace, entry_id),
    )
    rows = [(namespace, tag, entry_id) for tag in dict.fromkeys(tags) if str(tag).strip()]
    if rows:
        backend._conn.executemany("INSERT OR IGNORE INTO memory_tags(namespace, tag, entry_id) VALUES(?, ?, ?)", rows)


def purge_tag_postings_for(backend: SQLiteBackend, namespace: str, entry_ids: Sequence[str]) -> None:
    """Drop every ``memory_tags`` posting for *entry_ids* within *namespace*.

    CONTRACT: mirrors :func:`purge_edges_for` — the caller already holds
    ``backend._lock`` and owns the commit.
    """
    if not entry_ids:
        return
    for chunk in iter_bind_chunks(list(entry_ids), reserved_bindings=1):
        placeholders = ",".join("?" for _ in chunk)
        backend._conn.execute(
            f"DELETE FROM memory_tags WHERE namespace = ? AND entry_id IN ({placeholders})",  # noqa: S608 — placeholders is ? repeated; ids are parameterized
            (namespace, *chunk),
        )


# memory_graph_edges binds each purged id twice (source + target), so chunking
# keeps each statement below SQLite's conservative bind ceiling. All deletes
# run inside the caller's held lock and transaction, so the net effect remains
# atomic.


def purge_edges_for(backend: SQLiteBackend, entry_ids: Sequence[str], namespace: str) -> None:
    """Delete knowledge-graph edges referencing any of *entry_ids* in *namespace*.

    ``memory_graph_edges`` declares no FK cascade (``PRAGMA foreign_keys`` is
    off process-wide), so orphan edges must be pruned explicitly whenever their
    endpoint rows are deleted. A single ``DELETE`` with ``OR`` covers both sides
    of a directed edge. Shared by the per-row :func:`_crud_ops.delete` and the
    bulk ``delete_by_namespace`` paths — one source of truth for edge cleanup.

    CONTRACT: the caller MUST already hold ``backend._lock`` and own the commit.
    This helper neither acquires the lock nor commits, so the edge purge batches
    into the caller's outermost ``COMMIT`` and preserves transaction atomicity
    (test_storage_transaction_atomicity.py).
    """
    if not entry_ids:
        return
    ids = list(entry_ids)
    for chunk in iter_bind_chunks(ids, bindings_per_item=2, reserved_bindings=1):
        placeholders = ",".join("?" for _ in chunk)
        backend._conn.execute(
            f"DELETE FROM memory_graph_edges WHERE namespace = ? AND (source_id IN ({placeholders}) "  # noqa: S608 — placeholders is ? repeated; ids are parameterized values
            f"OR target_id IN ({placeholders}))",
            (namespace, *chunk, *chunk),
        )


def purge_orphan_edges(backend: SQLiteBackend) -> None:
    """Delete every edge whose source or target row no longer exists.

    :func:`purge_edges_for` is namespace-qualified (PRD-CORE-245 FR02) because a
    per-row delete knows exactly which namespace it is addressing. A bulk
    ``delete_by_namespace`` needs the complementary guarantee: an edge that
    names a row which is now gone is dangling regardless of which namespace the
    edge itself is filed under, and a BFS that follows it lands on a ghost node.
    This is the edge-table twin of the ``memories_fts`` ghost-row anti-join.

    CONTRACT: caller holds ``backend._lock`` and owns the commit.
    """
    backend._conn.execute(
        "DELETE FROM memory_graph_edges WHERE NOT EXISTS ("
        "  SELECT 1 FROM memories m WHERE m.namespace = memory_graph_edges.namespace"
        "    AND m.id = memory_graph_edges.source_id"
        ") OR NOT EXISTS ("
        "  SELECT 1 FROM memories m WHERE m.namespace = memory_graph_edges.namespace"
        "    AND m.id = memory_graph_edges.target_id"
        ")"
    )


def read_edges(backend: SQLiteBackend, namespace: str) -> list[GraphEdge]:
    """Every ``memory_graph_edges`` row of *namespace*, oldest first. CONTRACT: caller holds ``backend._lock``."""
    rows = backend._conn.execute(
        "SELECT source_id, target_id, edge_type, weight, created_at, COALESCE(edge_metadata, '{}') "
        "FROM memory_graph_edges WHERE namespace = ? ORDER BY id",
        (namespace,),
    ).fetchall()
    return [GraphEdge(*row) for row in rows]


def insert_edges(backend: SQLiteBackend, namespace: str, edges: Sequence[GraphEdge]) -> None:
    """File *edges* under *namespace*; an edge already there is kept. CONTRACT: as :func:`purge_edges_for`."""
    backend._conn.executemany(
        "INSERT OR IGNORE INTO memory_graph_edges "
        "(namespace, source_id, target_id, edge_type, weight, created_at, edge_metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(namespace, *edge) for edge in edges],
    )
