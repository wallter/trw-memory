"""Tag-posting and graph-edge sidecar-index maintenance for ``_crud_ops.py``.

Split out of ``_crud_ops.py`` (PRD-CORE-291 slice 3) when the parent module
crossed the effective-LOC ceiling. These four helpers are the ``memory_tags``
and ``memory_graph_edges`` sidecar-index bookkeeping shared by
``_crud_ops.store``/``update``/``delete`` and by ``_namespace_purge.py``'s
bulk ``delete_by_namespace`` — one source of truth for keeping those two
inverted indexes in sync with the ``memories`` table, whichever caller is
mutating it.

- ``_fts_row``/``_fts_delete_row``/``_fts_insert_row`` — the PRD-CORE-330
  rowid-keyed ``memories_fts`` row helpers (moved here, unchanged, by
  PRD-CORE-332 S1 to give ``_crud_ops.py`` effective-LOC headroom).
- ``_replace_tag_postings`` — re-point the ``memory_tags`` inverted index at
  one entry's current tags.
- ``purge_tag_postings_for`` — chunked bulk delete of ``memory_tags`` rows.
- ``_replace_postings``/``purge_postings_for`` — the tag AND anchor postings
  (``anchor_postings``, PRD-CORE-332 FR01) every ``memories`` writer maintains.
- ``purge_fts_rows_for`` — chunked bulk delete of ``memories_fts`` rows.
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

import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from trw_memory.storage._anchor_index import replace_anchor_postings
from trw_memory.storage._sql_utils import iter_bind_chunks
from trw_memory.storage.interface import GraphEdge

if TYPE_CHECKING:
    from trw_memory.models.memory import MemoryEntry
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


def _fts_row(entry: MemoryEntry) -> tuple[str, str, str, str, str]:
    """Return the ``memories_fts`` row tuple for *entry* (namespace-qualified)."""
    tags_json = json.dumps(entry.tags) if isinstance(entry.tags, list) else (entry.tags or "[]")
    return (entry.id, entry.namespace, entry.content, entry.detail or "", tags_json)


# --- PRD-CORE-330: indexed FTS delete via the memories_fts_rowid map -------
#
# ``memories_fts`` keys ``id``/``namespace`` as UNINDEXED fts5 columns, so a
# ``WHERE id = ? AND namespace = ?`` delete scans the whole FTS index
# (worker-1's CORE-309 S3 profiling; learning L-7L9A). ``memories_fts_rowid``
# maps ``(namespace, id) -> fts_rowid`` through its own indexed PK, so every
# helper below resolves the FTS row's rowid with a point lookup and deletes
# it with fts5's native ``WHERE rowid = ?`` -- never the old full-scan form.


def _fts_rowid_lookup(backend: SQLiteBackend, namespace: str, entry_id: str) -> int | None:
    """Return the mapped ``memories_fts`` rowid for ``(namespace, entry_id)``, or None."""
    row = backend._conn.execute(
        "SELECT fts_rowid FROM memories_fts_rowid WHERE namespace = ? AND id = ?",
        (namespace, entry_id),
    ).fetchone()
    return int(row[0]) if row is not None else None


def _fts_delete_row(backend: SQLiteBackend, namespace: str, entry_id: str) -> None:
    """Delete the FTS row (if any) for ``(namespace, entry_id)`` by its indexed rowid.

    A no-op when no mapping row exists (nothing was ever indexed for this
    id, or it was already deleted) -- mirrors the old code's DELETE, which
    was also a no-op when no matching FTS row existed.
    """
    if (fts_rowid := _fts_rowid_lookup(backend, namespace, entry_id)) is None:
        return
    backend._conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (fts_rowid,))
    backend._conn.execute("DELETE FROM memories_fts_rowid WHERE namespace = ? AND id = ?", (namespace, entry_id))


def _fts_insert_row(backend: SQLiteBackend, namespace: str, entry_id: str, row: tuple[str, str, str, str, str]) -> None:
    """Insert a new ``memories_fts`` row and record its assigned rowid in the map."""
    cursor = backend._conn.execute(
        "INSERT INTO memories_fts(id, namespace, content, detail, tags) VALUES (?, ?, ?, ?, ?)",
        row,
    )
    backend._conn.execute(
        "INSERT OR REPLACE INTO memories_fts_rowid(namespace, id, fts_rowid) VALUES (?, ?, ?)",
        (namespace, entry_id, cursor.lastrowid),
    )


def _replace_postings(
    backend: SQLiteBackend,
    namespace: str,
    entry_id: str,
    *,
    tags: Sequence[str] | None,
    anchors: Iterable[object] | None,
) -> None:
    """Re-point *entry_id*'s ``memory_tags`` and ``anchor_postings`` rows; ``None`` leaves that index alone.

    The one postings helper every row writer calls (PRD-CORE-332 FR01): ``store``/``store_many`` pass
    both, ``update`` passes only the fields it changed.
    """
    if tags is not None:
        _replace_tag_postings(backend, namespace, entry_id, tags)
    if anchors is not None:
        replace_anchor_postings(backend._conn, namespace, entry_id, anchors)


def _delete_keyed(backend: SQLiteBackend, table: str, id_column: str, namespace: str, ids: Sequence[str]) -> None:
    """``DELETE FROM table WHERE namespace = ? AND id_column IN (...)``, one statement per bind chunk."""
    for chunk in iter_bind_chunks(list(ids), reserved_bindings=1):
        placeholders = ",".join("?" for _ in chunk)
        backend._conn.execute(
            f"DELETE FROM {table} WHERE namespace = ? AND {id_column} IN ({placeholders})",  # noqa: S608 — table/column are module constants; ids are parameterized
            (namespace, *chunk),
        )


def purge_tag_postings_for(backend: SQLiteBackend, namespace: str, entry_ids: Sequence[str]) -> None:
    """Drop every ``memory_tags`` posting for *entry_ids* within *namespace*.

    CONTRACT: mirrors :func:`purge_edges_for` — the caller already holds
    ``backend._lock`` and owns the commit.
    """
    _delete_keyed(backend, "memory_tags", "entry_id", namespace, entry_ids)


def purge_postings_for(backend: SQLiteBackend, namespace: str, entry_ids: Sequence[str]) -> None:
    """Drop every ``memory_tags`` and ``anchor_postings`` row of *entry_ids* in *namespace* (PRD-CORE-332 FR01).

    CONTRACT: as :func:`purge_tag_postings_for`.
    """
    purge_tag_postings_for(backend, namespace, entry_ids)
    _delete_keyed(backend, "anchor_postings", "entry_id", namespace, entry_ids)


def purge_fts_rows_for(backend: SQLiteBackend, namespace: str, entry_ids: Sequence[str]) -> None:
    """Drop the ``memories_fts`` rows of *entry_ids* within *namespace* (PRD-CORE-309 B71-13, PRD-CORE-330).

    ``id`` is not an FTS5 key, so a bare ``WHERE id = ?``/``IN (...)`` against
    ``memories_fts`` itself scans the whole FTS index -- CORE-309 S3's original
    IN-list form still paid that cost once per bind chunk. PRD-CORE-330 added
    ``memories_fts_rowid``, a ``(namespace, id) -> fts_rowid`` map with its own
    indexed PK: resolve the chunk's rowids there first (an indexed lookup, not
    a scan), then delete ``memories_fts`` by rowid, fts5's native indexed
    operation. CONTRACT: as :func:`purge_tag_postings_for`.
    """
    ids = list(entry_ids)
    if not ids:
        return
    for chunk in iter_bind_chunks(ids, reserved_bindings=1):
        placeholders = ",".join("?" for _ in chunk)
        rows = backend._conn.execute(
            f"SELECT fts_rowid FROM memories_fts_rowid WHERE namespace = ? AND id IN ({placeholders})",  # noqa: S608
            (namespace, *chunk),
        ).fetchall()
        if rows:
            # One IN-list DELETE per chunk (not one statement per row): fts5
            # seeks each rowid individually either way, and a single
            # statement keeps this at the same round-trip count CORE-309 S3
            # measured, rather than N per-row round trips.
            rowid_placeholders = ",".join("?" for _ in rows)
            backend._conn.execute(
                f"DELETE FROM memories_fts WHERE rowid IN ({rowid_placeholders})",  # noqa: S608
                [r[0] for r in rows],
            )
        backend._conn.execute(
            f"DELETE FROM memories_fts_rowid WHERE namespace = ? AND id IN ({placeholders})",  # noqa: S608
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
