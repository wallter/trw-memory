"""One namespace's health, as the store itself measures it (PRD-CORE-280).

``memory_status`` answers these for a namespace, so trw-mcp's pipeline-health
probes and its store inventory read the daemon's own store instead of opening a
checkout's ``memory.db`` -- which, since memory moved to the daemon, holds nothing.

The graph half: PRD-CORE-245 FR07 split the graph in two. Materialised
``memory_graph_edges`` rows still carry the semantic edge types, but tag
co-occurrence is DERIVED at query time from the ``memory_tags`` index by
:func:`trw_memory.retrieval.tag_derivation.derive_tag_neighbours`, so a count
of edges alone reads zero for a corpus whose graph is entirely healthy.
"""

from __future__ import annotations

import sqlite3

from trw_memory.models.config import MemoryConfig
from trw_memory.retrieval.tag_derivation import derive_tag_neighbours
from trw_memory.storage._probe import pinned_canary_ids
from trw_memory.storage.sqlite_backend import SQLiteBackend

#: Rows team sync pulled from another project: recalled, but not authored here.
SYNCED_SOURCE = "team_sync"


def namespace_health(backend: SQLiteBackend, namespace: str, config: MemoryConfig) -> dict[str, object]:
    """``{entries, synced, edges, has_relations, embedded, max_recall_count, types}`` for *namespace*.

    ``types`` counts rows per type exactly (PRD-CORE-334 FR04).

    Canary rows count toward none of them. ``embedded`` is ``None`` when the store
    keeps no vectors, which a caller reports as not measured, never as zero; a read
    error propagates rather than reading as an empty store.
    """
    # Canaries are the store's own health-check rows: instrumentation, not knowledge. They are the rows the
    # probe calls canaries (pinned id AND content, ``classify_canary``), NOT rows carrying a ``system_canary``
    # metadata flag: a caller-writable flag would let a row hide itself from this count (UF-MEM-15).
    with backend._fresh_connection(), backend._lock:
        conn = backend._conn
        canaries = pinned_canary_ids(conn, namespace)
        skip = f"AND m.id NOT IN ({','.join('?' * len(canaries))})" if canaries else ""
        # `skip` is only "?" placeholders for the canary ids: no value is interpolated into the SQL.
        totals_sql = (
            "SELECT COUNT(*), COALESCE(SUM(m.source = ?), 0), COALESCE(MAX(m.recall_count), 0) FROM memories m "  # noqa: S608
            "WHERE m.namespace = ? " + skip
        )
        types_sql = (
            "SELECT COALESCE(NULLIF(m.type, ''), 'pattern') AS kind, COUNT(*) FROM memories m "  # noqa: S608
            "WHERE m.namespace = ? " + skip + " GROUP BY kind"
        )
        entries, synced, max_recall = conn.execute(totals_sql, (SYNCED_SOURCE, namespace, *canaries)).fetchone()
        types = conn.execute(types_sql, (namespace, *canaries)).fetchall()
        edges = conn.execute("SELECT COUNT(*) FROM memory_graph_edges WHERE namespace = ?", (namespace,)).fetchone()[0]
        has_relations = edges > 0 or _derives_a_relation(conn, namespace, config)
        # Counted in SQL, and a read error propagates: an unreadable index is not zero vectors.
        embedded = (
            conn.execute(
                "SELECT COUNT(*) FROM vec_index v JOIN memories m ON m.namespace = v.namespace AND m.id = v.entry_id "  # noqa: S608
                "WHERE v.namespace = ? " + skip,
                (namespace, *canaries),
            ).fetchone()[0]
            if backend.supports_vectors()
            else None
        )
    return {
        "entries": int(entries),
        "synced": int(synced),
        "edges": int(edges),
        "has_relations": has_relations,
        "embedded": None if embedded is None else int(embedded),
        "max_recall_count": int(max_recall),
        "types": {str(kind): int(count) for kind, count in types},
    }


#: How many of the most recently updated entries the derived half samples: one root
#: made a single isolated newest entry report the whole graph dead (PRD-FIX-141-FR02).
RELATION_SAMPLE_ROOTS = 3


def _derives_a_relation(conn: sqlite3.Connection, namespace: str, config: MemoryConfig) -> bool:
    """Whether one of the namespace's newest entries derives a tag neighbour (bounded, not a census)."""
    roots = conn.execute(
        "SELECT id FROM memories WHERE namespace = ? ORDER BY updated_at DESC, id DESC LIMIT ?",
        (namespace, RELATION_SAMPLE_ROOTS),
    ).fetchall()
    return any(derive_tag_neighbours(conn, str(row[0]), namespace=namespace, config=config) for row in roots)
