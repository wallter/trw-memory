"""Graph traversal — BFS over materialised edges plus derived tag neighbours.

Belongs to the ``graph.py`` facade; re-exported there for back-compat.

Extracted from ``graph.py`` by PRD-CORE-245 FR07. The facade measured 342
effective LOC against the 350 gate before this change, so the traversal — which
FR07 grows with the materialised/derived split — lands in its own module rather
than pushing the facade over the ratchet.

The split it implements: ``graph_query`` walks only edges that are actually
stored, and serves ``tag_cooccurrence`` (the one type schema 5 stopped
materialising) from the bounded derivation in
:mod:`trw_memory.retrieval.tag_derivation`, at depth 1, only when a caller names
it, and always appended after the materialised results.
"""

from __future__ import annotations

import sqlite3
from collections import deque

import structlog

from trw_memory.models.config import MemoryConfig
from trw_memory.storage._sql_utils import iter_bind_chunks

logger = structlog.get_logger(__name__)

__all__ = ["DERIVED_EDGE_TYPE", "MAX_TRAVERSAL_DEPTH", "graph_query"]

#: BFS depth ceiling. Deeper walks return the whole component on a dense store.
MAX_TRAVERSAL_DEPTH = 3


#: The one edge type that is no longer materialised. It stays a member of
#: :data:`VALID_EDGE_TYPES` — ``trw_graph_related`` still accepts it — but it is
#: answered by derivation over the ``memory_tags`` index instead of by a row.
DERIVED_EDGE_TYPE = "tag_cooccurrence"


def _derive_tag_edges(
    conn: sqlite3.Connection,
    root_ids: list[str],
    *,
    namespace: str | None,
    config: MemoryConfig | None,
) -> list[dict[str, str | int | float]]:
    """Derive depth-1 tag neighbours for each root (PRD-CORE-245 FR07).

    Returns nothing when no namespace was supplied: ``memory_tags`` is keyed on
    ``(namespace, tag, entry_id)`` and an unscoped derivation would span every
    tenant in the file, which is the containment failure this PRD removes.

    Raises:
        sqlite3.Error: propagated from the derivation, exactly as the
            materialised edge queries in :func:`graph_query` propagate theirs. A
            store that cannot be read has no neighbour count to report, and
            returning ``[]`` for it would make an unreadable store and a store
            with no tag relations the same answer. The one suppressed case is a
            pre-schema-5 store with no ``memory_tags`` table, which really does
            hold no derived relation.
    """
    if namespace is None:
        logger.debug("tag_derivation_skipped", reason="no_namespace")
        return []
    from trw_memory.retrieval.tag_derivation import derive_tag_neighbours

    effective_config = config if config is not None else MemoryConfig()
    seen: set[str] = set(root_ids)
    derived: list[dict[str, str | int | float]] = []
    for root_id in root_ids:
        for neighbour in derive_tag_neighbours(conn, root_id, namespace=namespace, config=effective_config):
            if neighbour.entry_id in seen:
                continue
            seen.add(neighbour.entry_id)
            derived.append(
                {
                    "id": neighbour.entry_id,
                    "depth": 1,
                    "edge_type": DERIVED_EDGE_TYPE,
                    "weight": neighbour.weight,
                }
            )
    return derived


def graph_query(
    conn: sqlite3.Connection,
    root_ids: list[str],
    depth: int = 2,
    edge_types: list[str] | None = None,
    namespace: str | None = None,
    max_nodes: int | None = None,
    config: MemoryConfig | None = None,
) -> list[dict[str, str | int | float]]:
    """BFS traversal from root nodes up to specified depth.

    Args:
        conn: SQLite connection.
        root_ids: Starting node IDs.
        depth: Max traversal depth (clamped to 3).
        edge_types: Filter by edge type(s). ``None`` = every MATERIALISED type.
            ``tag_cooccurrence`` is no longer materialised (PRD-CORE-245 FR07):
            it is served only when a caller names it explicitly, by the bounded
            single-root derivation in
            :mod:`trw_memory.retrieval.tag_derivation`, and only at depth 1 —
            deriving it across a multi-root BFS measured 912 ms against 1.0 ms
            for the materialised lookup it replaced. A default traversal
            therefore never returns a derived edge.
        namespace: When provided, restrict traversal to edges whose
            ``target_id`` resolves to a ``memories`` row in this namespace.
            ``memory_graph_edges`` has no namespace column and a single
            SQLite DB holds many namespaces, so without this predicate a
            root from namespace A can follow cross-namespace edges and
            surface (and recurse into) node IDs that belong to namespace B
            — a data-isolation leak. ``None`` keeps the legacy unscoped
            behaviour (mirrors the ``namespace`` scoping added to the
            vector-ops path).
        max_nodes: Optional hard cap on discovered nodes. ``None`` preserves
            the legacy internal traversal contract; public adapters set a cap.

    Returns:
        List of {"id": str, "depth": int, "edge_type": str, "weight": float}
        for each discovered node, excluding root nodes.
    """
    if not root_ids:
        return []
    if max_nodes is not None and max_nodes < 1:
        raise ValueError("max_nodes must be at least 1")

    if depth > MAX_TRAVERSAL_DEPTH:
        logger.debug("graph_query_depth_clamped", requested=depth, clamped=MAX_TRAVERSAL_DEPTH)
        depth = MAX_TRAVERSAL_DEPTH

    # PRD-CORE-245 FR07: split an explicit tag request off the materialised
    # walk. Derived results are APPENDED after materialised ones, so a caller
    # asking for both never sees tag coincidence outrank a semantic relation.
    derived: list[dict[str, str | int | float]] = []
    if edge_types is not None and DERIVED_EDGE_TYPE in edge_types:
        derived = _derive_tag_edges(conn, root_ids, namespace=namespace, config=config)
        edge_types = [edge_type for edge_type in edge_types if edge_type != DERIVED_EDGE_TYPE]
        if not edge_types:
            return derived[:max_nodes] if max_nodes is not None else derived

    if namespace is not None:
        allowed_roots: set[str] = set()
        for chunk in iter_bind_chunks(root_ids, reserved_bindings=1):
            placeholders = ", ".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT id FROM memories WHERE namespace = ? AND id IN ({placeholders})",  # noqa: S608
                (namespace, *chunk),
            ).fetchall()
            allowed_roots.update(str(row[0]) for row in rows)
        root_ids = [root_id for root_id in root_ids if root_id in allowed_roots]
        if not root_ids:
            return []

    visited: set[str] = set(root_ids)
    results: list[dict[str, str | int | float]] = []
    queue: deque[tuple[str, int]] = deque()

    for rid in root_ids:
        queue.append((rid, 0))

    # Namespace scoping is applied as a correlated EXISTS against `memories`
    # so the target row's namespace must match — edges pointing at foreign
    # namespaces (or at deleted rows) are skipped entirely.
    ns_clause = ""
    ns_param: tuple[str, ...] = ()
    if namespace is not None:
        ns_clause = (
            " AND EXISTS (SELECT 1 FROM memories m WHERE m.id = memory_graph_edges.target_id AND m.namespace = ?)"
        )
        ns_param = (namespace,)

    while queue:
        node_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue

        remaining = max_nodes - len(results) if max_nodes is not None else None
        limit_clause = " LIMIT ?" if remaining is not None else ""

        # Build query with optional edge type filter. Public callers pass a
        # node cap, so the storage query itself cannot materialize an unbounded
        # high-fanout row set before Python applies the traversal budget.
        if edge_types:
            placeholders = ", ".join("?" for _ in edge_types)
            sql = (
                f"SELECT target_id, edge_type, weight FROM memory_graph_edges "  # noqa: S608 — placeholders is ? repeated (no user input in SQL structure); values are parameterized
                f"WHERE source_id = ? AND edge_type IN ({placeholders}){ns_clause}{limit_clause}"
            )
            params: tuple[str | int, ...] = (node_id, *edge_types, *ns_param, *((remaining,) if remaining else ()))
        else:
            sql = (
                f"SELECT target_id, edge_type, weight FROM memory_graph_edges "  # noqa: S608 — ns_clause uses a parameterized ? placeholder; no user input in SQL structure
                f"WHERE source_id = ?{ns_clause}{limit_clause}"
            )
            params = (node_id, *ns_param, *((remaining,) if remaining else ()))

        for row in conn.execute(sql, params):
            target_id, edge_type, weight = row
            if target_id not in visited:
                visited.add(target_id)
                results.append(
                    {
                        "id": target_id,
                        "depth": current_depth + 1,
                        "edge_type": edge_type,
                        "weight": weight,
                    }
                )
                if max_nodes is not None and len(results) >= max_nodes:
                    return [*results, *derived][:max_nodes]
                queue.append((target_id, current_depth + 1))

    # Derived tag neighbours are appended AFTER every materialised edge, so the
    # explicit ordering rule in FR07 holds regardless of what the index returned.
    merged = [*results, *derived]
    return merged[:max_nodes] if max_nodes is not None else merged
