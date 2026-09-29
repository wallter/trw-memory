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

#: Root-id bound (B71-28/B71-99, PRD-CORE-307 FR07): recall can pass up to its
#: own limit (10,000) of root ids, and walking that many against a dense store
#: is the other half of the unbounded-work risk, so extra roots are dropped
#: rather than walked. Also the edge-scan basis below when the caller passes
#: no ``max_nodes`` (the legacy internal contract keeps returning every
#: reachable node, but the edge scan itself still needs a finite budget).
#: ``_EDGE_SCAN_BUDGET_MULTIPLIER`` bounds the total ``memory_graph_edges``
#: rows one call may examine across every node it visits, as a multiple of
#: the effective node budget — without it a single hub with far more edges
#: than ``max_nodes`` makes SQLite page through its whole adjacency once per
#: visit regardless of how few nodes are actually wanted.
_MAX_TRAVERSAL_ROOTS, _EDGE_SCAN_BUDGET_MULTIPLIER = 1000, 50


#: The one edge type that is no longer materialised. It stays a member of
#: :data:`VALID_EDGE_TYPES` — the graph traversal callable behind trw-mcp's trw_recall graph mode still accepts it — but it is
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


def _valid_namespace_targets(
    conn: sqlite3.Connection, target_ids: list[object], namespace: str, *, active_only: bool
) -> set[str]:
    """Which of *target_ids* resolve to a ``memories`` row in *namespace* (B71-135(b)).

    Replaces the old per-row correlated ``EXISTS`` in the adjacency SQL: this runs once per PAGE
    (chunked to respect SQLite's bind-parameter limit), not once per candidate edge, so it does not
    reintroduce the per-row cost the caller moved off SQL to keep ``edges_scanned`` an honest count
    of rows examined.
    """
    status_clause = " AND status = 'active'" if active_only else ""
    valid: set[str] = set()
    for chunk in iter_bind_chunks(target_ids, reserved_bindings=1):
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT id FROM memories WHERE namespace = ? AND id IN ({placeholders}){status_clause}",  # noqa: S608
            (namespace, *chunk),
        ).fetchall()
        valid.update(str(row[0]) for row in rows)
    return valid


def graph_query(
    conn: sqlite3.Connection,
    root_ids: list[str],
    depth: int = 2,
    edge_types: list[str] | None = None,
    namespace: str | None = None,
    max_nodes: int | None = None,
    config: MemoryConfig | None = None,
    active_only: bool = False,
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
        active_only: With *namespace*, follow only edges whose target row is
            ACTIVE, so an obsolete node neither fills *max_nodes* nor is walked through.

    Returns:
        List of {"id": str, "depth": int, "edge_type": str, "weight": float}
        for each discovered node, excluding root nodes.
    """
    if not root_ids:
        return []
    if max_nodes is not None and max_nodes < 1:
        raise ValueError("max_nodes must be at least 1")
    if len(root_ids) > _MAX_TRAVERSAL_ROOTS:
        root_ids = root_ids[:_MAX_TRAVERSAL_ROOTS]

    if depth > MAX_TRAVERSAL_DEPTH:
        logger.debug("graph_query_depth_clamped", requested=depth, clamped=MAX_TRAVERSAL_DEPTH)
        depth = MAX_TRAVERSAL_DEPTH

    # PRD-CORE-245 FR07: split an explicit tag request off the materialised
    # walk. Derived results are APPENDED after materialised ones, so a caller
    # asking for both never sees tag coincidence outrank a semantic relation.
    derived: list[dict[str, str | int | float]] = []
    if edge_types is not None and DERIVED_EDGE_TYPE in edge_types:
        derived = _derive_tag_edges(conn, root_ids, namespace=namespace, config=config)
        if active_only and namespace is not None and derived:
            placeholders = ", ".join("?" for _ in derived)
            active = {
                str(row[0])
                for row in conn.execute(
                    f"SELECT id FROM memories WHERE namespace = ? AND status = 'active' AND id IN ({placeholders})",  # noqa: S608
                    (namespace, *(str(d["id"]) for d in derived)),
                )
            }
            derived = [d for d in derived if str(d["id"]) in active]
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

    # The edge itself carries a namespace column on schema 5+; a pre-schema-5
    # table has no edge namespace to test, so the check is skipped for it.
    has_edge_namespace = namespace is not None and "namespace" in {
        str(row[1]) for row in conn.execute("PRAGMA table_info(memory_graph_edges)")
    }
    # B71-135(b): the target-namespace filter used to be a correlated ``EXISTS``
    # inside the adjacency SQL's own WHERE clause, so ``ORDER BY rowid LIMIT ?``
    # could not stop until it found `page_limit` MATCHING rows — on a target
    # namespace few edges resolve into, SQLite still had to walk every
    # candidate row in between (an EXISTS against `memories` is not covered by
    # any index on `memory_graph_edges`), while the Python-side counter below
    # only tallied what came back, undercounting the real work by orders of
    # magnitude. The fix drops the EXISTS predicate from SQL: the adjacency
    # query keeps its cheap, index-bound `edge_type` equality/IN filter (an
    # equality on the second column of ``idx_mge_source`` costs nothing extra
    # to examine — SQLite seeks straight to the matching entries), but the
    # target's namespace is now checked in Python, batched once per PAGE
    # against `memories` rather than once per candidate row, and
    # ``edges_scanned`` below counts the raw page SQLite actually returned
    # before that batched check — the true examined-row count for the one
    # predicate that cannot be answered from an index.
    #
    # B71-99 (PRD-CORE-307 FR07): the ``idx_mge_source`` index already orders
    # each source's rows for equality on source_id (and edge_type), and
    # ``ORDER BY rowid LIMIT ?`` lets SQLite use a bounded top-K sorter instead
    # of grouping the whole adjacency list before any LIMIT applies. Two edge
    # types to one target used to be collapsed in SQL by MAX(weight); that
    # collapse now happens in Python and keeps the FIRST-WRITTEN row instead
    # (documented behaviour change — see CHANGELOG). ``edge_scan_budget``
    # bounds the total rows any one ``graph_query`` call may examine across
    # every node it visits, so a single hub with far more edges than
    # ``max_nodes`` cannot make the walk do unbounded work;
    # ``_MAX_TRAVERSAL_ROOTS`` bounds the seed list itself.
    #
    # B71-135(b) P2/P1: on the DEFAULT (no ``edge_types``) path, ``ORDER BY
    # rowid`` alone needs a ``TEMP B-TREE`` sort of the WHOLE adjacency
    # (``idx_mge_source`` orders by edge_type first); paging by ``(edge_type,
    # rowid)`` fixes that, but ONE query with ``edge_type > ? OR (edge_type = ?
    # AND rowid > ?)`` is not a bounded seek across pages either -- SQLite
    # RE-WALKS the cursor's type from its start on every page, quadratic in
    # page count (146k -> 3.4M VM steps, 100 -> 500 rows/type). The fix is two
    # separate BOUNDED seeks per page, never an OR: (A) finish the CURRENT type
    # (``edge_type = ? AND rowid > ?``), then (B), only if short, take the
    # remainder from LATER types (``edge_type > ?``, a range scan that starts
    # exactly at the next type's first row). ``""`` is a safe "before
    # everything" sentinel: no ``VALID_EDGE_TYPES`` entry is empty, and (A) is
    # skipped while the cursor is still "".
    type_clause = f" AND edge_type IN ({', '.join('?' for _ in edge_types)})" if edge_types else ""
    edge_ns_expr = "namespace" if has_edge_namespace else "NULL"
    edge_ns_clause = " AND memory_graph_edges.namespace = ?" if has_edge_namespace else ""
    sql_same_type = (
        f"SELECT target_id, edge_type, weight, rowid, {edge_ns_expr} AS edge_namespace "  # noqa: S608 — edge_ns_expr is a fixed literal; values below are parameterized
        f"FROM memory_graph_edges WHERE source_id = ? AND edge_type = ?{edge_ns_clause} AND rowid > ? ORDER BY rowid LIMIT ?"
    )
    sql_later_types = (
        f"SELECT target_id, edge_type, weight, rowid, {edge_ns_expr} AS edge_namespace "  # noqa: S608 — edge_ns_expr is a fixed literal; values below are parameterized
        f"FROM memory_graph_edges WHERE source_id = ?{type_clause}{edge_ns_clause} "
        "AND edge_type > ? ORDER BY edge_type, rowid LIMIT ?"
    )
    # A target-namespace filter narrows what a raw page yields once the batched
    # membership check runs, so the tight `remaining`-sized request that keeps
    # the unfiltered path's round-trips minimal is unsafe here: it would ask
    # for exactly as many rows as still-wanted results, which a sparse match
    # rate can never fill, forcing endless small pages. A namespace-filtered
    # request instead asks for the full remaining budget each page. The
    # edge-type filter stays index-bound regardless, so it never needs this.
    has_target_filter = namespace is not None
    edge_scan_budget = (max_nodes or _MAX_TRAVERSAL_ROOTS) * _EDGE_SCAN_BUDGET_MULTIPLIER
    edges_scanned = 0
    while queue:
        if edges_scanned >= edge_scan_budget:
            break
        node_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        last_edge_type, last_rowid = "", 0
        while True:
            remaining = max_nodes - len(results) if max_nodes is not None else None
            budget_left = edge_scan_budget - edges_scanned
            if budget_left <= 0 or (remaining is not None and remaining <= 0):
                break
            page_limit = budget_left if (has_target_filter or remaining is None) else min(remaining, budget_left)
            rows = []
            if last_edge_type:  # "" means nothing examined yet for this node: query A has nothing to continue
                params_a = (
                    node_id,
                    last_edge_type,
                    *((namespace,) if has_edge_namespace else ()),
                    last_rowid,
                    page_limit,
                )
                rows = conn.execute(sql_same_type, params_a).fetchall()
            if len(rows) < page_limit:
                params_b = (
                    node_id,
                    *(edge_types or ()),
                    *((namespace,) if has_edge_namespace else ()),
                    last_edge_type,
                    page_limit - len(rows),
                )
                rows = [*rows, *conn.execute(sql_later_types, params_b).fetchall()]
            edges_scanned += len(rows)  # the FULL raw page across BOTH bounded seeks: what SQLite actually examined
            valid_targets = (
                _valid_namespace_targets(conn, [row[0] for row in rows], namespace, active_only=active_only)
                if namespace is not None and rows
                else None
            )
            for target_id, edge_type, weight, rowid, _edge_namespace in rows:
                last_edge_type, last_rowid = edge_type, rowid
                if valid_targets is not None and target_id not in valid_targets:
                    continue
                if target_id in visited:
                    continue
                visited.add(target_id)
                results.append({"id": target_id, "depth": current_depth + 1, "edge_type": edge_type, "weight": weight})
                if max_nodes is not None and len(results) >= max_nodes:
                    return [*results, *derived][:max_nodes]
                queue.append((target_id, current_depth + 1))
            if len(rows) < page_limit or edges_scanned >= edge_scan_budget:
                break

    # Derived tag neighbours are appended AFTER every materialised edge, so the
    # explicit ordering rule in FR07 holds regardless of what the index returned.
    merged = [*results, *derived]
    return merged[:max_nodes] if max_nodes is not None else merged
