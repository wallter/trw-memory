"""Graph-augmented recall expansion for MemoryClient.recall().

After BM25+dense+RRF fusion, follows knowledge-graph edges from the
top-K result IDs (root_ids) to their immediate neighbours.  Entries
that appear in the graph but not in the initial result set are
appended with a discounted score so they rank below direct matches
while still surfacing in the final result.

Graceful degradation:
- Backend is ``None`` or has no SQLite connection → returns ``results`` unchanged.
- ``graph_query`` raises → returns ``results`` unchanged (logged at DEBUG).
- No graph edges from any root → returns ``results`` unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from trw_memory.graph import graph_query
from trw_memory.models.memory import MemoryStatus

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient, MemoryResultDict

logger = structlog.get_logger(__name__)

_GRAPH_SCORE_DISCOUNT: float = 0.5


def graph_expand_results(
    client: MemoryClient,
    results: list[MemoryResultDict],
    *,
    depth: int = 1,
) -> list[MemoryResultDict]:
    """Expand *results* by following graph edges from the returned entry IDs.

    Args:
        client: The active MemoryClient (provides backend + connection).
        results: Current recall result set (MemoryResultDict list).
        depth: BFS depth for graph traversal (clamped to 3 by graph_query).
            Default 1 = immediate neighbours only.

    Returns:
        *results* extended with graph-neighbour entries not already present,
        each scored at ``max_score * _GRAPH_SCORE_DISCOUNT * edge_weight``
        and tagged ``source="graph"``.  Returns *results* unchanged when the
        backend has no SQLite connection or the graph is empty.
    """
    if not results:
        return results

    backend = client._backend
    if backend is None:
        return results

    conn = getattr(backend, "_conn", None)
    if conn is None:
        logger.debug("graph_expand_skip", reason="no_sqlite_connection")
        return results

    root_ids = [r["memory_id"] for r in results]

    # Scope BFS to the client's namespace so graph expansion never surfaces
    # neighbours belonging to a foreign namespace (data-isolation leak —
    # memory_graph_edges has no namespace column).
    namespace = getattr(client, "_namespace", None)

    try:
        nodes = graph_query(conn, root_ids, depth=depth, namespace=namespace)
    except Exception:  # pragma: no cover -- sqlite / value errors
        logger.debug("graph_expand_error", exc_info=True)
        return results

    if not nodes:
        return results

    seen_ids: set[str] = {r["memory_id"] for r in results}
    max_score = max((float(r["score"]) for r in results), default=1.0)
    base_score = max_score * _GRAPH_SCORE_DISCOUNT

    expanded: list[MemoryResultDict] = list(results)
    added = 0
    for node in nodes:
        node_id = str(node["id"])
        if node_id in seen_ids:
            continue
        entry = backend.get(node_id)
        if entry is None or entry.status != MemoryStatus.ACTIVE:
            continue
        node_score = base_score * float(node.get("weight", 1.0))
        result = _entry_to_result(entry, score=node_score)
        result["source"] = "graph"
        expanded.append(result)
        seen_ids.add(node_id)
        added += 1

    logger.debug(
        "graph_expand_complete",
        root_count=len(root_ids),
        graph_nodes=len(nodes),
        neighbours_added=added,
    )
    return expanded


def _entry_to_result(entry: object, score: float = 0.0) -> MemoryResultDict:
    from trw_memory._client_distilled_tiering import entry_to_result as _impl
    from trw_memory.models.memory import MemoryEntry

    if not isinstance(entry, MemoryEntry):
        raise TypeError(f"expected MemoryEntry, got {type(entry).__name__}")
    return _impl(entry, score=score)


__all__ = ["graph_expand_results"]
