"""MCP tools: memory_admit_shared and memory_vectors -- two recall steps that need a store (PRD-CORE-280 FR01).

A migrated checkout never opens its store, so the two steps of its recall that
read one run here, inside the namespace grant, before any backend is opened:

- ``memory_admit_shared`` runs shared results the checkout fetched from the
  platform through the admission gate of the granted namespace's store. The gate
  rate-limits, audits and quarantines what it refuses, so a second run is not a
  no-op and a lost call is not replayed.
- ``memory_vectors`` returns the stored vectors of some ids in the daemon's active
  embedding space, with that space and its calibrated collapse threshold, for
  near-duplicate collapse (PRD-CORE-302 C2). A read.
- ``memory_record_surfaced`` counts the rows a caller actually showed as accessed
  (and, at session start, surfaced), so over-fetched candidates it filtered out
  never inflate a score. It increments, so a lost call is not replayed.
- ``memory_graph_related`` returns a learning's active knowledge-graph neighbours in
  its namespace, bounded in depth and breadth (trw-mcp's trw_recall graph mode). A read.
"""

from __future__ import annotations

import dataclasses

from trw_memory.embeddings._similarity_calibration import calibrated_threshold
from trw_memory.embeddings._space_gate import active_embedding_space, admit_space_vectors
from trw_memory.graph import MAX_TRAVERSAL_DEPTH, VALID_EDGE_TYPES, graph_query
from trw_memory.models.config import MemoryConfig
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import StorageBackend
from trw_memory.sync._remote_admission import admit_remote_results
from trw_memory.tools._embedder import resolve_embedder
from trw_memory.tools._recall_helpers import GRAPH_RELATED_MAX, hydrate_active
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import serve_namespace


def memory_admit_shared_impl(
    results: list[dict[str, object]], namespace: str, *, backend: StorageBackend, config: MemoryConfig
) -> dict[str, object]:
    """Return ``{"status": "ok", "admitted": [...], "refused": n, "gate_errors": n}``; refusals quarantine in *namespace*."""
    outcome = admit_remote_results(results, config=config, backend=backend, namespace=namespace)
    return {
        "status": "ok",
        "admitted": outcome.admitted,
        "refused": outcome.refused,
        "gate_errors": outcome.gate_errors,
    }


#: Recall's near-duplicate collapse threshold on the reference scale (moved from trw-mcp, PRD-CORE-302 C2).
RECALL_DUP_THRESHOLD = 0.9


def memory_vectors_impl(
    ids: list[str], namespace: str, *, backend: StorageBackend, config: MemoryConfig
) -> dict[str, object]:
    """``{"status": "ok", "vectors": {id: [...]}, "space": {...}, "dup_threshold": t}`` in the active space.

    Vectors, space and threshold come from one call, so a caller can never mix
    model generations; with no embedder the contract C1 ``unavailable`` answer.
    """
    embedder = resolve_embedder(config, surface="memory_vectors")
    if isinstance(embedder, dict):
        return embedder
    space = active_embedding_space(embedder)
    if space is None:
        return {"status": "unavailable", "reason": "embedder_error"}
    records = backend.get_vector_records(list(dict.fromkeys(ids)), namespace=namespace) if ids else {}
    return {
        "status": "ok",
        "vectors": admit_space_vectors(records, space, namespace=namespace, surface="memory_vectors"),
        "space": dataclasses.asdict(space),
        "dup_threshold": calibrated_threshold(RECALL_DUP_THRESHOLD, embedder),
    }


#: The most ids one ``memory_record_surfaced`` call counts.
SURFACED_MAX = 1000


def memory_record_surfaced_impl(
    namespace: str, ids: list[str], session_start: bool, *, backend: StorageBackend
) -> dict[str, object]:
    """Count each distinct id this store holds once as accessed (and surfaced); ``{"status", "counted": [ids]}``."""
    from datetime import datetime, timezone

    from trw_memory.lifecycle._recall import record_recall_access

    held = [i for i in dict.fromkeys(ids) if backend.get(i, namespace=namespace) is not None]
    now = datetime.now(timezone.utc)
    # One transaction: a failed session count must not leave the access counts committed.
    with backend.transaction():
        record_recall_access(backend, held, namespace=namespace, accessed_at=now)
        if session_start and held:
            backend.increment_session_counts(held, namespace=namespace, updated_at=now)
    return {"status": "ok", "counted": held}


def memory_graph_related_impl(
    namespace: str, learning_id: str, depth: int, edge_types: list[str] | None, limit: int, *, backend: StorageBackend
) -> dict[str, object]:
    """``{"status": "ok", "related": [row + edge fields], "truncated": bool}``; the first *limit* nodes are read."""
    conn = getattr(backend, "_conn", None)
    if conn is None:
        return {"error": "graph traversal needs the SQLite store", "status": "unavailable"}
    # Only ACTIVE targets are walked, so the limit+1 window and ``truncated`` count rows the caller can see.
    nodes = graph_query(
        conn,
        [learning_id],
        depth=depth,
        edge_types=edge_types,
        namespace=namespace,
        max_nodes=limit + 1,
        active_only=True,
    )
    return {
        "status": "ok",
        "related": hydrate_active(nodes[:limit], backend, namespace),
        "truncated": len(nodes) > limit,
    }


def register_recall_support_tools(mcp: McpServer) -> None:
    """Register memory_admit_shared and memory_vectors with a FastMCP server instance."""

    async def memory_admit_shared(namespace: str, results: list[dict[str, object]]) -> dict[str, object]:
        """Admit fetched shared results through *namespace*'s write gate; refused ones are quarantined."""
        return await serve_namespace(
            namespace,
            Permission.WRITE,
            "admit_shared",
            lambda backend, config: memory_admit_shared_impl(results, namespace, backend=backend, config=config),
        )

    async def memory_vectors(namespace: str, ids: list[str]) -> dict[str, object]:
        """Stored vectors of *ids* in *namespace* in the active space, with that space and its collapse threshold."""
        # Off the event loop: resolving the space loads the embedding model.
        return await serve_namespace(
            namespace,
            Permission.READ,
            "vectors",
            lambda backend, config: memory_vectors_impl(ids, namespace, backend=backend, config=config),
            exclusive=False,
        )

    mcp.tool()(memory_admit_shared)
    mcp.tool()(memory_vectors)

    async def memory_graph_related(
        namespace: str, learning_id: str, depth: int = 1, edge_types: list[str] | None = None, limit: int = 50
    ) -> dict[str, object]:
        """Active knowledge-graph neighbours of *learning_id* in *namespace*, up to *depth* hops and *limit* rows."""
        if (
            not (1 <= depth <= MAX_TRAVERSAL_DEPTH and 1 <= limit <= GRAPH_RELATED_MAX)
            or not set(edge_types or ()) <= VALID_EDGE_TYPES
        ):
            return {
                "error": f"invalid traversal: depth={depth}, limit={limit}, edge_types={edge_types}",
                "status": "invalid",
            }
        return await serve_namespace(
            namespace,
            Permission.READ,
            "graph_related",
            lambda backend, _config: memory_graph_related_impl(
                namespace, learning_id, depth, edge_types, limit, backend=backend
            ),
        )

    mcp.tool()(memory_graph_related)

    async def memory_record_surfaced(namespace: str, ids: list[str], session_start: bool = False) -> dict[str, object]:
        """Count *ids* of *namespace* as accessed (and surfaced, at session start): the rows a caller showed."""
        if not ids:  # the most is bounded before the call arrives (daemon/_arg_bounds.py)
            return {"error": "ids must hold at least one entry", "status": "invalid"}
        return await serve_namespace(
            namespace,
            Permission.WRITE,
            "record_surfaced",
            lambda backend, _config: memory_record_surfaced_impl(namespace, ids, session_start, backend=backend),
        )

    mcp.tool()(memory_record_surfaced)
