"""MCP tools: memory_admit_shared and memory_vectors -- two recall steps that need a store (PRD-CORE-280 FR01).

A migrated checkout never opens its store, so the two steps of its recall that
read one run here, inside the namespace grant, before any backend is opened:

- ``memory_admit_shared`` runs shared results the checkout fetched from the
  platform through the admission gate of the granted namespace's store. The gate
  rate-limits, audits and quarantines what it refuses, so a second run is not a
  no-op and a lost call is not replayed.
- ``memory_vectors`` returns the stored vectors of some ids that were encoded in
  one embedding space, for near-duplicate collapse. A read.
- ``memory_record_surfaced`` counts the rows a caller actually showed as accessed
  (and, at session start, surfaced), so over-fetched candidates it filtered out
  never inflate a score. It increments, so a lost call is not replayed.
- ``memory_graph_related`` returns a learning's active knowledge-graph neighbours in
  its namespace, bounded in depth and breadth (trw-mcp's ``trw_graph_related``). A read.
"""

from __future__ import annotations

from trw_memory.embeddings._space_gate import admit_space_vectors
from trw_memory.embeddings.provenance import EmbeddingSpace
from trw_memory.graph import MAX_TRAVERSAL_DEPTH, VALID_EDGE_TYPES, graph_query
from trw_memory.models.config import MemoryConfig
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import StorageBackend
from trw_memory.sync._remote_admission import admit_remote_results
from trw_memory.tools._recall_helpers import hydrate_active
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import in_namespace


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


def memory_vectors_impl(
    ids: list[str], namespace: str, space: dict[str, object], *, backend: StorageBackend
) -> dict[str, object]:
    """Return ``{"status": "ok", "vectors": {id: [...]}}``: only vectors encoded in *space*."""
    try:
        wanted = EmbeddingSpace(**space)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        return {"error": f"invalid embedding space: {exc}", "status": "invalid"}
    records = backend.get_vector_records(list(dict.fromkeys(ids)), namespace=namespace) if ids else {}
    return {
        "status": "ok",
        "vectors": admit_space_vectors(records, wanted, namespace=namespace, surface="memory_vectors"),
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


#: The most neighbours one ``memory_graph_related`` call returns.
GRAPH_RELATED_MAX = 1000


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
        return in_namespace(
            namespace,
            Permission.WRITE,
            "admit_shared",
            lambda backend, config: memory_admit_shared_impl(results, namespace, backend=backend, config=config),
        )

    async def memory_vectors(namespace: str, ids: list[str], space: dict[str, object]) -> dict[str, object]:
        """Return the stored vectors of *ids* in *namespace* that were encoded in *space*."""
        return in_namespace(
            namespace,
            Permission.READ,
            "vectors",
            lambda backend, _config: memory_vectors_impl(ids, namespace, space, backend=backend),
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
        return in_namespace(
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
        if not ids or len(ids) > SURFACED_MAX:
            return {"error": f"ids must hold 1 to {SURFACED_MAX} entries, not {len(ids)}", "status": "invalid"}
        return in_namespace(
            namespace,
            Permission.WRITE,
            "record_surfaced",
            lambda backend, _config: memory_record_surfaced_impl(namespace, ids, session_start, backend=backend),
        )

    mcp.tool()(memory_record_surfaced)
