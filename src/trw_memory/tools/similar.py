"""MCP tool: memory_similar -- the trw_learn dedup KNN, run where the store is (PRD-CORE-280 FR01).

A migrated checkout never opens its store, so its near-duplicate check sends the
new learning's vector and the space it was encoded in; the daemon runs the KNN
over the granted namespace and applies the same comparability rule trw-mcp used
in-process (``comparable_neighbours``). ``complete: false`` means the window
cannot support a dense verdict and the caller decides exhaustively. A read.
"""

from __future__ import annotations

from trw_memory.embeddings._space_gate import comparable_neighbours
from trw_memory.embeddings.provenance import EmbeddingSpace
from trw_memory.models.memory import MemoryStatus
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import in_namespace


def memory_similar_impl(
    namespace: str, vector: list[float], space: dict[str, object] | None, top_k: int, *, backend: StorageBackend
) -> dict[str, object]:
    """``{"status": "ok", "window": n, "complete": bool, "hits": [{"id", "similarity", "active"}]}``.

    ``window`` is how many stored vectors the KNN returned. With no *space* nothing is
    provably comparable: the window is reported and no hit is (nor any vector read).
    """
    wanted: EmbeddingSpace | None = None
    if space is not None:
        try:
            wanted = EmbeddingSpace(**space)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            return {"error": f"invalid embedding space: {exc}", "status": "invalid"}
    if wanted is None:
        size = len(backend.search_vectors(vector, top_k=top_k, namespace=namespace))
        return {"status": "ok", "window": size, "complete": True, "hits": []}
    window = comparable_neighbours(backend, vector, wanted, namespace=namespace, top_k=top_k, surface="memory_similar")
    if window is None:
        return {"status": "ok", "window": top_k, "complete": False, "hits": []}
    hits: list[dict[str, object]] = []
    for entry_id, distance in window:
        entry = backend.get(entry_id, namespace=namespace)
        if entry is not None:
            # Unit-normalised vectors: distance² = 2 * (1 - cosine_similarity).
            similarity = 1.0 - (distance * distance) / 2.0
            hits.append({"id": entry_id, "similarity": similarity, "active": entry.status == MemoryStatus.ACTIVE})
    return {"status": "ok", "window": len(window), "complete": True, "hits": hits}


def register_similar_tool(mcp: McpServer) -> None:
    """Register memory_similar with a FastMCP server instance."""

    async def memory_similar(
        namespace: str, vector: list[float], space: dict[str, object] | None, top_k: int = 10
    ) -> dict[str, object]:
        """Nearest stored learnings in *namespace* to *vector* (encoded in *space*), for duplicate detection."""
        return in_namespace(
            namespace,
            Permission.READ,
            "similar",
            lambda backend, _config: memory_similar_impl(namespace, vector, space, top_k, backend=backend),
        )

    mcp.tool()(memory_similar)


__all__ = ["memory_similar_impl", "register_similar_tool"]
