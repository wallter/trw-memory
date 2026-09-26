"""MCP tool: memory_reembed -- move a namespace's vectors into the active space (PRD-CORE-302 FR07).

The daemon owns the store, so a re-embed run anywhere else would be refused as a
second writer. This runs the SDK's contract (``_client_reembed.reembed_rows``:
idempotent, resumable keyset pages, fail closed without an identifiable space)
inside the daemon. Rows are re-encoded from their current text, never re-stamped:
provenance on a vector computed from other text would be a false claim.
"""

from __future__ import annotations

from trw_memory._client_reembed import reembed_rows
from trw_memory.exceptions import EmbeddingUnavailableError, StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools._embedder import coverage_status, resolve_embedder
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import serve_namespace

__all__ = ["memory_reembed_impl", "register_reembed_tool"]


def memory_reembed_impl(
    namespace: str, cursor: str | None, *, backend: StorageBackend, config: MemoryConfig
) -> dict[str, object]:
    """One bounded pass: ``{"status": "ok", <counts>, "cursor": ...}``, or why nothing was re-encoded.

    Call again with the returned ``cursor`` until it is ``None``; only the last pass reports
    ``outside_active_space`` (rc9 sweep B2: a call never holds a daemon worker for a whole namespace).
    """
    embedder = resolve_embedder(config, surface="reembed")
    if isinstance(embedder, dict):
        return embedder
    try:
        result = reembed_rows(backend, embedder, namespace=namespace, config=config, cursor=cursor, bounded=True)
    except ValueError as exc:
        return {"status": "invalid", "error": str(exc)}
    except (EmbeddingUnavailableError, StorageError) as exc:
        return {"status": "unavailable", "reason": "embedder_error", "error": str(exc)}
    coverage = coverage_status(backend, namespace, config) if result["cursor"] is None else None
    return {"status": "ok", **result, "outside_active_space": coverage and coverage["outside_active_space"]}


def register_reembed_tool(mcp: McpServer) -> None:
    """Register memory_reembed with a FastMCP server instance."""

    async def memory_reembed(namespace: str, cursor: str | None = None) -> dict[str, object]:
        """Re-encode *namespace*'s vectors outside the active embedding space, one bounded pass per call; safe to rerun."""
        return await serve_namespace(
            namespace,
            Permission.WRITE,
            "reembed",
            lambda backend, config: memory_reembed_impl(namespace, cursor, backend=backend, config=config),
            exclusive=False,
        )

    mcp.tool()(memory_reembed)
