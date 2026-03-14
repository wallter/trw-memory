"""MCP tool: memory_forget — delete memory entries by ID or query.

Thin wrapper that validates namespace, then either deletes a single entry
by ID or performs a bulk search-and-delete.
"""

from __future__ import annotations

import structlog

from trw_memory.exceptions import ConfigError, StorageError
from trw_memory.namespace import validate_namespace
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools._types import McpServer

logger = structlog.get_logger()


def memory_forget_impl(
    memory_id: str | None,
    query: str | None,
    namespace: str,
    *,
    backend: StorageBackend,
) -> dict[str, object]:
    """Core implementation of memory_forget (callable without MCP).

    At least one of *memory_id* or *query* must be provided.

    Args:
        memory_id: Specific entry ID to delete. Takes precedence over query.
        query: Free-text query — all matching entries in the namespace are deleted.
        namespace: Namespace scope for the deletion operation.
        backend: Storage backend instance.

    Returns:
        {"deleted": int, "status": "ok"}
        or {"error": str, "status": "invalid"} on validation failure.
    """
    # Validate that at least one selector is provided
    if memory_id is not None and not memory_id.strip():
        return {
            "error": "memory_id must be non-empty when provided.",
            "status": "invalid",
        }
    if not memory_id and not query:
        return {
            "error": "At least one of memory_id or query must be provided.",
            "status": "invalid",
        }

    try:
        validate_namespace(namespace)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}

    # --- Delete by ID (with namespace isolation) ---
    if memory_id:
        deleted_count = 0
        try:
            entry = backend.get(memory_id)
            if entry is not None and entry.namespace == namespace:
                was_deleted = backend.delete(memory_id)
                deleted_count = 1 if was_deleted else 0
        except StorageError as exc:
            logger.warning("memory_forget_delete_error", memory_id=memory_id, error=str(exc))

        logger.info(
            "memory_forget",
            memory_id=memory_id,
            deleted=deleted_count,
            namespace=namespace,
        )
        return {"deleted": deleted_count, "status": "ok"}

    # --- Bulk delete via search query ---
    assert query is not None  # narrowed above
    try:
        matches = backend.search(
            query,
            top_k=10_000,
            namespace=namespace,
        )
    except StorageError as exc:
        logger.warning("memory_forget_search_error", query=query[:80], error=str(exc))
        return {"deleted": 0, "status": "ok"}

    deleted_count = 0
    for entry in matches:
        try:
            if backend.delete(entry.id):
                deleted_count += 1
        except StorageError as exc:
            logger.warning("memory_forget_delete_error", memory_id=entry.id, error=str(exc))

    logger.info(
        "memory_forget_bulk",
        query=query[:80],
        matches=len(matches),
        deleted=deleted_count,
        namespace=namespace,
    )
    return {"deleted": deleted_count, "status": "ok"}


def register_forget_tool(mcp: McpServer) -> None:
    """Register memory_forget with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from pathlib import Path as _Path

    from trw_memory.models.config import MemoryConfig
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    async def memory_forget(
        memory_id: str | None = None,
        query: str | None = None,
        namespace: str = "project:default",
    ) -> dict[str, object]:
        """Delete memory entries by ID or bulk search query.

        Provide *memory_id* to delete a specific entry, or *query* to
        delete all entries matching the search query in the namespace.
        At least one must be provided.

        Args:
            memory_id: Specific memory entry ID to delete.
            query: Free-text query — all matching entries are deleted.
            namespace: Namespace scope for the operation.

        Returns:
            {"deleted": int, "status": "ok"}
        """
        cfg = MemoryConfig()
        db_path = _Path(cfg.storage_path) / cfg.sqlite_db_name
        with SQLiteBackend(db_path, dim=cfg.embedding_dim) as backend:
            return memory_forget_impl(
                memory_id,
                query,
                namespace,
                backend=backend,
            )

    mcp.tool()(memory_forget)
