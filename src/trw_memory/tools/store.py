"""MCP tool: memory_store — persist a new memory entry.

Validates namespace, creates a MemoryEntry with a unique M-prefixed ID,
stores it via the backend, and returns the memory_id and status.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog

from trw_memory.exceptions import ConfigError, StorageError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.namespace import validate_namespace
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger()


def memory_store_impl(
    content: str,
    namespace: str,
    *,
    backend: StorageBackend,
    tags: list[str] | None = None,
    importance: float = 0.5,
    detail: str = "",
    metadata: dict[str, str] | None = None,
) -> dict[str, object]:
    """Core implementation of memory_store (callable without MCP).

    Args:
        content: Core knowledge statement to store. Must be non-empty.
        namespace: Namespace scope (e.g., "project:default", "global").
        backend: Storage backend instance.
        tags: Optional list of tags to associate with the entry.
        importance: Importance score in [0.0, 1.0]. Defaults to 0.5.
        detail: Extended explanation or context. Defaults to "".
        metadata: Optional string key-value metadata. Defaults to {}.

    Returns:
        {"memory_id": str, "status": "stored", "namespace": str}
        or {"error": str, "status": "invalid"} on validation failure.
    """
    # Validate namespace
    try:
        validate_namespace(namespace)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}

    # Validate content
    if not content or not content.strip():
        return {"error": "content must be a non-empty string", "status": "invalid"}

    # Validate importance range
    if not (0.0 <= importance <= 1.0):
        return {
            "error": f"importance must be in [0.0, 1.0], got {importance}",
            "status": "invalid",
        }

    entry_id = "M-" + uuid4().hex
    now = datetime.now(timezone.utc)

    entry = MemoryEntry(
        id=entry_id,
        content=content.strip(),
        detail=detail,
        tags=tags or [],
        importance=importance,
        metadata=metadata or {},
        namespace=namespace,
        status=MemoryStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )

    try:
        backend.store(entry)
    except Exception as exc:  # broad catch: tool error boundary
        logger.exception("memory_store_failed", entry_id=entry_id, error=str(exc))
        return {"error": f"storage error: {exc}", "status": "error"}

    logger.info(
        "memory_stored",
        entry_id=entry_id,
        namespace=namespace,
        tags=tags or [],
    )

    return {
        "memory_id": entry_id,
        "status": "stored",
        "namespace": namespace,
    }


def register_store_tool(mcp: Any) -> None:
    """Register memory_store with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from pathlib import Path

    from trw_memory.models.config import MemoryConfig
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    @mcp.tool()  # type: ignore[untyped-decorator]
    async def memory_store(
        content: str,
        namespace: str = "project:default",
        tags: list[str] | None = None,
        importance: float = 0.5,
        detail: str = "",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """Store a new memory entry in the memory system.

        Args:
            content: Core knowledge statement to remember. Must be non-empty.
            namespace: Namespace scope (e.g., 'project:default', 'global').
            tags: Optional list of tags for categorisation.
            importance: Importance score 0.0-1.0 (default 0.5).
            detail: Extended explanation or context.
            metadata: Optional key-value string metadata.

        Returns:
            {"memory_id": str, "status": "stored", "namespace": str}
        """
        cfg = MemoryConfig()
        db_path = Path(cfg.storage_path) / cfg.sqlite_db_name
        with SQLiteBackend(db_path, dim=cfg.embedding_dim) as backend:
            return memory_store_impl(
                content,
                namespace,
                backend=backend,
                tags=tags,
                importance=importance,
                detail=detail,
                metadata=metadata,
            )
