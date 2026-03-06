"""MCP tool: memory_status — report on the current state of the memory store.

Returns total entry count, namespace breakdown, and a summary of the
active configuration. Namespace is optional — when None, reports globally.
"""

from __future__ import annotations

from typing import Any

import structlog

from trw_memory.exceptions import ConfigError, StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus
from trw_memory.namespace import validate_namespace
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger()

# Canonical namespaces to include in per-namespace breakdown when no filter applied
_COMMON_NAMESPACES = ["project:default", "global"]


def memory_status_impl(
    namespace: str | None,
    *,
    backend: StorageBackend,
    config: MemoryConfig | None = None,
) -> dict[str, object]:
    """Core implementation of memory_status (callable without MCP).

    Args:
        namespace: If provided, scope the count to this namespace.
            Must be a valid namespace if given.
        backend: Storage backend instance.
        config: MemoryConfig for the configuration summary. Uses defaults if None.

    Returns:
        {"total_entries": int, "namespaces": dict, "config": dict}
        or {"error": str, "status": "invalid"} on validation failure.
    """
    cfg = config or MemoryConfig()

    # Validate namespace if provided
    if namespace is not None:
        try:
            validate_namespace(namespace)
        except ConfigError as exc:
            return {"error": str(exc), "status": "invalid"}

    try:
        total_entries = backend.count(namespace=namespace)
    except Exception as exc:  # broad catch: tool error boundary
        logger.exception("memory_status_count_failed", error=str(exc))
        return {"error": f"storage error: {exc}", "status": "error"}

    # Build namespace breakdown
    namespaces: dict[str, object] = {}
    if namespace is not None:
        namespaces[namespace] = total_entries
    else:
        # Try to count a few common namespaces
        for ns in _COMMON_NAMESPACES:
            try:
                ns_count = backend.count(namespace=ns)
                namespaces[ns] = ns_count
            except Exception:  # broad catch: best-effort namespace count
                pass

        # Also include active entry count
        try:
            active_entries = backend.list_entries(
                status=MemoryStatus.ACTIVE,
                limit=10_000,
            )
            namespaces["__active__"] = len(active_entries)
        except Exception:  # broad catch: best-effort active count
            pass

    config_summary: dict[str, object] = {
        "storage_backend": cfg.storage_backend,
        "dedup_enabled": cfg.dedup_enabled,
        "consolidation_enabled": cfg.consolidation_enabled,
        "hot_max_entries": cfg.hot_max_entries,
        "retention_days": cfg.retention_days,
    }

    logger.debug(
        "memory_status",
        namespace=namespace,
        total_entries=total_entries,
    )

    return {
        "total_entries": total_entries,
        "namespaces": namespaces,
        "config": config_summary,
    }


def register_status_tool(mcp: Any) -> None:
    """Register memory_status with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from pathlib import Path

    from trw_memory.storage.sqlite_backend import SQLiteBackend

    @mcp.tool()  # type: ignore[untyped-decorator]
    async def memory_status(
        namespace: str | None = None,
    ) -> dict[str, object]:
        """Report the current state of the memory store.

        Args:
            namespace: If provided, scope to this namespace (e.g., 'project:default').
                When None, reports across all namespaces.

        Returns:
            {"total_entries": int, "namespaces": dict, "config": dict}
        """
        cfg = MemoryConfig()
        db_path = Path(cfg.storage_path) / cfg.sqlite_db_name
        with SQLiteBackend(db_path, dim=cfg.embedding_dim) as backend:
            return memory_status_impl(
                namespace,
                backend=backend,
                config=cfg,
            )
