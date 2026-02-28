"""MCP tool: memory_consolidate — cluster and merge similar memory entries.

Thin wrapper around lifecycle.consolidation.consolidate_cycle. Validates
namespace, runs the consolidation cycle (or a dry-run preview), and returns
a structured result dict.
"""

from __future__ import annotations

from typing import Any

import structlog

from trw_memory.exceptions import ConfigError
from trw_memory.lifecycle.consolidation import consolidate_cycle
from trw_memory.namespace import validate_namespace
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger()


def memory_consolidate_impl(
    namespace: str,
    *,
    backend: StorageBackend,
    dry_run: bool = False,
) -> dict[str, object]:
    """Core implementation of memory_consolidate (callable without MCP).

    Args:
        namespace: Namespace to consolidate within (e.g., "project:default").
        backend: Storage backend instance.
        dry_run: If True, preview clusters without modifying storage.

    Returns:
        {"clusters_found": int, "entries_consolidated": int, "dry_run": bool}
        or {"error": str, "status": "invalid"} on validation failure.
    """
    try:
        validate_namespace(namespace)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}

    try:
        result = consolidate_cycle(
            backend,
            embedder=None,  # graceful degradation — no clusters without embedder
            dry_run=dry_run,
            namespace=namespace,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("memory_consolidate_failed", namespace=namespace, error=str(exc))
        return {"error": f"consolidation error: {exc}", "status": "error"}

    # Normalise result keys for the MCP contract
    clusters_found = int(str(result.get("clusters_found", 0)))
    consolidated_count = int(str(result.get("consolidated_count", 0)))

    logger.info(
        "memory_consolidate",
        namespace=namespace,
        dry_run=dry_run,
        clusters_found=clusters_found,
        entries_consolidated=consolidated_count,
    )

    return {
        "clusters_found": clusters_found,
        "entries_consolidated": consolidated_count,
        "dry_run": bool(result.get("dry_run", dry_run)),
    }


def register_consolidate_tool(mcp: Any) -> None:
    """Register memory_consolidate with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from pathlib import Path

    from trw_memory.models.config import MemoryConfig
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    @mcp.tool()  # type: ignore[untyped-decorator]
    async def memory_consolidate(
        namespace: str = "project:default",
        dry_run: bool = False,
    ) -> dict[str, object]:
        """Consolidate similar memory entries by clustering and merging.

        Uses embedding-based clustering to find semantically similar entries,
        then merges each cluster into a single consolidated entry. Originals
        are archived.

        Args:
            namespace: Namespace to consolidate (e.g., 'project:default').
            dry_run: If True, preview clusters without writing changes.

        Returns:
            {"clusters_found": int, "entries_consolidated": int, "dry_run": bool}
        """
        cfg = MemoryConfig()
        db_path = Path(cfg.storage_path) / cfg.sqlite_db_name
        with SQLiteBackend(db_path, dim=cfg.embedding_dim) as backend:
            return memory_consolidate_impl(
                namespace,
                backend=backend,
                dry_run=dry_run,
            )
