"""MCP tool: memory_recall — hybrid search across memory entries.

Thin wrapper that validates namespace, delegates to the retrieval pipeline,
applies score filtering, and returns a structured result dict.
"""

from __future__ import annotations

from typing import Any

import structlog

from trw_memory.exceptions import ConfigError
from trw_memory.lifecycle.scoring import entry_utility, rank_by_utility
from trw_memory.models.memory import MemoryStatus
from trw_memory.namespace import validate_namespace
from trw_memory.retrieval import hybrid_search
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger()


def memory_recall_impl(
    query: str,
    namespace: str,
    *,
    backend: StorageBackend,
    limit: int = 25,
    min_score: float = 0.0,
    tags: list[str] | None = None,
    include_namespaces: list[str] | None = None,
) -> dict[str, object]:
    """Core implementation of memory_recall (callable without MCP).

    Args:
        query: Free-text search query. Empty string returns all active entries.
        namespace: Primary namespace to search (e.g., "project:default").
        backend: Storage backend instance.
        limit: Maximum number of results to return.
        min_score: Minimum utility score threshold (0.0 = no filter).
        tags: If provided, only entries containing ALL of these tags are returned.
        include_namespaces: Additional namespaces to search alongside primary.

    Returns:
        {"memories": list[dict], "total_matches": int, "query": str}
        or {"error": str, "status": "invalid"} on validation failure.
    """
    try:
        validate_namespace(namespace)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}

    # Validate any additional namespaces
    extra_ns: list[str] = []
    for ns in (include_namespaces or []):
        try:
            validate_namespace(ns)
            extra_ns.append(ns)
        except ConfigError:
            pass  # skip invalid extra namespaces silently

    all_namespaces = [namespace] + extra_ns

    # Gather active entries across all requested namespaces
    all_entries = []
    for ns in all_namespaces:
        ns_entries = backend.list_entries(
            status=MemoryStatus.ACTIVE,
            namespace=ns,
            limit=10_000,
        )
        all_entries.extend(ns_entries)

    # Apply tag filter
    if tags:
        tag_set = set(tags)
        all_entries = [e for e in all_entries if tag_set.issubset(set(e.tags))]

    # Retrieve via hybrid search (gracefully degrades to BM25-only or empty)
    if query and all_entries:
        ranked = hybrid_search(
            query=query,
            entries=all_entries,
            top_k=limit * 4,  # over-fetch before score filtering
        )
    else:
        # Empty query: return all entries sorted by utility
        ranked = all_entries

    # Convert to dicts for scoring
    entry_dicts = [e.model_dump(mode="json") for e in ranked]

    # Re-rank by utility using scoring layer
    query_tokens = query.lower().split() if query else []
    ranked_dicts = rank_by_utility(entry_dicts, query_tokens, lambda_weight=0.4)

    # Apply min_score filter and limit
    if min_score > 0.0:
        ranked_dicts = [
            d for d in ranked_dicts
            if entry_utility(d) >= min_score
        ]

    result_dicts = ranked_dicts[:limit]

    logger.debug(
        "memory_recall",
        query=query[:80] if query else "(wildcard)",
        namespace=namespace,
        total_candidates=len(all_entries),
        returned=len(result_dicts),
    )

    return {
        "memories": result_dicts,
        "total_matches": len(result_dicts),
        "query": query,
    }


def register_recall_tool(mcp: Any) -> None:
    """Register memory_recall with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from pathlib import Path as _Path

    from trw_memory.models.config import MemoryConfig
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    async def memory_recall(
        query: str,
        namespace: str = "project:default",
        limit: int = 25,
        min_score: float = 0.0,
        tags: list[str] | None = None,
        include_namespaces: list[str] | None = None,
    ) -> dict[str, object]:
        """Search memory entries using hybrid BM25 + vector retrieval.

        Args:
            query: Free-text search query.
            namespace: Namespace scope (e.g., 'project:default', 'global').
            limit: Maximum results to return (default 25).
            min_score: Minimum utility score filter (0.0 = no filter).
            tags: Filter to entries containing ALL of these tags.
            include_namespaces: Additional namespaces to search alongside primary.

        Returns:
            {"memories": [...], "total_matches": int, "query": str}
        """
        cfg = MemoryConfig()
        db_path = _Path(cfg.storage_path) / cfg.sqlite_db_name
        with SQLiteBackend(db_path, dim=cfg.embedding_dim) as backend:
            return memory_recall_impl(
                query,
                namespace,
                backend=backend,
                limit=limit,
                min_score=min_score,
                tags=tags,
                include_namespaces=include_namespaces,
            )

    mcp.tool()(memory_recall)
