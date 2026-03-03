"""MCP tool: memory_recall — hybrid search across memory entries.

Thin wrapper that validates namespace, delegates to the retrieval pipeline,
applies score filtering, and returns a structured result dict.

When graph_depth > 0, the graph is queried for related entries (BFS traversal)
and they are appended under a "related" key in the response.
"""

from __future__ import annotations

import sqlite3
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
    graph_depth: int = 0,
    conn: sqlite3.Connection | None = None,
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
        graph_depth: If > 0, run BFS graph traversal from result IDs up to this
            depth and include related entries in the response.
        conn: SQLite connection for graph queries. If None and graph_depth > 0,
            the backend's internal connection is used (SQLiteBackend only).

    Returns:
        {"memories": list[dict], "total_matches": int, "query": str,
         "related": list[dict] (when graph_depth > 0)}
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
            logger.debug("recall_invalid_namespace_skipped", namespace=ns)

    all_namespaces = [namespace, *extra_ns]

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

    response: dict[str, object] = {
        "memories": result_dicts,
        "total_matches": len(result_dicts),
        "query": query,
    }

    # Graph traversal for related entries
    if graph_depth > 0 and result_dicts:
        related = _graph_related(result_dicts, graph_depth, backend, conn)
        response["related"] = related

    return response


def _graph_related(
    result_dicts: list[dict[str, Any]],
    depth: int,
    backend: StorageBackend,
    conn: sqlite3.Connection | None,
) -> list[dict[str, Any]]:
    """Query the knowledge graph for entries related to the recall results.

    Args:
        result_dicts: The primary recall results (as dicts).
        depth: BFS traversal depth.
        backend: Storage backend (used to resolve _conn if conn is None).
        conn: Explicit SQLite connection, or None.

    Returns:
        List of {"id": str, "depth": int, "edge_type": str, "weight": float}
        for each related node discovered.
    """
    from trw_memory.graph import graph_query

    # Resolve connection: explicit > backend._conn
    effective_conn = conn
    if effective_conn is None:
        effective_conn = getattr(backend, "_conn", None)
    if effective_conn is None:
        logger.debug("graph_related_skip", reason="no_sqlite_connection")
        return []

    root_ids = [
        str(d["id"]) for d in result_dicts if "id" in d
    ]

    try:
        return graph_query(effective_conn, root_ids, depth=depth)
    except Exception:
        logger.debug("graph_related_error", exc_info=True)
        return []


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
        graph_depth: int = 0,
    ) -> dict[str, object]:
        """Search memory entries using hybrid BM25 + vector retrieval.

        Args:
            query: Free-text search query.
            namespace: Namespace scope (e.g., 'project:default', 'global').
            limit: Maximum results to return (default 25).
            min_score: Minimum utility score filter (0.0 = no filter).
            tags: Filter to entries containing ALL of these tags.
            include_namespaces: Additional namespaces to search alongside primary.
            graph_depth: If > 0, include graph-related entries via BFS traversal
                from the result set (max depth 3).

        Returns:
            {"memories": [...], "total_matches": int, "query": str,
             "related": [...] (when graph_depth > 0)}
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
                graph_depth=graph_depth,
            )

    mcp.tool()(memory_recall)
