"""MCP tool: memory_recall — hybrid search across memory entries.

Thin wrapper that validates namespace, delegates to the retrieval pipeline,
applies score filtering, and returns a structured result dict.

When graph_depth > 0, the graph is queried for related entries (BFS traversal)
and they are appended under a "related" key in the response.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime, timezone

import structlog

from trw_memory.embeddings import get_local_embedder
from trw_memory.exceptions import ConfigError
from trw_memory.graph import list_org_shared_entries
from trw_memory.lifecycle._recall import record_recall_access
from trw_memory.lifecycle.scoring import entry_utility, rank_by_utility
from trw_memory.lifecycle.tiers._runtime import remember_entry_data_in_tiers, supports_tier_runtime, tier_candidates
from trw_memory.lifecycle.tiers._scoring import compute_importance_score
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.retrieval import hybrid_search
from trw_memory.security.rbac import Permission, require_namespace_permission
from trw_memory.security.runtime import append_audit_event
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools._types import McpServer

logger = structlog.get_logger(__name__)


def memory_recall_impl(  # noqa: C901 - existing orchestration-heavy recall pipeline; security hook was added surgically
    query: str,
    namespace: str,
    *,
    backend: StorageBackend,
    namespace_backend_factory: Callable[[str], StorageBackend] | None = None,
    limit: int = 25,
    min_score: float = 0.0,
    tags: list[str] | None = None,
    include_namespaces: list[str] | None = None,
    include_org_memories: bool = True,
    graph_depth: int = 0,
    conn: sqlite3.Connection | None = None,
    token_budget: int | None = None,
    config: MemoryConfig | None = None,
) -> dict[str, object]:
    """Core implementation of memory_recall (callable without MCP).

    Args:
        query: Free-text search query. Empty string returns all active entries.
        namespace: Primary namespace to search (e.g., "project:default").
        backend: Storage backend instance.
        namespace_backend_factory: Optional factory for opening additional
            namespace-scoped backends when include_namespaces is provided.
        limit: Maximum number of results to return.
        min_score: Minimum utility score threshold (0.0 = no filter).
        tags: If provided, only entries containing ALL of these tags are returned.
        include_namespaces: Additional namespaces to search alongside primary.
        include_org_memories: If True, append cross-validated high-importance
            memories from sibling local project namespaces.
        graph_depth: If > 0, run BFS graph traversal from result IDs up to this
            depth and include related entries in the response.
        conn: SQLite connection for graph queries. If None and graph_depth > 0,
            the backend's internal connection is used (SQLiteBackend only).
        token_budget: If provided, truncate results to fit within this token
            budget.  Must be a positive integer.  ``None`` disables budget
            fitting (all results returned up to *limit*).

    Returns:
        {"memories": list[dict], "total_matches": int, "query": str,
         "tokens_used": int, "tokens_budget": int | None,
         "tokens_truncated": bool,
         "related": list[dict] (when graph_depth > 0)}
        or {"error": str, "status": "invalid"} on validation failure.

    Raises:
        ValueError: If *token_budget* is not ``None`` and <= 0.
    """
    if token_budget is not None and token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")

    try:
        validate_namespace(namespace)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}
    cfg = config or MemoryConfig()
    require_namespace_permission(cfg, namespace, Permission.READ, "recall")

    if (
        namespace.startswith("team:")
        and isinstance(backend, SQLiteBackend)
        and NamespaceManager(backend).team_namespace_expired(namespace)
    ):
        logger.debug("memory_recall_team_namespace_expired", namespace=namespace)
        return {
            "memories": [],
            "total_matches": 0,
            "query": query,
            "tokens_used": 0,
            "tokens_budget": token_budget,
            "tokens_truncated": False,
            "namespace_expired": True,
        }

    # Validate any additional namespaces
    extra_ns: list[str] = []
    for ns in include_namespaces or []:
        try:
            validate_namespace(ns)
            extra_ns.append(ns)
        except ConfigError:  # per-item error handling: skip invalid namespaces, continue with valid ones  # noqa: PERF203
            logger.debug("recall_invalid_namespace_skipped", namespace=ns)

    all_namespaces = [namespace, *extra_ns]
    # Namespace-scoped local backends can only see one store at a time, so
    # cross-namespace recall must reopen the requested namespaces explicitly.
    all_entries = []
    stored_embeddings: dict[str, list[float]] = {}
    seen_namespaces: set[str] = set()
    with ExitStack() as stack:
        for ns in all_namespaces:
            if ns in seen_namespaces:
                continue
            seen_namespaces.add(ns)

            ns_backend = backend
            if ns != namespace and namespace_backend_factory is not None:
                ns_backend = stack.enter_context(namespace_backend_factory(ns))

            if (
                ns.startswith("team:")
                and isinstance(ns_backend, SQLiteBackend)
                and NamespaceManager(ns_backend).team_namespace_expired(ns)
            ):
                logger.debug("recall_expired_namespace_skipped", namespace=ns)
                continue

            ns_entries = ns_backend.list_entries(
                status=MemoryStatus.ACTIVE,
                namespace=ns,
                limit=10_000,
            )
            all_entries.extend(ns_entries)

            if query and ns_entries:
                stored_embeddings.update(
                    ns_backend.get_stored_embeddings([entry.id for entry in ns_entries])
                )

    # Apply tag filter
    if tags:
        tag_set = set(tags)
        all_entries = [e for e in all_entries if tag_set.issubset(set(e.tags))]

    # Retrieve via hybrid search (gracefully degrades to BM25-only or empty)
    if query and all_entries:
        embedder = get_local_embedder(model_name=cfg.embedding_model, dim=cfg.embedding_dim)
        query_embedding = embedder.embed(query) if embedder is not None else None
        # dense_search() needs the stored vector map, not just the entry IDs, so
        # tool recall must hydrate the embeddings before calling hybrid_search().
        ranked = hybrid_search(
            query=query,
            entries=all_entries,
            embedder=embedder,
            stored_embeddings=stored_embeddings or None,
            top_k=limit * 4,  # over-fetch before score filtering
        )
    else:
        query_embedding = None
        # Empty query: return all entries sorted by utility
        ranked = all_entries

    # Convert to dicts for scoring
    entry_dicts = [e.model_dump(mode="json") for e in ranked]

    # Re-rank by utility using scoring layer
    query_tokens = query.lower().split() if query else []
    ranked_dicts = rank_by_utility(entry_dicts, query_tokens, lambda_weight=0.4)
    tier_dicts: list[dict[str, object]] = []
    if supports_tier_runtime(backend):
        tier_dicts = tier_candidates(
            cfg,
            namespace,
            backend,
            query=query,
            tags=tags,
            limit=limit,
            query_embedding=query_embedding,
        )
    if tier_dicts:
        ranked_dicts = _merge_tier_entries(ranked_dicts, tier_dicts, query_tokens, cfg, query_embedding)

    # Apply min_score filter
    if min_score > 0.0:
        ranked_dicts = [d for d in ranked_dicts if float(str(d.get("score", entry_utility(d)))) >= min_score]

    # Token budget fitting BEFORE limit cap (PRD-CORE-123 FR03)
    from trw_memory.retrieval.token_budget import (
        apply_token_budget,
        estimate_entry_tokens,
    )

    result_dicts = ranked_dicts
    if include_org_memories:
        result_dicts.extend(
            _org_memory_results(
                cfg,
                namespace,
                query,
                tags,
                min_score=min_score,
                exclude_keys={
                    (str(result.get("namespace", namespace)), str(result["id"]))
                    for result in result_dicts
                    if "id" in result
                },
                limit=limit,
            )
        )
    tokens_used = 0
    tokens_truncated = False

    if token_budget is not None and result_dicts:
        result_dicts, tokens_used, tokens_truncated = apply_token_budget(
            result_dicts, token_budget
        )
    else:
        tokens_used = sum(estimate_entry_tokens(d) for d in result_dicts)

    # Apply limit cap AFTER token budget
    result_dicts = result_dicts[:limit]
    _record_access_by_namespace(result_dicts, backend, namespace, namespace_backend_factory)
    append_audit_event(
        cfg,
        "access",
        namespace=namespace,
        data={
            "entries_returned": len(result_dicts),
            "tag_filter": tags or [],
            "query": query,
        },
    )
    append_audit_event(
        cfg,
        "recall",
        namespace=namespace,
        data={
            "query": query,
            "total_matches": len(result_dicts),
            "graph_depth": graph_depth,
            "tokens_used": tokens_used,
            "tokens_truncated": tokens_truncated,
        },
    )
    if supports_tier_runtime(backend):
        recalled_at = datetime.now(timezone.utc).isoformat()
        for item in result_dicts:
            tier_payload = dict(item)
            tier_payload["last_accessed_at"] = recalled_at
            remember_entry_data_in_tiers(cfg, tier_payload)

    logger.debug(
        "memory_recall",
        query=query[:80] if query else "(wildcard)",
        namespace=namespace,
        total_candidates=len(all_entries),
        returned=len(result_dicts),
        tokens_used=tokens_used,
        tokens_budget=token_budget,
        tokens_truncated=tokens_truncated,
    )

    response: dict[str, object] = {
        "memories": result_dicts,
        "total_matches": len(result_dicts),
        "query": query,
        "tokens_used": tokens_used,
        "tokens_budget": token_budget,
        "tokens_truncated": tokens_truncated,
    }

    # Graph traversal for related entries
    if graph_depth > 0 and result_dicts:
        related = _graph_related(result_dicts, graph_depth, backend, conn)
        response["related"] = related

    return response


def _merge_tier_entries(
    ranked_dicts: list[dict[str, object]],
    tier_dicts: list[dict[str, object]],
    query_tokens: list[str],
    config: MemoryConfig,
    query_embedding: list[float] | None,
) -> list[dict[str, object]]:
    """Merge tier-only matches into the main recall candidate set."""
    merged: list[dict[str, object]] = list(ranked_dicts)
    seen_keys = {
        (str(item.get("namespace", "project:default")), str(item.get("id", "")))
        for item in ranked_dicts
    }
    for item in tier_dicts:
        key = (str(item.get("namespace", "project:default")), str(item.get("id", "")))
        if key in seen_keys:
            continue
        merged.append(item)
        seen_keys.add(key)
    for item in merged:
        relevance_hint = item.get("_tier_relevance")
        item["score"] = compute_importance_score(
            item,
            query_tokens,
            query_embedding=query_embedding,
            config=config,
            relevance_hint=float(str(relevance_hint)) if relevance_hint is not None else None,
        )
    merged.sort(key=lambda entry: float(str(entry.get("score", 0.0))), reverse=True)
    return merged


def _org_memory_results(
    config: MemoryConfig,
    namespace: str,
    query: str,
    tags: list[str] | None,
    min_score: float,
    *,
    exclude_keys: set[tuple[str, str]],
    limit: int,
) -> list[dict[str, object]]:
    """Build additive org-wide recall results from sibling project stores."""
    org_entries = list_org_shared_entries(
        config,
        namespace,
        exclude_keys=exclude_keys,
        limit=max(limit, 25),
    )
    if not org_entries:
        return []

    query_tokens = query.lower().split() if query else []
    tag_set = set(tags or [])
    org_results = []
    for entry in org_entries:
        if tag_set and not tag_set.issubset(set(entry.tags)):
            continue
        if query_tokens and not _entry_matches_query(entry.model_dump(mode="json"), query_tokens):
            continue

        item = entry.model_dump(mode="json")
        item["scope"] = "org"
        if min_score > 0.0 and entry_utility(item, config=config) < min_score:
            continue
        org_results.append(item)

    return rank_by_utility(org_results, query_tokens, lambda_weight=0.4, config=config)


def _entry_matches_query(entry: dict[str, object], query_tokens: list[str]) -> bool:
    """Return whether any query token appears in content, detail, or tags."""
    if not query_tokens:
        return True
    content = str(entry.get("content", "")).lower()
    detail = str(entry.get("detail", "")).lower()
    raw_tags = entry.get("tags", [])
    tag_text = " ".join(str(tag).lower() for tag in raw_tags) if isinstance(raw_tags, list) else ""
    return any(token in f"{content} {detail} {tag_text}" for token in query_tokens)


def _graph_related(
    result_dicts: list[dict[str, object]],
    depth: int,
    backend: StorageBackend,
    conn: sqlite3.Connection | None,
) -> list[dict[str, object]]:
    """Query the knowledge graph for entries related to the recall results.

    Args:
        result_dicts: The primary recall results (as dicts).
        depth: BFS traversal depth.
        backend: Storage backend (used to resolve _conn if conn is None).
        conn: Explicit SQLite connection, or None.

    Returns:
        Related entry dicts augmented with graph metadata (`depth`, `edge_type`,
        `weight`). Entries missing from storage are skipped.
    """
    from trw_memory.graph import graph_query

    # Resolve connection: explicit > backend._conn
    effective_conn = conn
    if effective_conn is None:
        effective_conn = getattr(backend, "_conn", None)
    if effective_conn is None:
        logger.debug("graph_related_skip", reason="no_sqlite_connection")
        return []

    root_ids = [str(d["id"]) for d in result_dicts if "id" in d]

    try:
        related_nodes = graph_query(effective_conn, root_ids, depth=depth)
    except (sqlite3.Error, ValueError, KeyError):
        logger.debug("graph_related_error", exc_info=True)
        return []

    hydrated: list[dict[str, object]] = []
    for node in related_nodes:
        entry = backend.get(str(node["id"]))
        if entry is None:
            continue
        item = entry.model_dump(mode="json")
        item.update(node)
        hydrated.append(item)
    return hydrated


def _record_access_by_namespace(
    result_dicts: list[dict[str, object]],
    backend: StorageBackend,
    namespace: str,
    namespace_backend_factory: Callable[[str], StorageBackend] | None,
) -> None:
    """Persist access metadata for returned entries across namespace stores."""
    grouped: dict[str, list[str]] = {}
    for result in result_dicts:
        if "id" not in result:
            continue
        result_namespace = str(result.get("namespace", namespace))
        grouped.setdefault(result_namespace, []).append(str(result["id"]))

    if not grouped:
        return

    with ExitStack() as stack:
        for result_namespace, ids in grouped.items():
            target_backend = backend
            if result_namespace != namespace:
                if namespace_backend_factory is None:
                    continue
                target_backend = stack.enter_context(namespace_backend_factory(result_namespace))
            record_recall_access(target_backend, ids)


def register_recall_tool(mcp: McpServer) -> None:
    """Register memory_recall with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from trw_memory.integrations._backend import create_backend_from_config

    async def memory_recall(
        query: str,
        namespace: str = "project:default",
        limit: int = 25,
        min_score: float = 0.0,
        tags: list[str] | None = None,
        include_namespaces: list[str] | None = None,
        include_org_memories: bool = True,
        graph_depth: int = 0,
        token_budget: int | None = None,
    ) -> dict[str, object]:
        """Search memory entries using hybrid BM25 + vector retrieval.

        Args:
            query: Free-text search query.
            namespace: Namespace scope (e.g., 'project:default', 'global').
            limit: Maximum results to return (default 25).
            min_score: Minimum utility score filter (0.0 = no filter).
            tags: Filter to entries containing ALL of these tags.
            include_namespaces: Additional namespaces to search alongside primary.
            include_org_memories: If True, append org-wide cross-validated
                sibling-project memories after local matches.
            graph_depth: If > 0, include graph-related entries via BFS traversal
                from the result set (max depth 3).
            token_budget: If provided, truncate results to fit within this
                token budget. Must be a positive integer. Returns metadata
                about token usage in the response.

        Returns:
            {"memories": [...], "total_matches": int, "query": str,
             "tokens_used": int, "tokens_budget": int | None,
             "tokens_truncated": bool,
             "related": [...] (when graph_depth > 0)}
        """
        cfg = MemoryConfig()
        with create_backend_from_config(cfg, namespace) as backend:
            return memory_recall_impl(
                query,
                namespace,
                backend=backend,
                namespace_backend_factory=lambda extra_ns: create_backend_from_config(cfg, extra_ns),
                limit=limit,
                min_score=min_score,
                tags=tags,
                include_namespaces=include_namespaces,
                include_org_memories=include_org_memories,
                graph_depth=graph_depth,
                token_budget=token_budget,
                config=cfg,
            )

    mcp.tool()(memory_recall)
