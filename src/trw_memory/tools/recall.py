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
from trw_memory.lifecycle.scoring import entry_utility, rank_by_utility
from trw_memory.lifecycle.tiers._runtime import remember_entry_data_in_tiers, supports_tier_runtime, tier_candidates
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.retrieval import hybrid_search
from trw_memory.retrieval.admission_policy import apply_admission_filter
from trw_memory.retrieval.source_policy import apply_source_policy
from trw_memory.security.rbac import Permission, require_namespace_permission
from trw_memory.security.runtime import append_audit_event, initialize_canaries, probe_canaries, should_halt_recalls
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools._recall_helpers import (
    _apply_sec001_recall_policy,
    _graph_related,
    _merge_tier_entries,
    _org_memory_results,
    _record_access_by_namespace,
)
from trw_memory.tools._types import McpServer

logger = structlog.get_logger(__name__)


def memory_recall_impl(
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
    include_distilled: bool = True,
    distilled_weight: float | None = None,
    include_source_kinds: list[str] | None = None,
    exclude_source_kinds: list[str] | None = None,
    source_weights: dict[str, float] | None = None,
    exclude_expired: bool = True,
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
    initialize_canaries(cfg, backend=backend)
    if should_halt_recalls(cfg, backend=backend):
        from trw_memory.exceptions import CanaryTamperError

        raise CanaryTamperError("recall halted after canary tamper")
    probe_canaries(cfg, backend=backend)

    if namespace.startswith("team:") and NamespaceManager(backend).team_namespace_expired(namespace):
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
        except ConfigError:  # per-item error handling: skip invalid namespaces, continue with valid ones
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

            if ns.startswith("team:") and NamespaceManager(ns_backend).team_namespace_expired(ns):
                logger.debug("recall_expired_namespace_skipped", namespace=ns)
                continue

            # Push the tag predicate into SQL so the row LIMIT applies AFTER the
            # tag filter (the in-memory tag filter must not run over a truncated
            # set, which would silently drop older tagged entries beyond the cap).
            # Honor the configured candidate-pool knob instead of a hardcoded
            # 10_000 (trw-memory-14) and mirror the SDK recall path
            # (_client_recall_hybrid) exactly: max(limit * 5,
            # hybrid_search_candidate_pool_size) so the two recall surfaces
            # converge and operators can bound recall memory/latency via
            # MEMORY_HYBRID_SEARCH_CANDIDATE_POOL_SIZE.
            candidate_pool_size = max(limit * 5, cfg.hybrid_search_candidate_pool_size)
            ns_entries = ns_backend.list_entries(
                status=MemoryStatus.ACTIVE,
                namespace=ns,
                limit=candidate_pool_size,
                tags=tags or None,
            )
            all_entries.extend(ns_entries)

            if query and ns_entries:
                stored_embeddings.update(ns_backend.get_stored_embeddings([entry.id for entry in ns_entries]))

    # Retrieve via hybrid search (gracefully degrades to BM25-only or empty)
    if query and all_entries:
        embedder = get_local_embedder(model_name=cfg.embedding_model, dim=cfg.embedding_dim)
        namespace_size = len(all_entries)
        effective_bm25_candidates = max(cfg.bm25_candidates, namespace_size)
        effective_vector_candidates = max(cfg.vector_candidates, namespace_size)
        effective_top_k = limit * cfg.recall_top_k_multiplier
        if tags:
            effective_top_k = max(effective_top_k, namespace_size)

        from trw_memory.retrieval.temporal_query import prepare_temporal_query

        rewrite = prepare_temporal_query(
            query,
            current_recency_weight=cfg.recall_recency_weight,
            auto_temporal=cfg.recall_auto_temporal,
            strip_prefix=cfg.recall_strip_temporal_prefix,
        )
        retrieval_query = rewrite.retrieval_query
        effective_recency_weight = rewrite.recency_weight
        temporal = rewrite.classification
        if temporal is not None and temporal.is_temporal:
            logger.debug(
                "temporal_query_detected",
                query=query[:80],
                retrieval_query=retrieval_query[:80],
                confidence=temporal.confidence,
                recency_weight=effective_recency_weight,
                patterns=temporal.matched_patterns,
                prefix_stripped=rewrite.prefix_stripped,
                surface="memory_recall_tool",
            )
        query_embedding = embedder.embed(retrieval_query) if embedder is not None else None
        # dense_search() needs the stored vector map, not just the entry IDs, so
        # tool recall must hydrate the embeddings before calling hybrid_search().
        # Forward the stripped-query embedding so dense search uses the same
        # search text as BM25 and rerank.  Tier scoring below receives the same
        # embedding, keeping the tool path internally consistent.
        ranked = hybrid_search(
            query=retrieval_query,
            entries=all_entries,
            embedder=embedder,
            query_embedding=query_embedding,
            stored_embeddings=stored_embeddings or None,
            bm25_candidates=effective_bm25_candidates,
            vector_candidates=effective_vector_candidates,
            rrf_k=cfg.rrf_k,
            importance_alpha=cfg.rrf_importance_alpha,
            top_k=effective_top_k,
            recency_weight=effective_recency_weight,
            recency_halflife_days=cfg.recall_recency_halflife_days,
            fusion_mode=cfg.recall_fusion_mode,
            validity_age_decay=cfg.recall_validity_age_decay,
            rerank=cfg.recall_rerank,
            rerank_model=cfg.recall_rerank_model,
            rerank_candidates=cfg.recall_rerank_candidates,
            # rerank_query omitted intentionally: when temporal boilerplate is
            # stripped, the cross-encoder inherits retrieval_query so it scores
            # against the same topical text as BM25 and dense search.
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

    # Recall-policy parity (PRD-DIST-2049 recall-policy seam unification): apply
    # the SAME confidence / currentness admission filter the SDK recall path
    # (MemoryClient.recall) enforces, instead of silently bypassing it on the
    # tool surface. Mirrors the SDK ordering — admission filter runs on the
    # local candidate pool BEFORE the org-memory merge, so org results stay
    # additive. Default config (recall_confidence_filter=None /
    # recall_filter_historical_only=False) returns the list unchanged,
    # preserving prior tool-path behavior bit-for-bit.
    ranked_dicts = apply_admission_filter(
        ranked_dicts,
        confidence_floor=cfg.recall_confidence_filter,
        exclude_historical_only=cfg.recall_filter_historical_only,
        namespace=namespace,
    )

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
    result_dicts = apply_source_policy(
        result_dicts,
        include_distilled=include_distilled,
        distilled_weight=distilled_weight,
        include_source_kinds=include_source_kinds,
        exclude_source_kinds=exclude_source_kinds,
        source_weights=source_weights,
        exclude_expired=exclude_expired,
    )
    if min_score > 0.0:
        result_dicts = [d for d in result_dicts if float(str(d.get("score", entry_utility(d)))) >= min_score]

    # SEC-001 recall filter runs on the FULL ranked candidate set BEFORE the
    # token-budget fitting and limit cap (trw-memory-3 / trw-memory-8). Running
    # it after the limit cap silently under-delivered: a caller requesting
    # `limit` entries received `limit - filtered` while clean entries ranked
    # beyond the cap were never considered, and `tokens_used` over-reported by
    # counting entries the filter later dropped. Filtering first lets the
    # budget + cap operate on the secured set, so the caller gets up to `limit`
    # admitted entries and `tokens_used` matches what is actually returned.
    result_dicts = _apply_sec001_recall_policy(result_dicts, config=cfg, namespace=namespace)

    tokens_used = 0
    tokens_truncated = False

    if token_budget is not None and result_dicts:
        result_dicts, tokens_used, tokens_truncated = apply_token_budget(result_dicts, token_budget)
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

    # SEC-001 filtering already applied above (before token budget + limit cap).
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
        related = _graph_related(result_dicts, graph_depth, backend, conn, namespace=namespace)
        response["related"] = related

    return response


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
        include_distilled: bool = True,
        distilled_weight: float | None = None,
        include_source_kinds: list[str] | None = None,
        exclude_source_kinds: list[str] | None = None,
        source_weights: dict[str, float] | None = None,
        exclude_expired: bool = True,
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
            include_distilled: Include git-distilled records when True.
            distilled_weight: Optional git-distilled score weight override.
            include_source_kinds: Optional allowlist of source families.
            exclude_source_kinds: Optional denylist of source families.
            source_weights: Optional per-source score weights.
            exclude_expired: When True, expired transient results are removed.

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
                include_distilled=include_distilled,
                distilled_weight=distilled_weight,
                include_source_kinds=include_source_kinds,
                exclude_source_kinds=exclude_source_kinds,
                source_weights=source_weights,
                exclude_expired=exclude_expired,
            )

    mcp.tool()(memory_recall)
