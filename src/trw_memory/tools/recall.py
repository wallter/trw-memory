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

from trw_memory.daemon._offload import run_offloaded
from trw_memory.embeddings import get_local_embedder
from trw_memory.embeddings._space_gate import active_embedding_space, admit_space_vectors
from trw_memory.embeddings.provenance import StoredVector
from trw_memory.exceptions import ConfigError
from trw_memory.lifecycle._recall import drop_expired_entries, rank_by_utility
from trw_memory.lifecycle.tiers._runtime import remember_entries_data_in_tiers, supports_tier_runtime, tier_candidates
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.namespaces.validation import validate_namespace

# Re-exported, not merely imported: ``_recall_retrieval`` looks this name up on
# THIS module so a test patching ``trw_memory.tools.recall.hybrid_search_scored``
# still reaches the call, and so the tool path's retrieval boundary stays
# readable from its facade. The ``as`` form is what keeps ruff from removing an
# import with no local reference.
from trw_memory.retrieval import hybrid_search_scored as hybrid_search_scored
from trw_memory.retrieval.admission_policy import apply_admission_filter
from trw_memory.retrieval.lexical import tokenize_query
from trw_memory.retrieval.recall_policy import RECALL_PREFETCH_MULTIPLIER, acquire_candidates
from trw_memory.retrieval.source_policy import SourcePolicy
from trw_memory.security.namespace_scope import authorize_namespaces
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
from trw_memory.tools._recall_retrieval import build_scored_candidates
from trw_memory.tools._types import McpServer

logger = structlog.get_logger(__name__)


def _rescale_supplementary_scores(
    rows: list[dict[str, object]],
    retrieval_keys: set[tuple[str, str]],
    namespace: str,
) -> None:
    """Place rows retrieval never scored BELOW every row it did (FR03).

    Mutates in place. Supplementary rows keep their relative order and are
    spread across the open interval ``(0, weakest_retrieval_score)``; when there
    are no retrieval rows at all — a wildcard recall, or a query that matched
    nothing locally — they keep the score they arrived with, because there is no
    retrieval ranking for them to stay under.
    """
    supplements = [
        row for row in rows if (str(row.get("namespace", namespace)), str(row.get("id", ""))) not in retrieval_keys
    ]
    if not supplements or len(supplements) == len(rows):
        return
    retrieval_scores = [
        float(str(row.get("score", 0.0)))
        for row in rows
        if (str(row.get("namespace", namespace)), str(row.get("id", ""))) in retrieval_keys
    ]
    floor = min(retrieval_scores) if retrieval_scores else 0.0
    if floor <= 0.0:
        for row in supplements:
            row["score"] = 0.0
        return
    step = floor / (len(supplements) + 1)
    for position, row in enumerate(supplements, start=1):
        row["score"] = round(floor - step * position, 9)


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
    include_source_kinds: list[str] | None = None,
    exclude_source_kinds: list[str] | None = None,
    exclude_expired: bool = True,
    status: str | None = "active",
    record_access: bool = True,
) -> dict[str, object]:
    """Core implementation of memory_recall (callable without MCP).

    Args:
        query: Free-text search query. Empty string returns every entry in *status*.
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
        status: Lifecycle status searched; ``None`` searches every status.

    Returns:
        {"memories": list[dict], "total_matches": int, "query": str,
         "tokens_used": int, "tokens_budget": int | None,
         "tokens_truncated": bool,
         "related": list[dict] (when graph_depth > 0),
         "partial": True and "namespaces_omitted": {"denied": int,
         "expired": int} when one of the requested namespaces was refused by
         the authorizer or skipped as an expired team namespace -- present ONLY
         when something was omitted, so a complete answer keeps its shape}
        or {"error": str, "status": "invalid"} on validation failure.

    Raises:
        ValueError: If *token_budget* is not ``None`` and <= 0.
    """
    if token_budget is not None and token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")

    try:
        validate_namespace(namespace)
        wanted_status = MemoryStatus(status) if status is not None else None
    except (ConfigError, ValueError) as exc:
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

    # PRD-CORE-245 FR05: the permission result IS the scope. Previously the
    # check above ran and its outcome was discarded by the time the loop below
    # opened backends; now the scope decides which namespaces may be opened at
    # all, so a namespace that fails the check is never read from disk rather
    # than read and later filtered.
    scope = authorize_namespaces(cfg, [namespace, *(include_namespaces or [])], Permission.READ, "recall")
    all_namespaces = [ns for ns in [namespace, *(include_namespaces or [])] if ns in scope]
    # A namespace dropped here (refused, or expired below) narrows the corpus the
    # answer was computed over. The counts ride out on the response so the caller
    # reads a narrowed result as narrowed instead of as a complete one.
    expired_skipped = 0
    # Rank to trw_recall's depth, then cap to ``limit`` last, so a row admission
    # or the recall filter drops is refilled from the ranked tail.
    depth = limit * RECALL_PREFETCH_MULTIPLIER
    # Namespace-scoped local backends can only see one store at a time, so
    # cross-namespace recall must reopen the requested namespaces explicitly.
    all_entries = []
    vector_records: list[tuple[str, dict[str, StoredVector]]] = []
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
                expired_skipped += 1
                # Drop it from the scope too, so an expired team namespace
                # cannot satisfy the pipeline's membership assertion via some
                # other path (FR05).
                scope = scope.without(ns)
                continue

            # The acquisition every recall surface shares (PRD-CORE-298 FR05): the
            # recency pool, with the tag predicate in SQL so the LIMIT applies
            # after it, plus the rows full-text search finds past the pool.
            ns_entries = acquire_candidates(
                ns_backend, query, namespace=ns, limit=depth, config=cfg, status=wanted_status, tags=tags or None
            ).entries
            all_entries.extend(ns_entries)

            if query and ns_entries:
                vector_records.append(
                    (ns, ns_backend.get_vector_records([entry.id for entry in ns_entries], namespace=ns))
                )

    # PRD-CORE-278 FR07: resolve the embedder whenever there is a query, even
    # when the namespace holds nothing to rank. A readiness probe that recalls in
    # a reserved, empty namespace used to return instantly and leave the model
    # cold, so the first real call paid the 5s load (sub_6PZlZpuFaO90dcFq). An
    # empty query still resolves nothing: there is nothing to embed.
    embedder = get_local_embedder(model_name=cfg.embedding_model, dim=cfg.embedding_dim) if query else None
    # Dense-score only vectors from the active embedder's space, reported once
    # per namespace; excluded rows stay in the pool for BM25.
    stored_embeddings: dict[str, list[float]] = {}
    active_space = active_embedding_space(embedder)
    for ns, records in vector_records if embedder is not None else ():
        stored_embeddings.update(admit_space_vectors(records, active_space, namespace=ns, surface="memory_recall_tool"))

    entry_dicts, query_embedding = build_scored_candidates(
        query,
        all_entries,
        cfg=cfg,
        scope=scope,
        embedder=embedder,
        stored_embeddings=stored_embeddings,
        limit=depth,
        tags=tags,
    )

    # A query keeps the pipeline's order, as trw_recall does (PRD-CORE-298 FR05):
    # re-sorting by a normalised score would tie every non-positive reranker score
    # and let utility reorder them. A wildcard has no retrieval order, so it is
    # ordered by utility.
    query_tokens = tokenize_query(query) if query else []
    ranked_dicts = entry_dicts if query else rank_by_utility(entry_dicts, query_tokens, config=cfg)
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
            query_space=active_space,
        )
    retrieval_keys = {(str(row.get("namespace", namespace)), str(row.get("id", ""))) for row in ranked_dicts}
    if tier_dicts:
        ranked_dicts = _merge_tier_entries(ranked_dicts, tier_dicts, query_tokens, cfg, query_embedding)

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
                open_backend=backend,
            )
        )
    # PRD-CORE-278 FR03: a supplementary row — tier-only or org — was never
    # scored by retrieval, and its own utility score lives on a different, much
    # larger scale (importance ~0.8 against a fused RRF score ~0.16). Handing
    # both to the source policy, which sorts by score, let a supplement overtake
    # every retrieval hit; that mixed-scale comparison is the defect this PRD
    # exists to end. Supplements are rescaled INTO the interval below the
    # weakest retrieval score, keeping their own relative order, so one number
    # on one scale explains the whole result and ``min_score`` filters the same
    # number the response reports.
    _rescale_supplementary_scores(result_dicts, retrieval_keys, namespace)
    # Source admission only, as trw_recall applies it (PRD-CORE-298 FR05): the
    # order stays the pipeline's, with supplements after it. Re-sorting by source
    # weight here made mixed-source results diverge from trw_recall's order.
    admission = SourcePolicy.resolve(
        include_distilled=include_distilled,
        include_source_kinds=include_source_kinds,
        exclude_source_kinds=exclude_source_kinds,
        exclude_expired=exclude_expired,
    )
    result_dicts = [row for row in result_dicts if admission.allows(row)]
    # ``min_score`` is applied ONCE, here, on the score the response reports.
    if min_score > 0.0:
        result_dicts = [row for row in result_dicts if float(str(row.get("score", 0.0))) >= min_score]

    # SEC-001 recall filter runs on the FULL ranked candidate set BEFORE the
    # token-budget fitting and limit cap (trw-memory-3 / trw-memory-8). Running
    # it after the limit cap silently under-delivered: a caller requesting
    # `limit` entries received `limit - filtered` while clean entries ranked
    # beyond the cap were never considered, and `tokens_used` over-reported by
    # counting entries the filter later dropped. Filtering first lets the
    # budget + cap operate on the secured set, so the caller gets up to `limit`
    # admitted entries and `tokens_used` matches what is actually returned.
    result_dicts = _apply_sec001_recall_policy(result_dicts, config=cfg, namespace=namespace)

    # PRD-CORE-278 FR05: the ONE expiry pass, on the MERGED set. It runs here and
    # not inside the ranker because the tier and org merges add rows AFTER
    # ranking, and an entry the ranker dropped came straight back. Expiry is
    # ineligibility, not a caller preference: ``exclude_expired=False`` does not
    # re-admit an expired record (the validity prior already excluded it inside
    # retrieval), so this pass closes the merge hole rather than offering a
    # second opinion.
    result_dicts = drop_expired_entries(result_dicts)

    tokens_truncated = False

    if token_budget is not None and result_dicts:
        result_dicts, _, tokens_truncated = apply_token_budget(result_dicts, token_budget)

    # Apply limit cap AFTER token budget
    result_dicts = result_dicts[:limit]
    tokens_used = sum(estimate_entry_tokens(d) for d in result_dicts)
    if record_access:
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
        remember_entries_data_in_tiers(cfg, [{**item, "last_accessed_at": recalled_at} for item in result_dicts])

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
    if scope.denied or expired_skipped:
        # Counts only, never the refused names: a namespace label is
        # operator-chosen but still user data (PRD-CORE-245 NFR03).
        logger.warning(
            "memory_recall_scope_narrowed",
            namespace=namespace,
            denied=scope.denied,
            expired=expired_skipped,
            searched=len(seen_namespaces) - expired_skipped,
        )
        response["partial"] = True
        response["namespaces_omitted"] = {"denied": scope.denied, "expired": expired_skipped}

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
        include_source_kinds: list[str] | None = None,
        exclude_source_kinds: list[str] | None = None,
        exclude_expired: bool = True,
        status: str | None = "active",
        record_access: bool = True,
    ) -> dict[str, object]:
        """Search memory entries using hybrid BM25 + vector retrieval.

        Search population is BOUNDED, not exhaustive: the store scan loads at
        most max(limit * 25, hybrid_search_candidate_pool_size) entries per
        namespace (default 1000), selected as the most recently updated rows,
        plus the active rows a full-text search for the query finds beyond them
        (at most max(bm25_candidates * 2, 100)). When the tier runtime is live (the default; off under encryption) its
        hot/warm/cold index is searched as well, and it holds every entry
        written through it plus, for a store that predates it, the most
        recently updated max(hot_max_entries * 8, 200) rows seeded at first
        warmup. An entry older than the store-scan bound, missed by full-text
        search and absent from the tier index is not searched, so an empty result
        does not mean the fact is absent from the store. Raise
        MEMORY_HYBRID_SEARCH_CANDIDATE_POOL_SIZE to widen the scan; measured
        cost on a 6500-row namespace was 139.6 ms at 1000 versus 1045.8 ms at
        10000.

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
            include_source_kinds: Optional allowlist of source families.
            exclude_source_kinds: Optional denylist of source families.
            exclude_expired: When True, expired transient results are removed.
            status: Lifecycle status to search (default 'active'; null: any).
            record_access: Count the returned rows as accessed. A caller that filters
                the page before showing it passes False and reports what it showed
                through ``memory_record_surfaced``.

        Returns:
            {"memories": [...], "total_matches": int, "query": str,
             "tokens_used": int, "tokens_budget": int | None,
             "tokens_truncated": bool,
             "related": [...] (when graph_depth > 0),
             "partial": true + "namespaces_omitted" counts when a requested
             namespace was refused or had expired}
        """

        def _run() -> dict[str, object]:
            # PRD-CORE-279 FR04: the whole synchronous body -- backend open,
            # search, backend close -- runs in ONE worker thread, so the SQLite
            # connection never crosses a thread boundary.
            cfg = MemoryConfig()
            # PRD-CORE-298 FR05: the recall path verifies each store once per process.
            with create_backend_from_config(cfg, namespace, check_integrity_once=True) as backend:
                return memory_recall_impl(
                    query,
                    namespace,
                    backend=backend,
                    namespace_backend_factory=lambda extra_ns: create_backend_from_config(
                        cfg, extra_ns, check_integrity_once=True
                    ),
                    limit=limit,
                    min_score=min_score,
                    tags=tags,
                    include_namespaces=include_namespaces,
                    include_org_memories=include_org_memories,
                    graph_depth=graph_depth,
                    token_budget=token_budget,
                    config=cfg,
                    include_distilled=include_distilled,
                    include_source_kinds=include_source_kinds,
                    exclude_source_kinds=exclude_source_kinds,
                    exclude_expired=exclude_expired,
                    status=status,
                    record_access=record_access,
                )

        return await run_offloaded(_run)

    mcp.tool()(memory_recall)
