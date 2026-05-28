# ruff: noqa: I001
"""Recall cluster — public ``recall`` + 8 internal helpers.

Belongs to ``client.py``. Re-exported there for back-compat.

Module-level async helpers + pure-function staticmethods extracted from
``MemoryClient.recall(...)`` and its 8 supporting methods. The
``MemoryClient`` methods become thin delegators that call these
module-level helpers passing ``self`` as the ``client`` parameter
(backend-handle pattern from PRD-DIST-245 batch 87).

Public surface (all delegated from ``MemoryClient``):

- ``recall_impl`` — main async entry point.
- ``apply_recall_security`` — recall-window filter + canary probe.
- ``apply_budget`` — token-budget filtering (pure).
- ``merge_org_results`` — append cross-validated sibling memories.
- ``try_hybrid_recall`` — BM25 + dense + RRF.
- ``fallback_recall`` — LIKE + TF + importance scoring.
- ``record_recall_access`` — persist access metadata.
- ``tier_results`` — local tier candidates.
- ``remember_results_in_tiers`` — keep tiers aligned.
- ``merge_tier_results`` — fuse tier-only candidates (pure).
- ``tier_result_from_entry`` — tier-entry → result-dict (pure).

Extracted as PRD-DIST-246 batch 105.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import structlog

from trw_memory.lifecycle._recall import record_recall_access
from trw_memory.lifecycle.tiers._runtime import get_tier_manager, tier_runtime_enabled
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.retrieval.source_policy import apply_source_policy
from trw_memory.security.rbac import Permission
from trw_memory.security.recall_filter import filter_recall_window
from trw_memory.security.runtime import (
    append_audit_event,
    probe_canaries,
    should_halt_recalls,
)
from trw_memory.security.telemetry_emit import build_security_traceability, emit_security_event

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient, MemoryResultDict

logger = structlog.get_logger(__name__)


def _client_logger() -> Any:
    """Parent-module logger lookup so test patches on ``trw_memory.client.logger`` propagate."""
    from trw_memory import client as _c

    return _c.logger


# Fallback recall scoring (used when hybrid retrieval pipeline is unavailable).
_FALLBACK_TF_WEIGHT: float = 0.7
_FALLBACK_IMPORTANCE_WEIGHT: float = 0.3
_FALLBACK_TF_SCALE: float = 10.0


def _entry_to_result(entry: MemoryEntry, score: float = 0.0) -> MemoryResultDict:
    from trw_memory._client_distilled_tiering import entry_to_result as _impl

    return _impl(entry, score=score)


def _create_local_backend(config: Any, namespace: str) -> Any:
    from trw_memory.client import _create_local_backend as _impl

    return _impl(config, namespace)


async def recall_impl(
    client: MemoryClient,
    query: str,
    limit: int = 10,
    tags: list[str] | None = None,
    min_score: float = 0.0,
    *,
    include_org_memories: bool = True,
    include_shared: bool = False,
    token_budget: int | None = None,
    include_distilled: bool = True,
    distilled_weight: float | None = None,
    include_source_kinds: list[str] | None = None,
    exclude_source_kinds: list[str] | None = None,
    source_weights: dict[str, float] | None = None,
    exclude_expired: bool = True,
    confidence_floor: float | None = None,
    exclude_historical_only: bool | None = None,
) -> list[MemoryResultDict]:
    """Async impl for :meth:`MemoryClient.recall`.

    See the method docstring for full arg/return semantics.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if token_budget is not None and token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    client._require_permission(Permission.READ, "recall")
    client._maybe_start_retry_drain()
    await client._apply_pending_remote_retirements()
    backend = client._backend
    if backend is not None and should_halt_recalls(client._config, backend=backend):
        from trw_memory.exceptions import CanaryTamperError

        raise CanaryTamperError("recall halted after canary tamper")
    embedder = client._get_embedder() if query.strip() else None
    query_embedding: list[float] | None = None
    if embedder is not None:
        query_embedding = await asyncio.to_thread(embedder.embed, query)

    async with client._lock:
        backend = client._get_backend()
        if client._namespace.startswith("team:") and NamespaceManager(backend).team_namespace_expired(
            client._namespace
        ):
            logger.debug(
                "memory_recall_team_namespace_expired",
                op="recall",
                namespace=client._namespace,
            )
            return []
        if tier_runtime_enabled(client._config):
            tier_local_results = client._tier_results(backend, query, tags, limit, query_embedding)
            client._tier_manager = get_tier_manager(client._config, client._namespace)
        else:
            tier_local_results = []

    # PRD-DIST-2049 c802: resolve per-call kwargs vs config defaults (per-call wins
    # when non-None; config drives when caller omits).
    effective_confidence_floor = (
        confidence_floor if confidence_floor is not None else client._config.recall_confidence_filter
    )
    effective_exclude_historical = (
        exclude_historical_only if exclude_historical_only is not None else client._config.recall_filter_historical_only
    )

    hybrid_results = await client._try_hybrid_recall(query, limit, tags)
    if hybrid_results is not None:
        filtered = [r for r in hybrid_results if r["score"] >= min_score]
        # PRD-DIST-2049 c802: apply admission filter on the FULL hybrid candidate
        # pool (top ~limit*3 = 30) AND on the tier candidate pool BEFORE merge.
        # This lets the filter promote baseline records that would otherwise be
        # displaced past top-K by zombie / historical_only competitors — closes
        # the c800/c801 contamination lever rather than just decorating the
        # already-displaced top-K.
        filtered = apply_admission_filter(
            filtered,
            confidence_floor=effective_confidence_floor,
            exclude_historical_only=effective_exclude_historical,
            namespace=client._namespace,
        )
        filtered_tier = apply_admission_filter(
            tier_local_results,
            confidence_floor=effective_confidence_floor,
            exclude_historical_only=effective_exclude_historical,
            namespace=client._namespace,
        )
        final_pre_policy = client._merge_tier_results(
            filtered[:limit],
            filtered_tier,
            limit,
            query.lower().split(),
            client._config,
            query_embedding,
        )
        if min_score > 0.0:
            final_pre_policy = [result for result in final_pre_policy if result["score"] >= min_score]
        if include_org_memories:
            final_pre_policy = await client._merge_org_results(query, final_pre_policy, limit, tags, min_score)
        if include_shared:
            final_pre_policy = await client._merge_shared_results(query, final_pre_policy, limit)
        final_scored = cast(
            "list[MemoryResultDict]",
            apply_source_policy(
                final_pre_policy,
                include_distilled=include_distilled,
                distilled_weight=distilled_weight,
                include_source_kinds=include_source_kinds,
                exclude_source_kinds=exclude_source_kinds,
                source_weights=source_weights,
                exclude_expired=exclude_expired,
            ),
        )
        if min_score > 0.0:
            final_scored = [result for result in final_scored if result["score"] >= min_score]
        final = client._apply_recall_security(client._apply_budget(final_scored[:limit], token_budget))
        await client._record_recall_access(final)
        append_audit_event(
            client._config,
            "recall",
            actor="",
            namespace=client._namespace,
            data={"query": query[:80], "entries_returned": len(final)},
        )
        client._remember_results_in_tiers(final)
        _client_logger().debug(
            "memory_recalled",
            op="recall",
            outcome="success",
            query=query[:80],
            namespace=client._namespace,
            result_count=len(final),
            search_path="hybrid",
        )
        return final

    results = await client._fallback_recall(query, limit, tags, min_score)
    # PRD-DIST-2049 c802: apply filter on fallback candidates AND tier candidates
    # before merge, mirroring the hybrid path.
    results = apply_admission_filter(
        results,
        confidence_floor=effective_confidence_floor,
        exclude_historical_only=effective_exclude_historical,
        namespace=client._namespace,
    )
    filtered_tier_fallback = apply_admission_filter(
        tier_local_results,
        confidence_floor=effective_confidence_floor,
        exclude_historical_only=effective_exclude_historical,
        namespace=client._namespace,
    )
    results = client._merge_tier_results(
        results,
        filtered_tier_fallback,
        limit,
        query.lower().split(),
        client._config,
        query_embedding,
    )
    if min_score > 0.0:
        results = [result for result in results if result["score"] >= min_score]
    if include_org_memories:
        results = await client._merge_org_results(query, results, limit, tags, min_score)
    if include_shared:
        results = await client._merge_shared_results(query, results, limit)
    filtered_results = cast(
        "list[MemoryResultDict]",
        apply_source_policy(
            results,
            include_distilled=include_distilled,
            distilled_weight=distilled_weight,
            include_source_kinds=include_source_kinds,
            exclude_source_kinds=exclude_source_kinds,
            source_weights=source_weights,
            exclude_expired=exclude_expired,
        ),
    )
    if min_score > 0.0:
        filtered_results = [result for result in filtered_results if result["score"] >= min_score]
    final = client._apply_recall_security(client._apply_budget(filtered_results[:limit], token_budget))
    await client._record_recall_access(final)
    append_audit_event(
        client._config,
        "recall",
        actor="",
        namespace=client._namespace,
        data={"query": query[:80], "entries_returned": len(final)},
    )
    client._remember_results_in_tiers(final)
    return final


def apply_recall_security(
    client: MemoryClient,
    results: list[MemoryResultDict],
) -> list[MemoryResultDict]:
    if client._backend is not None:
        probe_canaries(client._config, backend=client._backend)
    if not client._config.enable_recall_filter:
        return results
    score_by_id: dict[str, float] = {}
    result_by_id: dict[str, MemoryResultDict] = {}
    entries: list[MemoryEntry] = []
    for idx, result in enumerate(results):
        synthetic_id = f"{result['namespace']}::{result['memory_id']}::{idx}"
        score_by_id[synthetic_id] = result["score"]
        result_by_id[synthetic_id] = result
        raw_result = {"id": synthetic_id, **result}
        recalled_at = datetime.now(timezone.utc)
        for timestamp_field in ("created_at", "updated_at"):
            if raw_result.get(timestamp_field) in {"", "None", None}:
                raw_result[timestamp_field] = recalled_at
        if raw_result.get("last_accessed_at") in {"", "None", None}:
            raw_result["last_accessed_at"] = None
        entries.append(MemoryEntry.model_validate(raw_result))
    filtered = filter_recall_window(entries, mode=client._config.recall_filter_mode)
    session_id = os.environ.get("TRW_SESSION_ID", "").strip() or client._namespace
    run_id = os.environ.get("TRW_RUN_ID", "").strip() or None
    emit_security_event(
        client._config,
        emitter="recall_filter",
        session_id=session_id,
        run_id=run_id,
        payload={
            "event_name": "recall_filter_outcome",
            "path": "client_recall",
            "namespace": client._namespace,
            "mode": client._config.recall_filter_mode,
            "window_size": len(entries),
            "accepted_count": len(filtered.accepted),
            "would_reject_count": len(filtered.would_reject),
            "actions": dict(filtered.actions),
            "traceability": build_security_traceability(
                live_path="client.MemoryClient._apply_recall_security",
                requirement_ids=["FR-003", "NFR-010", "NFR-011"],
            ),
        },
    )
    secured: list[MemoryResultDict] = []
    for entry in filtered.accepted:
        if entry.metadata.get("system_canary") == "true":
            continue
        original = dict(result_by_id[entry.id])
        original["content"] = entry.content
        original["detail"] = entry.detail
        original["metadata"] = dict(entry.metadata)
        original["score"] = score_by_id[entry.id]
        secured.append(cast("MemoryResultDict", original))
    return secured


# `apply_budget` and `merge_org_results` extracted to
# _client_recall_helpers.py (PRD-DIST-246 batch 105 sub-split).
from trw_memory._client_recall_helpers import (  # noqa: E402
    apply_admission_filter as apply_admission_filter,
    apply_budget as apply_budget,
    merge_org_results as merge_org_results,
)


async def try_hybrid_recall(
    client: MemoryClient,
    query: str,
    limit: int,
    tags: list[str] | None,
) -> list[MemoryResultDict] | None:
    """Hybrid pipeline (BM25 + dense + RRF). Returns None to signal fallback.

    PRD-DIST-2047 Phase 2 (recall-latency telemetry): emits a structlog event
    ``hybrid_recall_complete`` carrying per-call timings + namespace shape +
    effective candidate caps + returned-result count, so operators can right-
    size ``hybrid_search_candidate_pool_size`` against measured cost. The
    event fires on every terminating exit (success, no-candidates, hybrid-
    search-failed) so operators can attribute latency to outcome.
    """
    try:
        from trw_memory.retrieval.pipeline import hybrid_search
    except ImportError:
        return None

    total_start = perf_counter()

    async with client._lock:
        backend = client._get_backend()
        # PRD-DIST-2047 c796: load up to hybrid_search_candidate_pool_size
        # entries (default 1000) so BM25 + dense can rank the full namespace.
        # Pre-c796 the pool was capped at limit*5 (=50 for default limit=10),
        # which silently lost targets ranked past position 50 on namespaces > 50.
        candidate_pool_size = max(limit * 5, client._config.hybrid_search_candidate_pool_size)
        list_entries_start = perf_counter()
        all_entries = backend.list_entries(
            namespace=client._namespace,
            limit=candidate_pool_size,
        )
        list_entries_ms = (perf_counter() - list_entries_start) * 1000.0
        stored_embeddings = backend.get_stored_embeddings([entry.id for entry in all_entries])

    namespace_size = len(all_entries)
    if not all_entries:
        _emit_hybrid_recall_telemetry(
            outcome="no_candidates",
            namespace=client._namespace,
            namespace_size=namespace_size,
            candidate_pool_size=candidate_pool_size,
            effective_bm25_candidates=0,
            effective_vector_candidates=0,
            effective_top_k=limit * client._config.recall_top_k_multiplier,
            returned_count=0,
            list_entries_ms=list_entries_ms,
            hybrid_search_ms=0.0,
            total_ms=(perf_counter() - total_start) * 1000.0,
        )
        return None

    embedder = client._get_embedder()
    # PRD-DIST-2047 c796: auto-scale bm25/vector candidate caps to namespace
    # size so the 50-default acts as a FLOOR, not a CEILING. Eliminates the
    # structural cap on recall@10 for namespaces > 50 records.
    effective_bm25_candidates = max(client._config.bm25_candidates, namespace_size)
    effective_vector_candidates = max(client._config.vector_candidates, namespace_size)

    # PRD-DIST-2050 c804: deepen the candidate pool when the admission filter
    # is opt-in enabled, so baseline records ranked past top-30 can survive the
    # filter and enter the merged top-K. Default multiplier=3 preserves pre-c804
    # behaviour (top-30); operators raise via MEMORY_RECALL_TOP_K_MULTIPLIER.
    effective_top_k = limit * client._config.recall_top_k_multiplier
    hybrid_search_start = perf_counter()
    try:
        ranked = hybrid_search(
            query=query,
            entries=all_entries,
            embedder=embedder,
            stored_embeddings=stored_embeddings or None,
            bm25_candidates=effective_bm25_candidates,
            vector_candidates=effective_vector_candidates,
            top_k=effective_top_k,
        )
    except Exception:
        hybrid_search_ms = (perf_counter() - hybrid_search_start) * 1000.0
        logger.debug(
            "hybrid_search_failed",
            op="recall",
            outcome="failure",
            exc_info=True,
        )
        _emit_hybrid_recall_telemetry(
            outcome="hybrid_search_failed",
            namespace=client._namespace,
            namespace_size=namespace_size,
            candidate_pool_size=candidate_pool_size,
            effective_bm25_candidates=effective_bm25_candidates,
            effective_vector_candidates=effective_vector_candidates,
            effective_top_k=effective_top_k,
            returned_count=0,
            list_entries_ms=list_entries_ms,
            hybrid_search_ms=hybrid_search_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
        )
        return None
    hybrid_search_ms = (perf_counter() - hybrid_search_start) * 1000.0

    if not ranked:
        _emit_hybrid_recall_telemetry(
            outcome="empty_ranking",
            namespace=client._namespace,
            namespace_size=namespace_size,
            candidate_pool_size=candidate_pool_size,
            effective_bm25_candidates=effective_bm25_candidates,
            effective_vector_candidates=effective_vector_candidates,
            effective_top_k=effective_top_k,
            returned_count=0,
            list_entries_ms=list_entries_ms,
            hybrid_search_ms=hybrid_search_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
        )
        return None

    if tags:
        tag_set = set(tags)
        ranked = [e for e in ranked if tag_set.issubset(set(e.tags))]

    results: list[MemoryResultDict] = []
    for rank, entry in enumerate(ranked):
        score = round(1.0 / (1 + rank), 4)
        results.append(_entry_to_result(entry, score=score))

    _emit_hybrid_recall_telemetry(
        outcome="ok",
        namespace=client._namespace,
        namespace_size=namespace_size,
        candidate_pool_size=candidate_pool_size,
        effective_bm25_candidates=effective_bm25_candidates,
        effective_vector_candidates=effective_vector_candidates,
        effective_top_k=effective_top_k,
        returned_count=len(results),
        list_entries_ms=list_entries_ms,
        hybrid_search_ms=hybrid_search_ms,
        total_ms=(perf_counter() - total_start) * 1000.0,
    )
    return results


def _emit_hybrid_recall_telemetry(
    *,
    outcome: str,
    namespace: str,
    namespace_size: int,
    candidate_pool_size: int,
    effective_bm25_candidates: int,
    effective_vector_candidates: int,
    effective_top_k: int,
    returned_count: int,
    list_entries_ms: float,
    hybrid_search_ms: float,
    total_ms: float,
) -> None:
    """PRD-DIST-2047 Phase 2: emit a per-recall latency + shape event.

    Operators sample this event stream to right-size
    ``hybrid_search_candidate_pool_size`` for very large namespaces (where
    BM25 cost grows linearly with namespace_size). Latencies are reported in
    milliseconds rounded to 3 decimals.
    """
    logger.info(
        "hybrid_recall_complete",
        op="recall",
        outcome=outcome,
        namespace=namespace,
        namespace_size=namespace_size,
        candidate_pool_size=candidate_pool_size,
        effective_bm25_candidates=effective_bm25_candidates,
        effective_vector_candidates=effective_vector_candidates,
        effective_top_k=effective_top_k,
        returned_count=returned_count,
        list_entries_ms=round(list_entries_ms, 3),
        hybrid_search_ms=round(hybrid_search_ms, 3),
        total_ms=round(total_ms, 3),
    )


async def fallback_recall(
    client: MemoryClient,
    query: str,
    limit: int,
    tags: list[str] | None,
    min_score: float,
) -> list[MemoryResultDict]:
    """LIKE + TF + importance scoring. Used when hybrid pipeline is unavailable."""
    async with client._lock:
        entries = client._get_backend().search(
            query,
            top_k=limit * 3,
            tags=tags,
            namespace=client._namespace,
        )

    query_terms = set(query.lower().split())
    results: list[MemoryResultDict] = []
    for entry in entries:
        if not query_terms:
            tf_score = entry.importance
        else:
            text_tokens = f"{entry.content} {entry.detail} {' '.join(entry.tags)}".lower().split()
            matches = sum(1 for t in text_tokens if t in query_terms)
            tf_score = (
                min(1.0, matches / max(len(text_tokens), 1) * _FALLBACK_TF_SCALE) * _FALLBACK_TF_WEIGHT
                + entry.importance * _FALLBACK_IMPORTANCE_WEIGHT
            )
        if tf_score >= min_score:
            results.append(_entry_to_result(entry, score=round(tf_score, 4)))

    results.sort(key=lambda r: float(r["score"]), reverse=True)
    final = results[:limit]
    _client_logger().debug(
        "memory_recalled",
        op="recall",
        outcome="success",
        query=query[:80],
        namespace=client._namespace,
        result_count=len(final),
        search_path="fallback",
    )
    return final


async def record_recall_access_impl(
    client: MemoryClient,
    results: list[MemoryResultDict],
) -> None:
    """Persist access metadata for the entries that were actually returned."""
    grouped: dict[str, list[str]] = {}
    for result in results:
        if result.get("source") == "shared":
            continue
        grouped.setdefault(result["namespace"], []).append(result["memory_id"])

    if not grouped:
        return

    async with client._lock:
        for namespace, entry_ids in grouped.items():
            if namespace == client._namespace:
                record_recall_access(client._get_backend(), entry_ids)
            else:
                with _create_local_backend(client._config, namespace) as backend:
                    record_recall_access(backend, entry_ids)
            append_audit_event(
                client._config,
                "access",
                actor="",
                namespace=namespace,
                data={"entry_ids": entry_ids, "entries_accessed": len(entry_ids)},
            )


# Tier helpers extracted to _client_recall_helpers.py (PRD-DIST-246
# batch 105 sub-split). Re-exports preserve the call-sites in
# `recall_impl` above.
from trw_memory._client_recall_helpers import (  # noqa: E402
    merge_tier_results as merge_tier_results,
    remember_results_in_tiers as remember_results_in_tiers,
    tier_result_from_entry as tier_result_from_entry,
    tier_results as tier_results,
)
