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
- ``try_hybrid_recall`` — BM25 + dense + RRF (re-exported from
  ``_client_recall_hybrid``).
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
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import structlog

from trw_memory._client_distilled_tiering import entry_to_result as _entry_to_result
from trw_memory.lifecycle._recall import record_recall_access
from trw_memory.lifecycle.tiers._runtime import get_tier_manager, tier_runtime_enabled
from trw_memory.models.memory import MemoryStatus
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.retrieval.source_policy import apply_source_policy
from trw_memory.security.rbac import Permission
from trw_memory.security.runtime import (
    append_audit_event,
    should_halt_recalls,
)

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
    as_of: datetime | None = None,
    include_superseded: bool = False,
    include_graph_expansion: bool = False,
    query_expansion: str | None = None,
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
        exp_text: str | None = query_expansion if query_expansion and query_expansion.strip() else None
        if exp_text is not None:
            # HyDE multi-vector: embed both query and hypothetical expansion,
            # then average. Averaging captures query specificity (BM25 intent)
            # AND expansion semantics (hypothetical-answer space). This
            # consistently outperforms using either embedding alone.
            raw_vec = await asyncio.to_thread(embedder.embed, query)
            exp_vec = await asyncio.to_thread(embedder.embed, exp_text)
            if raw_vec is not None and exp_vec is not None:
                dim = len(raw_vec)
                query_embedding = [0.5 * (raw_vec[i] + exp_vec[i]) for i in range(dim)]
            else:
                query_embedding = raw_vec if raw_vec is not None else exp_vec
        else:
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

    hybrid_results = await client._try_hybrid_recall(
        query,
        limit,
        tags,
        query_embedding=query_embedding,
        as_of=as_of,
        include_superseded=include_superseded,
    )
    if hybrid_results is not None and include_graph_expansion:
        from trw_memory._client_recall_graph import graph_expand_results

        async with client._lock:
            hybrid_results = graph_expand_results(client, hybrid_results)
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
        final = await _finalize_recall(client, final_scored, query=query, limit=limit, token_budget=token_budget)
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
    if include_graph_expansion:
        from trw_memory._client_recall_graph import graph_expand_results

        async with client._lock:
            results = graph_expand_results(client, results)
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
    return await _finalize_recall(client, filtered_results, query=query, limit=limit, token_budget=token_budget)


async def _finalize_recall(
    client: MemoryClient,
    results: list[MemoryResultDict],
    *,
    query: str,
    limit: int,
    token_budget: int | None,
) -> list[MemoryResultDict]:
    """Apply final security, accounting, audit, and tier side effects in order."""
    from trw_memory._client_recall_graph import filter_conflicting_results

    async with client._lock:
        results = filter_conflicting_results(client, results)
    final = client._apply_recall_security(client._apply_budget(results[:limit], token_budget))
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


from trw_memory._client_recall_security import (  # noqa: E402
    apply_recall_security as apply_recall_security,
)


# `apply_budget` and `merge_org_results` extracted to
# _client_recall_helpers.py (PRD-DIST-246 batch 105 sub-split).
from trw_memory._client_recall_helpers import (  # noqa: E402
    apply_admission_filter as apply_admission_filter,
    apply_budget as apply_budget,
    merge_org_results as merge_org_results,
)


# Hybrid recall pipeline (`try_hybrid_recall` + its private telemetry helper
# `_emit_hybrid_recall_telemetry`) extracted to `_client_recall_hybrid.py` as a
# cohesive deep module (loc-tracker self-improve split). Re-export preserves the
# `MemoryClient._try_hybrid_recall` delegator's import path.
from trw_memory._client_recall_hybrid import (  # noqa: E402
    try_hybrid_recall as try_hybrid_recall,
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
            status=MemoryStatus.ACTIVE,
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
