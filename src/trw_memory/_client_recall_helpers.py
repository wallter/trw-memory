"""Recall-helper sub-cluster — tier ops + budget + org-merge.

Belongs to ``client.py`` recall pipeline. Re-exported via
``_client_recall.py``. Split out from the parent recall module so each
file stays under the 350 effective-LOC gate (PRD-DIST-246 batch 105).

7 helpers:

- ``apply_budget`` — pure token-budget filtering.
- ``merge_org_results`` — append cross-validated sibling memories.
- ``tier_results`` — collect local tier-managed candidates.
- ``remember_results_in_tiers`` — keep hot/warm tiers aligned.
- ``merge_tier_results`` — fuse tier-only candidates with composite score.
- ``tier_result_from_entry`` — tier-entry → result-dict.
- ``apply_admission_filter`` — PRD-DIST-2049 c802 confidence / currentness filter.

Extracted as PRD-DIST-246 batch 105 (sub-split).
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

import structlog

from trw_memory._client_distilled_tiering import entry_to_result as _entry_to_result
from trw_memory.lifecycle.scoring import entry_utility
from trw_memory.lifecycle.tiers._runtime import remember_entry_data_in_tiers, tier_candidates
from trw_memory.lifecycle.tiers._scoring import compute_importance_score
from trw_memory.models.config import MemoryConfig
from trw_memory.retrieval.recall_selection import LocalCandidate, RecallInvocation, RemoteCandidate
from trw_memory.security.namespace_scope import NamespaceScopeError, authorize_namespaces
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import StorageBackend

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient, MemoryResultDict

logger = structlog.get_logger(__name__)


def apply_budget(
    results: list[MemoryResultDict],
    token_budget: int | None,
) -> list[MemoryResultDict]:
    """Token-budget filtering (pure). When ``token_budget`` is None, returns input unchanged."""
    if token_budget is None or not results:
        return results

    from trw_memory.retrieval.token_budget import apply_token_budget

    raw: list[dict[str, object]] = list(results)  # type: ignore[arg-type]
    filtered, _used, _truncated = apply_token_budget(raw, token_budget)
    return filtered  # type: ignore[return-value]


async def merge_org_results(
    client: MemoryClient,
    query: str,
    local_results: list[MemoryResultDict],
    limit: int,
    tags: list[str] | None,
    min_score: float,
) -> list[MemoryResultDict]:
    candidates = await _org_candidates(
        client, query, {(r["namespace"], r["memory_id"]) for r in local_results}, limit, tags, min_score
    )
    from trw_memory._client_distilled_tiering import candidate_to_result

    return client._merge_shared_candidates(local_results, [candidate_to_result(c) for c in candidates])


async def collect_org_candidates(
    client: MemoryClient,
    query: str,
    local: list[LocalCandidate],
    limit: int,
    tags: list[str] | None,
    min_score: float,
    invocation: RecallInvocation,
) -> list[LocalCandidate]:
    seen_ids = {c.entry.id for c in local}
    seen_content = {c.entry.content for c in local}
    candidates = await _org_candidates(
        client,
        query,
        {(c.entry.namespace, c.entry.id) for c in local},
        limit,
        tags,
        min_score,
        invocation,
        exclude_ids=seen_ids,
        exclude_content=seen_content,
    )
    admitted = []
    for candidate in candidates:
        if candidate.entry.id in seen_ids or candidate.entry.content in seen_content:
            continue
        if replace(invocation, namespace=candidate.entry.namespace).allows_entry(candidate.entry):
            admitted.append(candidate)
            seen_ids.add(candidate.entry.id)
            seen_content.add(candidate.entry.content)
    return admitted


async def _org_candidates(
    client: MemoryClient,
    query: str,
    exclude_keys: set[tuple[str, str]],
    limit: int,
    tags: list[str] | None,
    min_score: float,
    invocation: RecallInvocation | None = None,
    *,
    exclude_ids: set[str] | None = None,
    exclude_content: set[str] | None = None,
) -> list[LocalCandidate]:
    try:
        from trw_memory import client as _c

        producer = functools.partial(
            _c.list_org_shared_entries,
            client._config,
            client._namespace,
            exclude_keys=exclude_keys,
            limit=limit if invocation is not None else max(limit, 25),
        )
        if invocation is not None:
            producer = functools.partial(
                producer,
                invocation=invocation,
                entry_filter=lambda entry: (
                    entry.importance >= max(0.8, min_score)
                    and (exclude_ids is None or entry.id not in exclude_ids)
                    and (exclude_content is None or entry.content not in exclude_content)
                    and (not tags or set(tags).issubset(entry.tags))
                    and (not query.strip() or client._matches_query(_entry_to_result(entry), query))
                ),
            )
        org_entries = await asyncio.to_thread(producer)
    except NamespaceScopeError:
        raise
    except Exception:  # trw-fail-silent-allow: org entries SUPPLEMENT local recall, so a broken org lookup must degrade rather than fail the whole call; the info-level event above is what keeps the degradation observable
        # info, not debug: under the shipped default (`debug: false`) structlog's
        # filtering bound logger drops a debug event before any processor runs,
        # so the only record that org recall was attempted and broke would be
        # destroyed -- and an empty list then reads as "there are no org
        # entries" rather than "we never got to look". That is the exact
        # we-checked-vs-we-never-checked collapse this release is fixing.
        logger.info("memory_org_recall_failed", namespace=client._namespace, exc_info=True)
        return []
    scope = authorize_namespaces(client._config, {entry.namespace for entry in org_entries}, Permission.READ, "recall")
    candidates: list[LocalCandidate] = []
    for entry in org_entries:
        if entry.namespace not in scope.namespaces:
            raise NamespaceScopeError("org acquisition returned an unauthorized namespace")
        if not entry.cross_validated or entry.importance < max(0.8, min_score):
            continue
        if tags and not set(tags).issubset(entry.tags):
            continue
        if query.strip() and not client._matches_query(_entry_to_result(entry), query):
            continue
        candidates.append(LocalCandidate(entry, entry.importance, source="org"))
    return candidates


def tier_results(
    client: MemoryClient,
    backend: StorageBackend,
    query: str,
    tags: list[str] | None,
    limit: int,
    query_embedding: list[float] | None = None,
    *,
    invocation: RecallInvocation | None = None,
) -> list[MemoryResultDict] | list[LocalCandidate]:
    """Collect local tier-managed candidates for this namespace."""
    candidates = tier_candidates(
        client._config,
        client._namespace,
        backend,
        query=query,
        tags=tags,
        limit=limit,
        query_embedding=query_embedding,
        invocation=invocation,
    )
    if invocation is not None:
        return cast("list[LocalCandidate]", candidates)
    return [tier_result_from_entry(candidate) for candidate in cast("list[dict[str, object]]", candidates)]


def remember_results_in_tiers(
    client: MemoryClient,
    results: list[MemoryResultDict],
) -> None:
    """Keep the hot/warm tiers aligned with the entries callers actually saw."""
    recalled_at = datetime.now(timezone.utc).isoformat()
    for result in results:
        if result.get("source", "local") != "local":
            continue
        payload: dict[str, object] = {
            "id": result["memory_id"],
            "content": result["content"],
            "detail": result["detail"],
            "tags": result["tags"],
            "importance": result["importance"],
            "namespace": result["namespace"],
            "last_accessed_at": recalled_at,
        }
        if result["created_at"]:
            payload["created_at"] = result["created_at"]
        if result["updated_at"]:
            payload["updated_at"] = result["updated_at"]
        remember_entry_data_in_tiers(client._config, payload)


def merge_tier_results(
    local_results: list[MemoryResultDict],
    tier_only_results: list[MemoryResultDict],
    limit: int,
    query_tokens: list[str],
    config: MemoryConfig,
    query_embedding: list[float] | None = None,
) -> list[MemoryResultDict]:
    """Merge tier-only candidates into the normal local recall results."""
    # PRD-DIST-2051 c806: when opt-in flag is set AND hybrid has produced
    # enough candidates, preserve the BM25+dense+RRF ordering instead of
    # rescoring via compute_importance_score. c805 per-layer trace showed the
    # rescore mixes incomparable score scales (RRF 1/(1+rank) vs tier-only
    # entry_utility absolute) and pushes high-rank hybrid results past top-K.
    if config.recall_preserve_hybrid_order and len(local_results) >= limit:
        return local_results[:limit]
    if not tier_only_results:
        return local_results[:limit]
    merged = list(local_results)
    seen_ids = {result["memory_id"] for result in local_results}
    seen_content = {result["content"] for result in local_results}
    for result in tier_only_results:
        if result["memory_id"] in seen_ids or result["content"] in seen_content:
            continue
        merged.append(result)
        seen_ids.add(result["memory_id"])
        seen_content.add(result["content"])
    if len(merged) == len(local_results):
        return local_results[:limit]
    for result in merged:
        relevance_hint = result.get("_relevance_hint")
        result["score"] = round(
            compute_importance_score(
                cast("dict[str, object]", result),
                query_tokens,
                query_embedding=query_embedding,
                config=config,
                relevance_hint=float(relevance_hint) if relevance_hint is not None else None,
            ),
            4,
        )
    merged.sort(key=lambda result: float(result["score"]), reverse=True)
    return merged[:limit]


def tier_result_from_entry(entry: dict[str, object]) -> MemoryResultDict:
    """Convert a tier-managed entry dict into the client recall result shape."""
    from trw_memory.client import MemoryClient

    raw_score = entry.get("score")
    score = float(str(raw_score)) if raw_score is not None else entry_utility(entry)
    raw_tags = entry.get("tags", [])
    raw_metadata = entry.get("metadata") or {}
    # PRD-DIST-2049 c802: preserve metadata so the recall-time admission
    # filter (and any downstream consumer) can read `currentness_status` and
    # related fields. Pre-c802 this helper dropped metadata silently.
    metadata: dict[str, str] = (
        {str(k): str(v) for k, v in raw_metadata.items()} if isinstance(raw_metadata, dict) else {}
    )
    tier_result: MemoryResultDict = {
        "memory_id": str(entry.get("id", entry.get("memory_id", ""))),
        "content": str(entry.get("content", "")),
        "detail": str(entry.get("detail", "")),
        "tags": [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else [],
        "importance": MemoryClient._coerce_float(entry.get("importance", 0.0)),
        "score": round(score, 4),
        "created_at": str(entry.get("created_at", "")),
        "updated_at": str(entry.get("updated_at", entry.get("created_at", ""))),
        "namespace": str(entry.get("namespace", "default")),
        "source": "local",
        "last_accessed_at": str(entry.get("last_accessed_at", "")),
        "q_value": MemoryClient._coerce_float(entry.get("q_value", 0.0)),
        "q_observations": int(str(entry.get("q_observations", 0))),
        "recurrence": int(str(entry.get("recurrence", 1))),
        "access_count": int(str(entry.get("access_count", 0))),
        "metadata": metadata,
        "_relevance_hint": MemoryClient._coerce_float(entry.get("_tier_relevance", score)),
    }
    return tier_result


# ``apply_admission_filter`` was relocated to the shared recall-policy Module
# ``trw_memory.retrieval.admission_policy`` (PRD-DIST-2049 recall-policy seam
# unification) so the SDK recall path and the MCP tool path consume a single
# Implementation. Re-exported here so existing call sites + test patches that
# reference ``trw_memory._client_recall_helpers.apply_admission_filter`` keep
# working unchanged.
from trw_memory.retrieval.admission_policy import (  # noqa: E402
    apply_admission_filter as apply_admission_filter,
)


def merge_local_candidates(
    local: list[LocalCandidate],
    tiers: list[LocalCandidate],
    limit: int,
    query_tokens: list[str],
    config: MemoryConfig,
    query_embedding: list[float] | None,
    *,
    invocation: RecallInvocation | None = None,
) -> list[LocalCandidate]:
    """Merge acquired entries without a requested-result cut or source weighting."""
    seen = {(c.entry.namespace, c.entry.id) for c in local}
    content = {c.entry.content for c in local}
    added = [c for c in tiers if (c.entry.namespace, c.entry.id) not in seen and c.entry.content not in content]
    if not added:
        return local
    merged = [*local, *added]
    if config.recall_preserve_hybrid_order and len(local) >= limit:
        # Keep the whole pool for final admission/refill, but do not compare
        # reciprocal hybrid ranks with tier-only absolute utility scores.
        return [*local, *(replace(c, tier_fallback=True) for c in added)]
    return [
        replace(
            c,
            raw_score=round(
                compute_importance_score(
                    c.entry.model_dump(mode="json"),
                    query_tokens,
                    query_embedding=query_embedding,
                    config=config,
                    relevance_hint=c.relevance_hint,
                    reference_time=invocation.temporal.reference_time if invocation else None,
                ),
                4,
            ),
        )
        for c in merged
    ]


def remember_selected_candidates(
    client: MemoryClient, candidates: list[LocalCandidate], results: list[MemoryResultDict]
) -> None:
    rows = {(r["namespace"], r["memory_id"]): r for r in results}
    for candidate in candidates:
        if candidate.source != "local":
            continue
        row = rows[(candidate.entry.namespace, candidate.entry.id)]
        payload = candidate.entry.model_dump(mode="json")
        # Security-masked returned content is what may enter the cache; validity
        # and provenance still come from the authoritative entry, not projection.
        payload.update(
            content=row["content"], detail=row["detail"], last_accessed_at=datetime.now(timezone.utc).isoformat()
        )
        remember_entry_data_in_tiers(client._config, payload)


async def finish_candidates(
    client: MemoryClient,
    candidates: list[LocalCandidate],
    remote: list[RemoteCandidate],
    invocation: RecallInvocation,
    *,
    query: str,
    limit: int,
    min_score: float,
    token_budget: int | None,
) -> list[MemoryResultDict]:
    from trw_memory._client_distilled_tiering import candidate_to_result
    from trw_memory._client_recall import _finalize_recall

    # Explicit caller weighting takes precedence, even at the default value.
    explicit_weights = bool(invocation.source.explicit_weight_overrides) or invocation.source.explicit_distilled_weight
    ranked: list[tuple[tuple[int, int, int, float], MemoryResultDict]] = []
    for candidate in candidates:
        entry = candidate.entry
        policy = replace(invocation, namespace=entry.namespace) if candidate.source == "org" else invocation
        if not policy.allows_entry(entry):
            continue
        eligible = policy.temporal.eligible(entry)
        if not eligible and not policy.temporal.include_superseded:
            continue
        row = candidate_to_result(candidate)
        bucket, negative_score = policy.source.rank_key(row)
        row["score"] = -negative_score
        if candidate.raw_score >= min_score and row["score"] >= min_score:
            fallback = int(candidate.tier_fallback and not explicit_weights)
            ranked.append(((int(not eligible), bucket, fallback, negative_score), row))
    unknown_windows = 0
    for remote_candidate in remote:
        eligible_remote = remote_candidate.temporal_eligibility(invocation.temporal)
        if eligible_remote is False and not invocation.temporal.include_superseded:
            continue
        unknown_windows += eligible_remote is None
        remote_row = dict(remote_candidate.result)
        if not invocation.source.allows(remote_row) or not apply_admission_filter(
            [remote_row],
            confidence_floor=invocation.confidence_floor,
            exclude_historical_only=invocation.exclude_historical_only,
        ):
            continue
        bucket, negative_score = invocation.source.rank_key(remote_row)
        remote_row["score"] = -negative_score
        for field in ("valid_from", "invalid_from", "invalidated_by"):
            remote_row.pop(field, None)
        if -negative_score >= min_score:
            ranked.append(
                ((int(eligible_remote is False), bucket, 0, negative_score), cast("MemoryResultDict", remote_row))
            )
    if unknown_windows:
        logger.debug(
            "recall_remote_temporal_coverage",
            unknown_windows=unknown_windows,
            historical_coverage="limited" if invocation.temporal.as_of is not None else "unknown",
        )
    ranked.sort(key=lambda item: item[0])
    return await _finalize_recall(
        client, [row for _, row in ranked], query=query, limit=limit, token_budget=token_budget, candidates=candidates
    )
