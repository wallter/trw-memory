"""Recall-helper sub-cluster — tier ops + budget + org-merge.

Belongs to ``client.py`` recall pipeline. Re-exported via
``_client_recall.py``. Split out from the parent recall module so each
file stays under the 350 effective-LOC gate (PRD-DIST-246 batch 105).

6 helpers:

- ``apply_budget`` — pure token-budget filtering.
- ``merge_org_results`` — append cross-validated sibling memories.
- ``tier_results`` — collect local tier-managed candidates.
- ``remember_results_in_tiers`` — keep hot/warm tiers aligned.
- ``merge_tier_results`` — fuse tier-only candidates with composite score.
- ``tier_result_from_entry`` — tier-entry → result-dict.

Extracted as PRD-DIST-246 batch 105 (sub-split).
"""

from __future__ import annotations

import asyncio
import functools
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

import structlog

from trw_memory.lifecycle.scoring import entry_utility
from trw_memory.lifecycle.tiers._runtime import remember_entry_data_in_tiers, tier_candidates
from trw_memory.lifecycle.tiers._scoring import compute_importance_score
from trw_memory.models.config import MemoryConfig
from trw_memory.storage.interface import StorageBackend

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient, MemoryResultDict

logger = structlog.get_logger(__name__)


def _entry_to_result(entry: object, score: float = 0.0) -> MemoryResultDict:
    from trw_memory._client_distilled_tiering import entry_to_result as _impl

    return _impl(entry, score=score)  # type: ignore[arg-type]


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
    """Append cross-validated sibling-project memories after local results."""
    try:
        from trw_memory import client as _c
        org_entries = await asyncio.to_thread(
            functools.partial(
                _c.list_org_shared_entries,
                client._config,
                client._namespace,
                exclude_keys={(result["namespace"], result["memory_id"]) for result in local_results},
                limit=max(limit, 25),
            )
        )
    except Exception:
        logger.debug(
            "memory_org_recall_failed",
            op="recall",
            outcome="failure",
            namespace=client._namespace,
            exc_info=True,
        )
        return local_results

    tag_set = set(tags or [])
    org_results: list[MemoryResultDict] = []
    for entry in org_entries:
        if not entry.cross_validated or entry.importance < 0.8:
            continue
        if tag_set and not tag_set.issubset(set(entry.tags)):
            continue

        candidate = _entry_to_result(entry, score=entry.importance)
        candidate["source"] = "org"
        if query.strip() and not client._matches_query(candidate, query):
            continue
        if min_score > 0.0 and candidate["score"] < min_score:
            continue
        org_results.append(candidate)

    return client._merge_shared_candidates(local_results, org_results)


def tier_results(
    client: MemoryClient,
    backend: StorageBackend,
    query: str,
    tags: list[str] | None,
    limit: int,
    query_embedding: list[float] | None = None,
) -> list[MemoryResultDict]:
    """Collect local tier-managed candidates for this namespace."""
    candidates = tier_candidates(
        client._config,
        client._namespace,
        backend,
        query=query,
        tags=tags,
        limit=limit,
        query_embedding=query_embedding,
    )
    return [tier_result_from_entry(candidate) for candidate in candidates]


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
        "_relevance_hint": MemoryClient._coerce_float(entry.get("_tier_relevance", score)),
    }
    return tier_result
