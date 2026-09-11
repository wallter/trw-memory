"""Hybrid preservation survives final policy without changing relevance evidence."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from trw_memory._client_recall_helpers import finish_candidates, merge_local_candidates
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.recall_selection import LocalCandidate, RecallInvocation
from trw_memory.retrieval.source_policy import SourcePolicy
from trw_memory.retrieval.temporal_selection import TemporalSelection
from trw_memory.security.namespace_scope import NamespaceScopeError

NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def candidate(name, score, *, kind=None, **fields):
    metadata = {"source_kind": kind} if kind else {}
    return LocalCandidate(MemoryEntry(id=name, content=name, metadata=metadata, **fields), score, relevance_hint=score)


def policy(**kwargs):
    return RecallInvocation(
        SourcePolicy.resolve(reference_time=NOW, **kwargs), TemporalSelection(reference_time=NOW), "default"
    )


async def select(monkeypatch, candidates, invocation, limit=1):
    """Exercise real final policy; capture the uncut pool at its finalization seam."""
    from trw_memory import _client_recall

    observed = []

    async def finalize(client, rows, **kwargs):
        observed.extend(rows)
        assert kwargs["candidates"] is candidates
        return rows[: kwargs["limit"]]

    monkeypatch.setattr(_client_recall, "_finalize_recall", finalize)
    rows = await finish_candidates(
        None, candidates, [], invocation, query="probe", limit=limit, min_score=0, token_budget=None
    )
    return [r["memory_id"] for r in rows], observed


def merge(local, tiers, *, preserve=True, limit=1, invocation=None):
    return merge_local_candidates(
        local, tiers, limit, ["probe"], MemoryConfig(recall_preserve_hybrid_order=preserve), None, invocation=invocation
    )


@pytest.mark.asyncio
async def test_mixed_scales_preserve_hybrid_without_losing_candidates_or_scores(monkeypatch):
    local = candidate("hybrid", 0.25)
    tier = candidate("tier", 0.59)
    merged = merge([local], [tier])
    assert len(merged) == 2
    assert merged[0] is local
    assert merged[1].entry is tier.entry
    assert [(c.raw_score, c.relevance_hint) for c in merged] == [(0.25, 0.25), (0.59, 0.59)]
    assert not tier.tier_fallback
    ids, pool = await select(monkeypatch, merged, policy())
    assert ids == ["hybrid"]
    assert [r["memory_id"] for r in pool] == ["hybrid", "tier"]
    assert [r["score"] for r in pool] == [0.25, 0.59]


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", ["excluded", "expired", "zero"])
async def test_rejected_hybrid_refills_from_retained_tier(monkeypatch, rejection):
    local = candidate("hybrid", 0.25, kind="episodic")
    invocation = policy()
    if rejection == "excluded":
        invocation = policy(exclude_source_kinds=["episodic"])
    elif rejection == "zero":
        invocation = policy(source_weights={"episodic": 0.0})
    else:
        local = replace(local, entry=local.entry.model_copy(update={"expires": (NOW - timedelta(days=1)).isoformat()}))
    ids, pool = await select(monkeypatch, merge([local], [candidate("tier", 0.59)]), invocation)
    assert ids == ["tier"]
    assert len(pool) == 1


@pytest.mark.asyncio
async def test_source_containment_precedes_hybrid_priority(monkeypatch):
    merged = merge([candidate("episodic", 0.99, kind="episodic")], [candidate("durable", 0.2)])
    ids, pool = await select(monkeypatch, merged, policy())
    assert ids == ["durable"]
    assert len(pool) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("weights", [{"source_weights": {"semantic_memory": 2.0}}, {"distilled_weight": 2.0}])
async def test_explicit_source_weight_overrides_preservation(monkeypatch, weights):
    kind = "semantic_memory" if "source_weights" in weights else "git"
    merged = merge([candidate("hybrid", 0.25)], [candidate("tier", 0.2, kind=kind)])
    ids, pool = await select(monkeypatch, merged, policy(**weights))
    assert ids == ["tier"]
    assert pool[0]["score"] == 0.4
    assert merged[1].raw_score == 0.2


@pytest.mark.asyncio
async def test_disabled_preservation_keeps_existing_common_rescore(monkeypatch):
    from trw_memory import _client_recall_helpers

    monkeypatch.setattr(
        _client_recall_helpers, "compute_importance_score", lambda row, *a, **kw: 0.8 if row["id"] == "tier" else 0.3
    )
    merged = merge([candidate("hybrid", 0.25)], [candidate("tier", 0.59)], preserve=False)
    assert not any(c.tier_fallback for c in merged)
    ids, _ = await select(monkeypatch, merged, policy())
    assert ids == ["tier"]


@pytest.mark.asyncio
async def test_namespace_fence_applies_before_priority(monkeypatch):
    merged = merge([candidate("hybrid", 0.25)], [candidate("foreign", 0.59, namespace="other")])
    with pytest.raises(NamespaceScopeError):
        await select(monkeypatch, merged, policy())


@pytest.mark.asyncio
@pytest.mark.parametrize("weight, expected", [(None, "hybrid"), (0.75, "tier")])
async def test_explicit_default_legacy_weight_preserves_presence(monkeypatch, weight, expected):
    invocation = policy(distilled_weight=weight)
    assert invocation.source.explicit_distilled_weight is (weight is not None)
    merged = merge([candidate("hybrid", 0.25)], [candidate("tier", 0.59, kind="git")])
    ids, _ = await select(monkeypatch, merged, invocation)
    assert ids == [expected]
    assert merged[1].raw_score == 0.59
