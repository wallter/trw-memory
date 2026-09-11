"""Characterize shared recall finishing before removing duplicate branches.

Acquisition and security are controlled boundaries here, not correctness proofs
for the search algorithms or security filter. Public source-selection tests use
real acquisition separately. No remote services, tiers or models participate.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from trw_memory._client_distilled_tiering import entry_to_result
from trw_memory.client import MemoryClient, MemoryResultDict
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.recall_selection import LocalCandidate


@pytest.fixture(params=["hybrid", "fallback"])
def routed_client(
    request: pytest.FixtureRequest, client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> MemoryClient:
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    monkeypatch.setattr("trw_memory._client_recall.tier_runtime_enabled", lambda _: False)
    rows = [
        LocalCandidate(
            MemoryEntry(
                id="historical", content="historical policy", metadata={"currentness_status": "historical_only"}
            ),
            raw_score=0.99,
        ),
        LocalCandidate(MemoryEntry(id="durable", content="durable policy"), raw_score=0.8),
        LocalCandidate(
            MemoryEntry(id="episode", content="episode policy", metadata={"source_kind": "episodic"}),
            raw_score=0.95,
        ),
    ]
    if request.param == "hybrid":
        monkeypatch.setattr(client, "_try_hybrid_recall", AsyncMock(return_value=rows))
        monkeypatch.setattr(client, "_fallback_recall", AsyncMock(side_effect=AssertionError("unexpected fallback")))
    else:
        monkeypatch.setattr(client, "_try_hybrid_recall", AsyncMock(return_value=None))
        monkeypatch.setattr(client, "_fallback_recall", AsyncMock(return_value=rows))
    return client


async def test_shared_finish_accounts_only_security_accepted_results(
    routed_client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = routed_client
    events: list[str] = []

    async def org(_client: MemoryClient, query: str, rows: list[LocalCandidate], *_: object) -> list[LocalCandidate]:
        events.append("org")
        return [LocalCandidate(MemoryEntry(id="org", content=query), raw_score=0.6, source="org")]

    async def shared(query: str, rows: list[MemoryResultDict], *_: object, **kwargs: object) -> list[MemoryResultDict]:
        events.append("shared")
        return [*rows, dict(entry_to_result(MemoryEntry(id="shared", content=query), score=0.5), source="shared")]

    def secure(rows: list[MemoryResultDict]) -> list[MemoryResultDict]:
        events.append("security")
        assert [r["memory_id"] for r in rows] == ["durable", "org", "shared"]
        return [dict(rows[0], content="masked")]

    async def access(rows: list[MemoryResultDict]) -> None:
        events.append("access")
        assert [(r["memory_id"], r["content"]) for r in rows] == [("durable", "masked")]

    def remember(_client: MemoryClient, candidates: list[LocalCandidate], rows: list[MemoryResultDict]) -> None:
        events.append("remember")
        assert [c.entry.id for c in candidates] == ["durable"]
        assert candidates[0].entry.content == "durable policy"
        assert [(r["memory_id"], r["content"]) for r in rows] == [("durable", "masked")]

    monkeypatch.setattr("trw_memory._client_recall_helpers.collect_org_candidates", org)
    monkeypatch.setattr(client, "_merge_shared_results", shared)
    monkeypatch.setattr(client, "_apply_recall_security", secure)
    monkeypatch.setattr(client, "_record_recall_access", access)
    monkeypatch.setattr("trw_memory._client_recall_helpers.remember_selected_candidates", remember)
    try:
        result = await client.recall(
            "policy",
            limit=10,
            include_shared=True,
            exclude_historical_only=True,
            exclude_source_kinds=["episodic"],
            min_score=0.4,
        )
        assert [r["memory_id"] for r in result] == ["durable"]
        assert events == ["org", "shared", "security", "access", "remember"]
    finally:
        client._get_backend().close()


async def test_security_failure_does_not_record_success(
    routed_client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = routed_client
    failure = RuntimeError("security refuses recall")
    access = AsyncMock()
    remember = Mock()
    monkeypatch.setattr(client, "_apply_recall_security", Mock(side_effect=failure))
    monkeypatch.setattr(client, "_record_recall_access", access)
    monkeypatch.setattr("trw_memory._client_recall_helpers.remember_selected_candidates", remember)
    try:
        with pytest.raises(RuntimeError) as caught:
            await client.recall("policy", include_org_memories=False)
        assert caught.value is failure
        access.assert_not_awaited()
        remember.assert_not_called()
    finally:
        client._get_backend().close()
