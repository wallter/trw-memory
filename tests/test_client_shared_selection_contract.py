"""Shared cache must cross the same admission boundary as HTTP results.

Real public recall and SSE cache; acquisition is empty, embedder disabled and
tiers suppressed to isolate shared ingress. No remote/model/GPU calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from trw_memory._client_lifecycle import handle_sse_event
from trw_memory._client_org_shared import shared_result_to_result, snapshot_cached_shared_results
from trw_memory.client import MemoryClient
from trw_memory.sync._remote_fetch import SharedFetchResult


@pytest.mark.parametrize("fetch_failure", [False, True])
@pytest.mark.parametrize("verdict", ["allow", "refuse", "error"])
async def test_cached_shared_result_is_admitted_before_normal_or_fallback_return(
    client: MemoryClient, monkeypatch: pytest.MonkeyPatch, fetch_failure: bool, verdict: str
) -> None:
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    monkeypatch.setattr(client, "_try_hybrid_recall", AsyncMock(return_value=[]))
    monkeypatch.setattr("trw_memory._client_recall.tier_runtime_enabled", lambda _: False)
    handle_sse_event(client, {"type": "learning_published", "id": "peer-1", "summary": "Cached policy"})
    gate = Mock(return_value=Mock(quarantined=verdict == "refuse", entry=Mock()))
    if verdict == "error":
        gate.side_effect = RuntimeError("admission unavailable")
    monkeypatch.setattr("trw_memory.security.runtime.prepare_entry_for_store", gate)
    monkeypatch.setattr("trw_memory.security.runtime.store_quarantined_entry", Mock())
    fetch = Mock(return_value=SharedFetchResult([], "disabled", 0, 0))
    if fetch_failure:
        fetch.side_effect = RuntimeError("fetch failed")
    monkeypatch.setattr("trw_memory.client.fetch_shared_memories", fetch)
    try:
        results = await client.recall("Cached policy", include_shared=True, include_org_memories=False)
        expected = ["peer-1"] if verdict == "allow" else []
        assert [row["memory_id"] for row in results] == expected
        gate.assert_called_once()
        admitted_entry = gate.call_args.args[0]
        assert admitted_entry.id == "peer-1"
        assert admitted_entry.namespace == "org:shared"
        assert admitted_entry.source == "agent"
    finally:
        client._get_backend().close()


@pytest.mark.parametrize("through_cache", [False, True])
@pytest.mark.parametrize("supplied", [False, True])
def test_remote_projection_preserves_supplied_validity_without_defaulting(
    client: MemoryClient, through_cache: bool, supplied: bool
) -> None:
    payload: dict[str, object] = {
        "type": "learning_published",
        "id": "peer-1",
        "summary": "Cached policy",
        "content": "Cached policy",
        "namespace": "default",
        "source": "local",
        "created_at": "2020-01-01T00:00:00+00:00",
        "metadata": {"source_kind": "episodic", "confidence": "0.7"},
        "expires": "2030-01-01",
    }
    if supplied:
        payload.update(valid_from="2019-01-01T00:00:00+00:00", invalid_from=None, invalidated_by=None)
    if through_cache:
        handle_sse_event(client, payload)
        result = snapshot_cached_shared_results(client, "policy")[0]
    else:
        result = shared_result_to_result(payload)
    assert result["source"] == result["namespace"] == "shared"
    assert result["metadata"] == payload["metadata"]
    assert result["expires"] == "2030-01-01"
    assert result["created_at"] == payload["created_at"]
    for field in ("valid_from", "invalid_from", "invalidated_by"):
        assert (field in result) == supplied
        if supplied:
            assert result.get(field) == payload[field]


@pytest.mark.parametrize(
    ("supplied", "year", "include_superseded", "expected"),
    [(False, 1970, False, True), (True, 2022, False, True), (True, 2025, False, False), (True, 2025, True, True)],
)
async def test_public_historical_shared_recall_uses_only_supplied_validity(
    client: MemoryClient,
    monkeypatch: pytest.MonkeyPatch,
    supplied: bool,
    year: int,
    include_superseded: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    monkeypatch.setattr(client, "_try_hybrid_recall", AsyncMock(return_value=[]))
    monkeypatch.setattr("trw_memory._client_recall.tier_runtime_enabled", lambda _: False)
    monkeypatch.setattr(
        "trw_memory.client.fetch_shared_memories", Mock(return_value=SharedFetchResult([], "disabled", 0, 0))
    )
    monkeypatch.setattr(
        "trw_memory.security.runtime.prepare_entry_for_store", Mock(return_value=Mock(quarantined=False))
    )
    event: dict[str, object] = {
        "type": "learning_published",
        "id": "peer-1",
        "summary": "Cached policy",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    if supplied:
        event.update(valid_from="2020-01-01T00:00:00+00:00", invalid_from="2024-01-01T00:00:00+00:00")
    handle_sse_event(client, event)
    try:
        result = await client.recall(
            "Cached policy",
            include_shared=True,
            include_org_memories=False,
            as_of=datetime(year, 1, 1, tzinfo=timezone.utc),
            include_superseded=include_superseded,
        )
        assert [row["memory_id"] for row in result] == (["peer-1"] if expected else [])
        for row in result:
            assert "valid_from" not in row and "invalid_from" not in row
    finally:
        client._get_backend().close()
