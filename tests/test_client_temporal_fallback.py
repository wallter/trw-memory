"""Standalone MemoryClient must preserve temporal semantics in keyword fallback."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.memory import MemoryEntry


@pytest.mark.parametrize("include_superseded", [False, True])
async def test_public_client_fallback_honors_historical_window(
    client: MemoryClient, monkeypatch: pytest.MonkeyPatch, include_superseded: bool
) -> None:
    backend = client._get_backend()
    try:
        backend.store(
            MemoryEntry(
                id="future",
                content="Temporalclient policy",
                importance=0.9,
                valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )
        )
        backend.store(
            MemoryEntry(
                id="historical",
                content="Temporalclient policy",
                importance=0.4,
                valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
                invalid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
                invalidated_by="future",
            )
        )
        monkeypatch.setattr(client, "_get_embedder", lambda: None)
        unavailable = AsyncMock(return_value=None)
        monkeypatch.setattr(client, "_try_hybrid_recall", unavailable)
        result = await client.recall(
            "Temporalclient",
            limit=1,
            as_of=datetime(2022, 1, 1, tzinfo=timezone.utc),
            include_superseded=include_superseded,
            include_shared=False,
            include_org_memories=False,
        )
        unavailable.assert_awaited_once()
        assert [row["memory_id"] for row in result] == ["historical"]
    finally:
        backend.close()


@pytest.mark.parametrize("route", ["fallback", "injected-hybrid", "bm25-hybrid"])
@pytest.mark.parametrize("metadata_expiry", [False, True])
@pytest.mark.parametrize("as_of_year", [2022, 2024, 2025])
async def test_public_client_source_expiry_uses_query_time(
    client: MemoryClient,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    metadata_expiry: bool,
    as_of_year: int,
) -> None:
    """Real fallback/BM25 or injected hybrid candidates traverse source policy."""
    from trw_memory.retrieval.recall_selection import LocalCandidate

    backend = client._get_backend()
    expiry = "2024-01-01"
    entry = MemoryEntry(
        id="historical-episode",
        content="Temporalclient policy",
        tags=["source_kind:episodic"],
        importance=0.7,
        valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
        expires="" if metadata_expiry else expiry,
        metadata={"expires": expiry} if metadata_expiry else {},
    )
    backend.store(entry)
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    if route == "bm25-hybrid":
        # Exercise the real hybrid acquisition/ranking path, with no model.
        # A fallback cannot rescue an empty/failed hybrid result into a pass.
        if as_of_year <= 2024:
            monkeypatch.setattr(
                client, "_fallback_recall", AsyncMock(side_effect=AssertionError("unexpected fallback"))
            )
    else:
        candidate_result = [LocalCandidate(entry, raw_score=0.7)] if route == "injected-hybrid" else None
        monkeypatch.setattr(client, "_try_hybrid_recall", AsyncMock(return_value=candidate_result))
    try:
        returned = await client.recall(
            "Temporalclient",
            limit=1,
            as_of=datetime(as_of_year, 1, 1, tzinfo=timezone.utc),
            include_shared=False,
            include_org_memories=False,
        )
        expected = [entry.id] if as_of_year <= 2024 else []
        assert [row["memory_id"] for row in returned] == expected
    finally:
        backend.close()


@pytest.mark.parametrize("as_of", ["2024-01-02T00:30:00+02:00", "2024-01-01T22:30:00Z"])
async def test_public_client_expiry_preserves_equivalent_instants(
    client: MemoryClient, monkeypatch: pytest.MonkeyPatch, as_of: str
) -> None:
    backend = client._get_backend()
    backend.store(
        MemoryEntry(
            id="offset-episode",
            content="Temporalclient policy",
            tags=["source_kind:episodic"],
            valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
            expires="2024-01-01",
        )
    )
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    monkeypatch.setattr(client, "_try_hybrid_recall", AsyncMock(return_value=None))
    try:
        results = await client.recall(
            "Temporalclient",
            as_of=datetime.fromisoformat(as_of.replace("Z", "+00:00")),
            include_shared=False,
            include_org_memories=False,
        )
        assert [row["memory_id"] for row in results] == ["offset-episode"]
    finally:
        backend.close()
