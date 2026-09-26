"""Public standalone hybrid recall selects eligible records before pool caps."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.memory import MemoryEntry


@pytest.mark.parametrize("include_superseded", [False, True])
@pytest.mark.parametrize("fts_enabled", [False, True])
async def test_public_hybrid_pool_does_not_starve_historical_candidate(
    client: MemoryClient, monkeypatch: pytest.MonkeyPatch, include_superseded: bool, fts_enabled: bool
) -> None:
    backend = client._get_backend()
    if fts_enabled and not backend.fts_available:
        pytest.skip("SQLite has no FTS5")
    monkeypatch.setattr(client._config, "hybrid_search_candidate_pool_size", 10)
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    monkeypatch.setattr(client, "_fallback_recall", AsyncMock(side_effect=AssertionError("hybrid must succeed")))
    if not fts_enabled:
        monkeypatch.setattr(backend, "_fts_available", False)
    try:
        for index in range(32):
            backend.store(
                MemoryEntry(
                    id=f"future-{index}",
                    content="Temporalpool policy",
                    importance=0.9,
                    updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
                )
            )
        backend.store(
            MemoryEntry(
                id="historical",
                content="Temporalpool policy",
                importance=0.4,
                updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
                invalid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
                invalidated_by="future-0",
            )
        )
        result = await client.recall(
            "Temporalpool",
            limit=1,
            as_of=datetime(2022, 1, 1, tzinfo=timezone.utc),
            include_superseded=include_superseded,
            include_shared=False,
            include_org_memories=False,
        )
        assert [row["memory_id"] for row in result] == ["historical"]
    finally:
        backend.close()
