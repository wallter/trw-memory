"""Public recall must apply source policy before candidate/result cuts (DIST-012)."""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.memory import MemoryEntry


@pytest.fixture(params=["sqlite", "yaml"])
def source_client(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MemoryClient]:
    for key in ("HOME", "TRW_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", request.param)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    # Only tier participation is suppressed; real backend acquisition/scoring stays live.
    from trw_memory import _client_lifecycle, _client_recall
    from trw_memory.lifecycle.tiers import _runtime

    for module in (_client_lifecycle, _client_recall, _runtime):
        monkeypatch.setattr(module, "tier_runtime_enabled", lambda _: False)
    client = MemoryClient(namespace="default", mode="local")
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    monkeypatch.setattr(client._config, "hybrid_search_candidate_pool_size", 10)
    yield client
    client._get_backend().close()


@pytest.fixture(params=["fallback", "hybrid"])
def recall_path(request: pytest.FixtureRequest, source_client: MemoryClient, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.param == "fallback":
        monkeypatch.setattr(source_client, "_try_hybrid_recall", AsyncMock(return_value=None))
    else:
        monkeypatch.setattr(
            source_client, "_fallback_recall", AsyncMock(side_effect=AssertionError("actual hybrid must not fall back"))
        )


def seed(client: MemoryClient, *, competitors: int = 1, family: str | None = "episodic") -> None:
    backend = client._get_backend()
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    backend.store(
        MemoryEntry(
            id="durable",
            content="Selectionprobe policy",
            importance=0.1,
            valid_from=start,
            updated_at=start,
            metadata={"source_kind": "semantic_memory"},
        )
    )
    for index in range(competitors):
        backend.store(
            MemoryEntry(
                id=f"competitor-{index}",
                content="Selectionprobe policy",
                importance=0.99,
                valid_from=start,
                updated_at=start + timedelta(days=index + 1),
                metadata={"source_kind": family} if family else {},
            )
        )


async def recall(client: MemoryClient, **kwargs):
    return await client.recall("Selectionprobe", include_shared=False, include_org_memories=False, **kwargs)


async def test_live_default_top_one_matches_top_two(source_client: MemoryClient, recall_path: None) -> None:
    seed(source_client)
    top_two = await recall(source_client, limit=2)
    top_one = await recall(source_client, limit=1)
    assert [row["memory_id"] for row in top_two] == ["durable", "competitor-0"]
    assert [row["memory_id"] for row in top_one] == ["durable"]


async def test_excluded_source_cannot_consume_acquisition_pool(source_client: MemoryClient, recall_path: None) -> None:
    # 128 exceeds pool=10 and SQLite FTS augmentation's minimum cap=100.
    seed(source_client, competitors=128)
    results = await recall(source_client, limit=1, exclude_source_kinds=["episodic"])
    assert [row["memory_id"] for row in results] == ["durable"]


async def test_explicit_transient_weight_override_removes_default_containment(
    source_client: MemoryClient, recall_path: None
) -> None:
    seed(source_client)
    results = await recall(source_client, limit=2, source_weights={"episodic": 2.0})
    assert [row["memory_id"] for row in results] == ["competitor-0", "durable"]


async def test_unknown_source_remains_neutral_not_transient(source_client: MemoryClient, recall_path: None) -> None:
    seed(source_client, family=None)
    neutral = await recall(source_client, limit=2, exclude_source_kinds=["episodic"])
    explicit = await recall(source_client, limit=2, source_weights={"unknown": 1.0}, exclude_source_kinds=["episodic"])
    assert [row["memory_id"] for row in neutral] == ["competitor-0", "durable"]
    assert [(row["memory_id"], row["score"]) for row in neutral] == [
        (row["memory_id"], row["score"]) for row in explicit
    ]


async def test_default_durable_containment_survives_owned_acquisition_cap(
    source_client: MemoryClient, recall_path: None
) -> None:
    # More admitted transient competitors than fallback's 3x cap, the configured
    # hybrid pool, and FTS augmentation; exclusion is deliberately NOT requested.
    seed(source_client, competitors=128)
    results = await recall(source_client, limit=1)
    assert [row["memory_id"] for row in results] == ["durable"]


async def test_reranking_cannot_promote_deferred_durable_over_live_transient(
    source_client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(source_client, "_fallback_recall", AsyncMock(side_effect=AssertionError("hybrid required")))
    reranked = []

    def reverse_ranking(query, entries, **kwargs):
        reranked.append([entry.id for entry in entries])
        return [(entry, float(i)) for i, entry in enumerate(entries)][::-1]  # last in = highest score, first out

    monkeypatch.setattr("trw_memory.retrieval.reranker.cross_encode_scores", reverse_ranking)
    backend = source_client._get_backend()
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    backend.store(
        MemoryEntry(id="live", content="Selectionprobe live", valid_from=start, metadata={"source_kind": "episodic"})
    )
    backend.store(
        MemoryEntry(
            id="closed",
            content="Selectionprobe closed",
            valid_from=start,
            invalid_from=start + timedelta(days=1),
            invalidated_by="live",
            metadata={"source_kind": "semantic_memory"},
        )
    )
    result = await recall(source_client, limit=2, include_superseded=True)
    assert reranked == [["live", "closed"]]
    assert [row["memory_id"] for row in result] == ["live", "closed"]
