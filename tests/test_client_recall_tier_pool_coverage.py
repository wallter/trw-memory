"""Tier discovery skips rows a complete hybrid pool already ranked.

When the hybrid candidate pool held the whole namespace, every hot/warm row
with a primary copy was already ranked by BM25 + dense (+ rerank), and
``merge_local_candidates`` would drop its tier duplicate anyway. Recall tells
tier discovery which rows those are, so it neither resolves nor vector-scores
them; only the cold archive and warm rows with no primary copy can add
candidates. A pool the cap may have cut keeps the full discovery.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import MemoryClient
from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.lifecycle.tiers import _warm_space
from trw_memory.lifecycle.tiers._manager import TierManager
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.persistence import write_yaml

pytest.importorskip("sqlite_vec")

ROWS = 12
SPACE = EmbeddingSpace("c" * 64, "test-encoder:pool-coverage", 2)


@pytest.fixture
def embedder() -> MagicMock:
    fake = MagicMock()
    fake.embed.return_value = [1.0, 0.0]
    fake.embed_query.return_value = [1.0, 0.0]
    # Vectors are dense-scored only within the embedder's recorded space.
    fake.embedding_space.return_value = SPACE
    return fake


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, list[Any]]]:
    """Record the covered ids tier discovery was given and every warm vector decode."""
    seen: dict[str, list[Any]] = {"covered": [], "decoded": []}
    real_search = TierManager.search
    real_decode = _warm_space.get_vector_records

    def search(self: TierManager, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("invocation") is not None:
            seen["covered"].append(kwargs.get("covered_ids", frozenset()))
        return real_search(self, *args, **kwargs)

    def decode(*args: Any, **kwargs: Any) -> Any:
        seen["decoded"].append(list(kwargs.get("entry_ids") or []))
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(TierManager, "search", search)
    monkeypatch.setattr(_warm_space, "get_vector_records", decode)
    yield seen


async def _client_with_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: MagicMock
) -> tuple[MemoryClient, set[str]]:
    """A client over ROWS stored lessons (mirrored into hot + warm); returns their ids.

    The namespace also holds the system canary rows, which the hybrid pool
    excludes by policy -- they are outside the pool, so never "covered".
    """
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    # Rerank is unconditional (PRD-CORE-284); an unavailable model keeps fusion order.
    monkeypatch.setattr("trw_memory.retrieval.reranker.cross_encode_scores", lambda *a, **k: None)
    client = MemoryClient(namespace="default", mode="local")
    with patch.object(client, "_get_embedder", return_value=embedder):
        stored = [await client.store(f"needle lesson number {i}", importance=0.5) for i in range(ROWS)]
    return client, {row["memory_id"] for row in stored}


async def test_complete_pool_skips_warm_vector_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: MagicMock, spies: dict[str, list[Any]]
) -> None:
    client, stored_ids = await _client_with_rows(tmp_path, monkeypatch, embedder)
    assert client._tier_manager is not None
    assert stored_ids <= {str(row["id"]) for row in client._tier_manager._warm_store.discovery_entries(None)}

    with patch.object(client, "_get_embedder", return_value=embedder):
        results = await client.recall("needle lesson", limit=5, include_org_memories=False)

    assert len(results) == 5
    # The pool (default cap 1000) held every lesson: discovery was told so ...
    assert spies["covered"] == [frozenset(stored_ids)]
    # ... and decoded no warm vector for any of them.
    assert not stored_ids & {entry_id for call in spies["decoded"] for entry_id in call}
    await client.close()


async def test_capped_pool_keeps_full_warm_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: MagicMock, spies: dict[str, list[Any]]
) -> None:
    client, stored_ids = await _client_with_rows(tmp_path, monkeypatch, embedder)
    # Pool cap = max(limit * 5, 10) = 10 < 12 lessons: the cap bound, so the pool
    # may be missing rows and tier discovery must see everything.
    client._config = client._config.model_copy(update={"hybrid_search_candidate_pool_size": 10})

    with patch.object(client, "_get_embedder", return_value=embedder):
        await client.recall("needle lesson", limit=2, include_org_memories=False)

    assert spies["covered"] == [frozenset()]
    assert len(spies["decoded"]) == 1
    assert set(spies["decoded"][0]) >= stored_ids
    await client.close()


async def test_warm_row_without_primary_copy_is_still_scored_under_complete_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: MagicMock, spies: dict[str, list[Any]]
) -> None:
    """A warm row the primary backend lacks is outside the pool, so it still counts."""
    client, stored_ids = await _client_with_rows(tmp_path, monkeypatch, embedder)
    assert client._tier_manager is not None
    client._tier_manager.warm_add(
        "M-warm-only",
        MemoryEntry(id="M-warm-only", content="opaque title", namespace="default", importance=0.9).model_dump(
            mode="json"
        ),
        [1.0, 0.0],
        provenance=VectorProvenance.for_vector(SPACE, "opaque title ", [1.0, 0.0]),
    )

    with patch.object(client, "_get_embedder", return_value=embedder):
        results = await client.recall("needle lesson", limit=20, include_org_memories=False)

    assert "M-warm-only" not in spies["covered"][0]
    assert len(spies["decoded"]) == 1
    decoded = set(spies["decoded"][0])
    assert "M-warm-only" in decoded
    assert not decoded & stored_ids
    assert "M-warm-only" in {row["memory_id"] for row in results}
    await client.close()


async def test_cold_hit_surfaces_and_promotes_under_complete_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: MagicMock, spies: dict[str, list[Any]]
) -> None:
    client, stored_ids = await _client_with_rows(tmp_path, monkeypatch, embedder)
    assert client._tier_manager is not None
    cold_partition = client._tier_manager._cold_dir() / "2026" / "04"
    cold_partition.mkdir(parents=True, exist_ok=True)
    cold_file = cold_partition / "archived-needle.yaml"
    write_yaml(
        cold_file,
        MemoryEntry(
            id="M-cold-needle",
            content="archived needle deployment lesson",
            namespace="default",
            importance=0.9,
        ).model_dump(mode="json"),
    )

    with patch.object(client, "_get_embedder", return_value=embedder):
        results = await client.recall("archived needle deployment", limit=20, include_org_memories=False)

    # The pool was complete (discovery got the covered set) and the cold
    # archive, which the hybrid pool cannot see, still contributed its hit.
    assert spies["covered"] == [frozenset(stored_ids)]
    assert "M-cold-needle" in {row["memory_id"] for row in results}
    assert not cold_file.exists()
    assert client._get_backend().get("M-cold-needle", namespace="default") is not None
    await client.close()


def test_covered_ids_rejected_outside_discovery(tmp_path: Path) -> None:
    """The legacy (non-invocation) search has no covered-row semantics."""
    with TierManager(base_dir=tmp_path) as manager, pytest.raises(ValueError, match="covered_ids"):
        manager.search(["needle"], covered_ids=frozenset({"x"}))
