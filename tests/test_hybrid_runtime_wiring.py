"""Runtime wiring tests for vector persistence and hybrid recall."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import structlog

from trw_memory.client import MemoryClient
from trw_memory.exceptions import StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.tools.store import memory_store_impl


class _StubEmbedder:
    def embed(self, text: str) -> list[float] | None:
        return [float(len(text)), 1.0, 0.5]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    def available(self) -> bool:
        return True

    def dim(self) -> int:
        return 3


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


async def test_client_store_upserts_vector_when_embedder_available(client: MemoryClient) -> None:
    """MemoryClient.store persists a vector alongside the entry when possible."""
    real_backend = client._backend
    assert real_backend is not None
    real_backend.close()

    backend = MagicMock()
    client._backend = backend

    with patch.object(MemoryClient, "_get_embedder", return_value=_StubEmbedder()):
        await client.store("vectorized content", detail="with detail")

    assert backend.store.called
    backend.upsert_vector.assert_called_once()
    args = backend.upsert_vector.call_args.args
    assert str(args[0]).startswith("M-")
    assert args[1] == [30.0, 1.0, 0.5]


async def test_client_store_rolls_back_when_vector_upsert_fails(client: MemoryClient) -> None:
    """A vector write failure must not leave the primary row committed.

    Post S1/S3/S9: the row + vector write share a single ``backend.transaction()``
    that ROLLS BACK on any exception — replacing the old compensating
    ``backend.delete()``. We assert the store + upsert run inside the
    transaction context and that the failure surfaces as a rolled-back
    StorageError.
    """
    real_backend = client._backend
    assert real_backend is not None
    real_backend.close()

    backend = MagicMock()
    backend.upsert_vector.side_effect = RuntimeError("dimension mismatch")
    client._backend = backend

    with (
        patch.object(MemoryClient, "_get_embedder", return_value=_StubEmbedder()),
        pytest.raises(StorageError, match="rolled back"),
    ):
        await client.store("vectorized content", detail="with detail")

    backend.store.assert_called_once()
    # Atomicity is now provided by the transaction context, not a manual delete.
    backend.transaction.assert_called_once()
    backend.delete.assert_not_called()


async def test_client_recall_passes_stored_embeddings_to_hybrid_search(client: MemoryClient) -> None:
    """MemoryClient.recall feeds backend vectors into the hybrid pipeline."""
    real_backend = client._backend
    assert real_backend is not None
    real_backend.close()

    entry = MemoryEntry(id="M-001", content="pydantic model", namespace="default")
    backend = MagicMock()
    backend.list_entries.return_value = [entry]
    backend.get_stored_embeddings.return_value = {"M-001": [0.9, 0.1, 0.0]}
    client._backend = backend

    with (
        patch.object(MemoryClient, "_get_embedder", return_value=_StubEmbedder()),
        patch("trw_memory.retrieval.pipeline.hybrid_search", return_value=[entry]) as hybrid_search_mock,
    ):
        await client.recall("pydantic")

    assert hybrid_search_mock.called
    assert hybrid_search_mock.call_args.kwargs["stored_embeddings"] == {"M-001": [0.9, 0.1, 0.0]}


async def test_client_recall_temporal_prefix_uses_stripped_dense_query(client: MemoryClient) -> None:
    """Temporal prefix stripping must apply to BM25, dense, and rerank text."""
    real_backend = client._backend
    assert real_backend is not None
    real_backend.close()

    entry = MemoryEntry(id="M-001", content="pydantic", namespace="default")
    backend = MagicMock()
    backend.list_entries.return_value = [entry]
    backend.get_stored_embeddings.return_value = {"M-001": [0.8, 0.2, 0.0]}
    client._backend = backend
    temporal = SimpleNamespace(
        is_temporal=True,
        recency_weight=0.42,
        confidence=0.9,
        matched_patterns=["latest"],
    )

    with (
        patch.object(MemoryClient, "_get_embedder", return_value=_StubEmbedder()),
        patch("trw_memory.retrieval.temporal_query.classify_temporal", return_value=temporal),
        patch("trw_memory.retrieval.temporal_query.strip_temporal_prefix", return_value="pydantic"),
        patch("trw_memory.retrieval.pipeline.hybrid_search", return_value=[entry]) as hybrid_search_mock,
    ):
        await client.recall("latest guidance on pydantic")

    kwargs = hybrid_search_mock.call_args.kwargs
    assert kwargs["query"] == "pydantic"
    assert kwargs["query_embedding"] == [8.0, 1.0, 0.5]
    assert "rerank_query" not in kwargs


async def test_client_recall_updates_access_metadata_only_for_returned_entries(client: MemoryClient) -> None:
    """Client recall should mark surfaced entries without touching other rows."""
    stored_target = await client.store("target context", importance=0.9)
    stored_other = await client.store("unrelated background", importance=0.2)

    results = await client.recall("target", limit=1)

    backend = client._backend
    assert backend is not None
    touched = backend.get(stored_target["memory_id"])
    untouched = backend.get(stored_other["memory_id"])

    assert results[0]["memory_id"] == stored_target["memory_id"]
    assert touched is not None
    assert touched.access_count == 1
    assert touched.last_accessed_at is not None
    assert untouched is not None
    assert untouched.access_count == 0
    assert untouched.last_accessed_at is None


def test_memory_store_impl_upserts_vector_when_embedder_available() -> None:
    """memory_store_impl persists vectors through the backend when available."""
    backend = MagicMock()
    with patch("trw_memory.tools.store.get_local_embedder", return_value=_StubEmbedder()):
        result = memory_store_impl("stored through tool", "project:default", backend=backend)

    assert result["status"] == "stored"
    backend.store.assert_called_once()
    backend.upsert_vector.assert_called_once()


def test_memory_store_impl_rolls_back_when_vector_upsert_fails() -> None:
    """Tool store reports a transaction rollback on vector failure.

    Post S1/S3/S9: ``memory_store_impl`` wraps ``backend.store()`` +
    ``backend.upsert_vector()`` in a single ``backend.transaction()`` that
    rolls back on any exception — replacing the old compensating
    ``backend.delete()`` path so the tool seam shares one atomicity model
    with ``MemoryClient.store()``. We assert the store + upsert run inside
    the transaction and that the failure surfaces as a rolled-back error
    rather than a manual delete.
    """
    backend = MagicMock()
    backend.upsert_vector.side_effect = RuntimeError("dimension mismatch")

    with patch("trw_memory.tools.store.get_local_embedder", return_value=_StubEmbedder()):
        result = memory_store_impl("stored through tool", "project:default", backend=backend)

    assert result["status"] == "error"
    assert "rolled back" in cast("str", result["error"])
    backend.store.assert_called_once()
    backend.upsert_vector.assert_called_once()
    # Atomicity is now provided by the transaction context, not a manual delete.
    backend.transaction.assert_called_once()
    backend.delete.assert_not_called()


def test_memory_store_impl_uses_configured_embedder_settings() -> None:
    """Tool store must use configured embedding model/dimension, not defaults."""
    backend = MagicMock()
    config = MemoryConfig(embedding_model="custom-model", embedding_dim=768)

    with patch("trw_memory.tools.store.get_local_embedder", return_value=None) as embedder_mock:
        result = memory_store_impl("stored through tool", "project:default", backend=backend, config=config)

    embedder_mock.assert_called_once_with(model_name="custom-model", dim=768)
    assert result["status"] == "stored"


def test_memory_store_impl_ignores_non_entry_backend_get_result() -> None:
    """Tool store should treat mock placeholders as no existing entry."""
    backend = MagicMock()
    backend.get.return_value = MagicMock()

    with patch("trw_memory.tools.store.get_local_embedder", return_value=None):
        result = memory_store_impl("stored through tool", "project:default", backend=backend)

    assert result["status"] == "stored"
    stored_entry = backend.store.call_args.args[0]
    assert isinstance(stored_entry, MemoryEntry)


def test_memory_recall_impl_passes_stored_embeddings_to_hybrid_search() -> None:
    """memory_recall_impl forwards backend vectors into hybrid_search."""
    entry = MemoryEntry(id="M-001", content="pydantic", namespace="project:default")
    backend = MagicMock()
    backend.list_entries.return_value = [entry]
    backend.get_stored_embeddings.return_value = {"M-001": [0.8, 0.2, 0.0]}

    with (
        patch("trw_memory.tools.recall.get_local_embedder", return_value=_StubEmbedder()),
        patch("trw_memory.tools.recall.hybrid_search", return_value=[entry]) as hybrid_search_mock,
    ):
        result = memory_recall_impl("pydantic", "project:default", backend=backend)

    assert "memories" in result
    assert hybrid_search_mock.called
    assert hybrid_search_mock.call_args.kwargs["stored_embeddings"] == {"M-001": [0.8, 0.2, 0.0]}


def test_memory_recall_impl_uses_configured_embedder_settings() -> None:
    """Tool recall must honor configured embedding settings."""
    entry = MemoryEntry(id="M-001", content="pydantic", namespace="project:default")
    backend = MagicMock()
    backend.list_entries.return_value = [entry]
    backend.get_stored_embeddings.return_value = {}
    config = MemoryConfig(embedding_model="custom-model", embedding_dim=768)

    with (
        patch("trw_memory.tools.recall.get_local_embedder", return_value=None) as embedder_mock,
        patch("trw_memory.tools.recall.hybrid_search", return_value=[entry]),
    ):
        result = memory_recall_impl("pydantic", "project:default", backend=backend, config=config)

    embedder_mock.assert_called_once_with(model_name="custom-model", dim=768)
    assert result["memories"][0]["content"] == "pydantic"


def test_memory_recall_impl_forwards_retrieval_config_to_hybrid_search() -> None:
    """Tool recall must use the same retrieval knobs as client recall."""
    entry = MemoryEntry(id="M-001", content="pydantic", namespace="project:default")
    backend = MagicMock()
    backend.list_entries.return_value = [entry]
    backend.get_stored_embeddings.return_value = {"M-001": [0.8, 0.2, 0.0]}
    config = MemoryConfig(
        rrf_k=22,
        rrf_importance_alpha=0.4,
        recall_recency_weight=0.6,
        recall_recency_halflife_days=9.0,
        recall_fusion_mode="combmax",
        recall_validity_age_decay=True,
        recall_rerank=True,
        recall_rerank_model="cross-encoder/custom",
        recall_rerank_candidates=12,
    )

    with (
        patch("trw_memory.tools.recall.get_local_embedder", return_value=_StubEmbedder()),
        patch("trw_memory.tools.recall.hybrid_search", return_value=[entry]) as hybrid_search_mock,
    ):
        memory_recall_impl("pydantic", "project:default", backend=backend, config=config)

    kwargs = hybrid_search_mock.call_args.kwargs
    assert kwargs["rrf_k"] == 22
    assert kwargs["importance_alpha"] == 0.4
    assert kwargs["recency_weight"] == 0.6
    assert kwargs["recency_halflife_days"] == 9.0
    assert kwargs["fusion_mode"] == "combmax"
    assert kwargs["validity_age_decay"] is True
    assert kwargs["rerank"] is True
    assert kwargs["rerank_model"] == "cross-encoder/custom"
    assert kwargs["rerank_candidates"] == 12
    assert kwargs["bm25_candidates"] >= 1
    assert kwargs["vector_candidates"] >= 1


def test_memory_recall_impl_applies_temporal_query_wiring() -> None:
    """Tool recall auto-recency and prefix stripping should match client recall."""
    entry = MemoryEntry(id="M-001", content="pydantic", namespace="project:default")
    backend = MagicMock()
    backend.list_entries.return_value = [entry]
    backend.get_stored_embeddings.return_value = {"M-001": [0.8, 0.2, 0.0]}
    config = MemoryConfig(recall_recency_weight=0.0, recall_auto_temporal=True)
    temporal = SimpleNamespace(
        is_temporal=True,
        recency_weight=0.42,
        confidence=0.9,
        matched_patterns=["latest"],
    )

    with (
        patch("trw_memory.tools.recall.get_local_embedder", return_value=_StubEmbedder()),
        patch("trw_memory.retrieval.temporal_query.classify_temporal", return_value=temporal),
        patch("trw_memory.retrieval.temporal_query.strip_temporal_prefix", return_value="pydantic"),
        patch("trw_memory.tools.recall.hybrid_search", return_value=[entry]) as hybrid_search_mock,
    ):
        memory_recall_impl("latest guidance on pydantic", "project:default", backend=backend, config=config)

    kwargs = hybrid_search_mock.call_args.kwargs
    assert kwargs["query"] == "pydantic"
    assert kwargs["query_embedding"] == [8.0, 1.0, 0.5]
    assert kwargs["recency_weight"] == 0.42
    assert "rerank_query" not in kwargs


def test_memory_recall_impl_updates_access_metadata_only_for_returned_entries(tmp_path: Path) -> None:
    """Tool recall should update surfaced rows after filtering and limit capping."""
    from trw_memory.integrations._backend import create_backend_from_config, make_entry

    config = MemoryConfig(storage_path=str(tmp_path / "storage"))
    target_entry = make_entry("highest utility", namespace="project:default", importance=0.9)
    other_entry = make_entry("lower utility", namespace="project:default", importance=0.1)

    with create_backend_from_config(config, "project:default") as backend:
        backend.store(target_entry)
        backend.store(other_entry)

        result = memory_recall_impl("", "project:default", backend=backend, limit=1, config=config)

        touched = backend.get(target_entry.id)
        untouched = backend.get(other_entry.id)

    memories = cast("list[dict[str, object]]", result["memories"])
    assert result["total_matches"] == 1
    assert memories[0]["id"] == target_entry.id
    assert touched is not None
    assert touched.access_count == 1
    assert touched.last_accessed_at is not None
    assert untouched is not None
    assert untouched.access_count == 0
    assert untouched.last_accessed_at is None


def test_client_get_embedder_uses_configured_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MemoryClient._get_embedder forwards configured model/dim."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("MEMORY_EMBEDDING_DIM", "768")

    client = MemoryClient(namespace="default", mode="local")
    try:
        with patch("trw_memory.embeddings.get_local_embedder", return_value=None) as embedder_mock:
            result = client._get_embedder()
    finally:
        backend = client._backend
        if backend is not None:
            backend.close()

    embedder_mock.assert_called_once_with(model_name="custom-model", dim=768)
    assert result is None


def test_client_get_embedder_reuses_provider_for_client_lifetime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated operations must not reload the sentence-transformers model."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    provider = _StubEmbedder()
    client = MemoryClient(namespace="default", mode="local")
    try:
        with patch("trw_memory.embeddings.get_local_embedder", return_value=provider) as embedder_mock:
            first = client._get_embedder()
            second = client._get_embedder()
    finally:
        backend = client._backend
        if backend is not None:
            backend.close()

    assert first is provider
    assert second is provider
    embedder_mock.assert_called_once_with(
        model_name=client._config.embedding_model,
        dim=client._config.embedding_dim,
    )


def test_client_get_embedder_caches_unavailable_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional provider must not be probed on every operation."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    client = MemoryClient(namespace="default", mode="local")
    try:
        with patch("trw_memory.embeddings.get_local_embedder", return_value=None) as embedder_mock:
            assert client._get_embedder() is None
            assert client._get_embedder() is None
    finally:
        backend = client._backend
        if backend is not None:
            backend.close()

    embedder_mock.assert_called_once()


async def test_client_close_releases_cached_embedder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client owns its provider reference and releases it on close."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    provider = _StubEmbedder()
    client = MemoryClient(namespace="default", mode="local")
    with patch("trw_memory.embeddings.get_local_embedder", return_value=provider):
        assert client._get_embedder() is provider

    await client.close()

    assert client._embedder is None


async def test_tag_filter_widens_top_k_to_full_pool(client: MemoryClient) -> None:
    """A tag-filtered recall must rank the full candidate pool, not just top_k.

    Regression for memory-retrieval-graph-1: the tag filter runs AFTER
    hybrid_search truncates to ``top_k`` (= limit * recall_top_k_multiplier,
    default 30). A tag-matching entry ranked past top_k would be silently
    dropped. When tags are supplied, ``effective_top_k`` must be at least the
    namespace size so the tag filter sees every loaded candidate.
    """
    real_backend = client._backend
    assert real_backend is not None
    real_backend.close()

    # 40 entries — larger than the default top_k of 30 (limit 10 * multiplier 3).
    entries = [
        MemoryEntry(id=f"M-{i:03d}", content=f"content {i}", namespace="default", tags=["wanted"]) for i in range(40)
    ]
    backend = MagicMock()
    backend.list_entries.return_value = entries
    backend.get_stored_embeddings.return_value = {}
    client._backend = backend

    with (
        patch.object(MemoryClient, "_get_embedder", return_value=_StubEmbedder()),
        patch("trw_memory.retrieval.pipeline.hybrid_search", return_value=entries) as hybrid_search_mock,
    ):
        await client.recall("content", tags=["wanted"])

    # Without the fix top_k would be 30 < 40, truncating the pool the tag filter
    # operates on. The fix raises it to the full namespace size.
    assert hybrid_search_mock.call_args.kwargs["top_k"] >= 40


async def test_no_tags_keeps_default_top_k(client: MemoryClient) -> None:
    """Tag-free recall must NOT widen top_k — preserves the configured pool depth."""
    real_backend = client._backend
    assert real_backend is not None
    real_backend.close()

    entries = [MemoryEntry(id=f"M-{i:03d}", content=f"content {i}", namespace="default") for i in range(40)]
    backend = MagicMock()
    backend.list_entries.return_value = entries
    backend.get_stored_embeddings.return_value = {}
    client._backend = backend

    with (
        patch.object(MemoryClient, "_get_embedder", return_value=_StubEmbedder()),
        patch("trw_memory.retrieval.pipeline.hybrid_search", return_value=entries) as hybrid_search_mock,
    ):
        await client.recall("content", limit=10)

    # No tags → unchanged default depth (limit 10 * multiplier 3 = 30).
    assert hybrid_search_mock.call_args.kwargs["top_k"] == 10 * client._config.recall_top_k_multiplier


def _hybrid_recall_event(
    logs: Sequence[MutableMapping[str, Any]],
) -> MutableMapping[str, Any]:
    """Return the single ``hybrid_recall_complete`` event emitted during a recall."""
    matches = [event for event in logs if event.get("event") == "hybrid_recall_complete"]
    assert len(matches) == 1, f"expected exactly one hybrid_recall_complete event, got {matches}"
    return matches[0]


class TestHybridRecallLatencyTelemetry:
    """PRD-DIST-2047 Phase 2: per-recall latency + shape event for operator tuning.

    Asserts that every terminating branch of ``try_hybrid_recall`` emits a
    ``hybrid_recall_complete`` structlog event so operators can right-size
    ``hybrid_search_candidate_pool_size`` against measured cost.
    """

    async def test_emits_telemetry_on_successful_recall(self, client: MemoryClient) -> None:
        """Successful hybrid recall must emit `hybrid_recall_complete` with outcome=ok."""
        real_backend = client._backend
        assert real_backend is not None
        real_backend.close()

        entry = MemoryEntry(id="M-001", content="pydantic model", namespace="default")
        backend = MagicMock()
        backend.list_entries.return_value = [entry]
        backend.get_stored_embeddings.return_value = {"M-001": [0.9, 0.1, 0.0]}
        client._backend = backend

        with (
            patch.object(MemoryClient, "_get_embedder", return_value=_StubEmbedder()),
            patch("trw_memory.retrieval.pipeline.hybrid_search", return_value=[entry]),
            structlog.testing.capture_logs() as logs,
        ):
            await client.recall("pydantic")

        event = _hybrid_recall_event(logs)
        assert event["outcome"] == "ok"
        assert event["namespace"] == "default"
        assert event["namespace_size"] == 1
        assert event["returned_count"] == 1
        # Auto-scaled caps are max(config default, namespace_size); namespace_size=1
        # is below the 50-floor so the floor wins.
        assert event["effective_bm25_candidates"] == max(client._config.bm25_candidates, 1)
        assert event["effective_vector_candidates"] == max(client._config.vector_candidates, 1)
        assert cast("int", event["candidate_pool_size"]) >= client._config.hybrid_search_candidate_pool_size
        # Latencies are non-negative floats rounded to 3 decimals.
        for key in ("list_entries_ms", "hybrid_search_ms", "total_ms"):
            assert isinstance(event[key], float)
            assert cast("float", event[key]) >= 0.0

    async def test_emits_telemetry_on_no_candidates(self, client: MemoryClient) -> None:
        """Empty namespace must emit `hybrid_recall_complete` with outcome=no_candidates."""
        real_backend = client._backend
        assert real_backend is not None
        real_backend.close()

        backend = MagicMock()
        backend.list_entries.return_value = []
        backend.get_stored_embeddings.return_value = {}
        client._backend = backend

        with (
            patch.object(MemoryClient, "_get_embedder", return_value=_StubEmbedder()),
            structlog.testing.capture_logs() as logs,
        ):
            await client.recall("pydantic")

        event = _hybrid_recall_event(logs)
        assert event["outcome"] == "no_candidates"
        assert event["namespace_size"] == 0
        assert event["returned_count"] == 0
        assert event["hybrid_search_ms"] == 0.0
        assert cast("float", event["list_entries_ms"]) >= 0.0
        assert cast("float", event["total_ms"]) >= 0.0

    async def test_emits_telemetry_on_hybrid_search_failure(self, client: MemoryClient) -> None:
        """hybrid_search raising must emit `hybrid_recall_complete` with outcome=hybrid_search_failed."""
        real_backend = client._backend
        assert real_backend is not None
        real_backend.close()

        entry = MemoryEntry(id="M-001", content="pydantic", namespace="default")
        backend = MagicMock()
        backend.list_entries.return_value = [entry]
        backend.get_stored_embeddings.return_value = {"M-001": [0.9, 0.1, 0.0]}
        client._backend = backend

        with (
            patch.object(MemoryClient, "_get_embedder", return_value=_StubEmbedder()),
            patch(
                "trw_memory.retrieval.pipeline.hybrid_search",
                side_effect=RuntimeError("boom"),
            ),
            structlog.testing.capture_logs() as logs,
        ):
            await client.recall("pydantic")

        event = _hybrid_recall_event(logs)
        assert event["outcome"] == "hybrid_search_failed"
        assert event["namespace_size"] == 1
        assert event["returned_count"] == 0
        # hybrid_search was called and took non-negative time even on failure.
        assert cast("float", event["hybrid_search_ms"]) >= 0.0
        assert cast("float", event["total_ms"]) >= 0.0
