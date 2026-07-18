"""Focused clustering tests for lifecycle/consolidation.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from trw_memory.lifecycle.consolidation import _mean_pairwise_similarity, find_clusters

from ._test_consolidation_support import (
    _V1,
    _V2,
    _V3,
    _V_OUTLIER,
    _InMemoryBackend,
    _make_embedder,
    _make_entry,
)


class TestFindClusters:
    def test_returns_empty_when_embedder_none(self) -> None:
        storage = _InMemoryBackend()
        result = find_clusters(storage, None)
        assert result == []

    def test_returns_empty_when_embedder_unavailable(self) -> None:
        storage = _InMemoryBackend()
        embedder = _make_embedder(available=False)
        result = find_clusters(storage, embedder)
        assert result == []

    def test_returns_empty_when_insufficient_entries(self) -> None:
        storage = _InMemoryBackend()
        storage.store(_make_entry("e1"))
        storage.store(_make_entry("e2"))
        embedder = _make_embedder(vectors=[_V1, _V2])
        result = find_clusters(storage, embedder, min_cluster_size=3)
        assert result == []

    def test_detects_cluster_of_similar_entries(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", content=f"content {i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])
        result = find_clusters(storage, embedder, similarity_threshold=0.5, min_cluster_size=3)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_excludes_consolidated_source_entries(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", source="consolidated"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])
        result = find_clusters(storage, embedder, similarity_threshold=0.5, min_cluster_size=3)
        assert result == []

    def test_excludes_already_archived_entries(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", consolidated_into="M-other"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])
        result = find_clusters(storage, embedder, similarity_threshold=0.5, min_cluster_size=3)
        assert result == []

    def test_outlier_not_merged_into_cluster(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", content=f"content {i}"))
        storage.store(_make_entry("outlier", content="outlier"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3, _V_OUTLIER])
        result = find_clusters(storage, embedder, similarity_threshold=0.5, min_cluster_size=3)
        assert len(result) == 1
        cluster_ids = {entry.id for entry in result[0]}
        assert "outlier" not in cluster_ids

    def test_namespace_filter_applied(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", namespace="ns-a"))
        for i in range(3):
            storage.store(_make_entry(f"f{i}", namespace="ns-b"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])
        result = find_clusters(storage, embedder, similarity_threshold=0.5, min_cluster_size=3, namespace="ns-a")
        if result:
            for entry in result[0]:
                assert entry.namespace == "ns-a"

    def test_max_entries_cap(self) -> None:
        storage = _InMemoryBackend()
        for i in range(10):
            storage.store(_make_entry(f"e{i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])
        embedder.embed_batch.return_value = [_V1, _V2, _V3]
        find_clusters(storage, embedder, similarity_threshold=0.5, min_cluster_size=3, max_entries=3)
        call_args = embedder.embed_batch.call_args
        assert call_args is not None
        texts_arg = call_args[0][0]
        assert len(texts_arg) <= 3

    def test_returns_empty_when_all_embeddings_are_none(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}"))
        embedder = _make_embedder(vectors=[None, None, None])

        result = find_clusters(storage, embedder, similarity_threshold=0.5, min_cluster_size=3)

        assert result == []


class TestMeanPairwiseSimilarity:
    def test_single_entry_returns_zero(self) -> None:
        embedder = _make_embedder(vectors=[_V1])
        cluster = [_make_entry("e1")]
        result = _mean_pairwise_similarity(cluster, embedder)
        assert result == 0.0

    def test_two_identical_vectors_returns_one(self) -> None:
        embedder = MagicMock()
        embedder.embed_batch.return_value = [_V1, _V1]
        cluster = [_make_entry("e1"), _make_entry("e2")]
        result = _mean_pairwise_similarity(cluster, embedder)
        assert abs(result - 1.0) < 0.01

    def test_two_orthogonal_vectors_returns_zero(self) -> None:
        v_a = [1.0, 0.0, 0.0, 0.0]
        v_b = [0.0, 1.0, 0.0, 0.0]
        embedder = MagicMock()
        embedder.embed_batch.return_value = [v_a, v_b]
        cluster = [_make_entry("e1"), _make_entry("e2")]
        result = _mean_pairwise_similarity(cluster, embedder)
        assert abs(result) < 0.01

    def test_none_embeddings_filtered(self) -> None:
        embedder = MagicMock()
        embedder.embed_batch.return_value = [_V1, None, _V2]
        cluster = [_make_entry("e1"), _make_entry("e2"), _make_entry("e3")]
        result = _mean_pairwise_similarity(cluster, embedder)
        assert result > 0.0

    def test_all_none_returns_zero(self) -> None:
        embedder = MagicMock()
        embedder.embed_batch.return_value = [None, None]
        cluster = [_make_entry("e1"), _make_entry("e2")]
        result = _mean_pairwise_similarity(cluster, embedder)
        assert result == 0.0
