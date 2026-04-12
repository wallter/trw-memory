"""Tests for lifecycle/consolidation.py — FR01-FR06."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest

import trw_memory.lifecycle.consolidation as consolidation_module
from trw_memory.exceptions import StorageError
from trw_memory.lifecycle.consolidation import (
    _archive_originals,
    _create_consolidated_entry,
    _mean_pairwise_similarity,
    _redact_paths,
    _summarize_cluster_fallback,
    consolidate_cycle,
    find_clusters,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.yaml_backend import YAMLBackend

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_entry(
    entry_id: str,
    content: str = "content",
    detail: str = "detail",
    importance: float = 0.5,
    tags: list[str] | None = None,
    evidence: list[str] | None = None,
    recurrence: int = 1,
    q_value: float = 0.5,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    source: Literal["human", "agent", "tool", "consolidated"] = "agent",
    consolidated_into: str | None = None,
    namespace: str = "default",
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        importance=importance,
        tags=tags or [],
        evidence=evidence or [],
        recurrence=recurrence,
        q_value=q_value,
        status=status,
        source=source,
        consolidated_into=consolidated_into,
        namespace=namespace,
    )


class _InMemoryBackend(StorageBackend):
    """Simple in-memory StorageBackend for testing."""

    def __init__(self) -> None:
        self._data: dict[str, MemoryEntry] = {}
        self._vectors: dict[str, list[float]] = {}

    def store(self, entry: MemoryEntry) -> None:
        self._data[entry.id] = entry

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._data.get(entry_id)

    def update(self, entry_id: str, **fields: object) -> MemoryEntry | None:
        existing = self._data.get(entry_id)
        if existing is None:
            return None
        data = existing.model_dump()
        for k, v in fields.items():
            if k == "status":
                # Normalize to MemoryStatus enum for model construction
                if isinstance(v, MemoryStatus):
                    data[k] = v
                else:
                    data[k] = MemoryStatus(str(v))
            elif isinstance(v, datetime):
                data[k] = v
            else:
                data[k] = v
        # model_dump() returns status as str (use_enum_values=True),
        # convert back to MemoryStatus for strict model construction
        if "status" in data and isinstance(data["status"], str):
            data["status"] = MemoryStatus(data["status"])
        self._data[entry_id] = MemoryEntry(**data)
        return self._data[entry_id]

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._data:
            del self._data[entry_id]
            self._vectors.pop(entry_id, None)
            return True
        return False

    def search(
        self,
        query: str,
        *,
        top_k: int = 25,
        tags: list[str] | None = None,
        status: MemoryStatus | None = None,
        min_importance: float = 0.0,
        namespace: str | None = None,
    ) -> list[MemoryEntry]:
        results = list(self._data.values())
        if status is not None:
            sv = status.value if isinstance(status, MemoryStatus) else str(status)
            results = [e for e in results if str(e.status) == sv]
        return results[:top_k]

    def count(self, namespace: str | None = None) -> int:
        return len(self._data)

    def list_entries(
        self,
        *,
        status: MemoryStatus | None = None,
        namespace: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        results = list(self._data.values())
        if status is not None:
            sv = status.value if isinstance(status, MemoryStatus) else str(status)
            results = [e for e in results if str(e.status) == sv]
        if namespace is not None:
            results = [e for e in results if e.namespace == namespace]
        return results[:limit]

    def close(self) -> None:
        pass

    def upsert_vector(self, entry_id: str, embedding: list[float]) -> None:
        self._vectors[entry_id] = embedding

    def get_stored_embeddings(self, entry_ids: list[str]) -> dict[str, list[float]]:
        return {entry_id: self._vectors[entry_id] for entry_id in entry_ids if entry_id in self._vectors}


def _make_embedder(
    dim: int = 4,
    available: bool = True,
    vectors: list[list[float] | None] | None = None,
) -> MagicMock:
    """Create a mock EmbeddingProvider."""
    embedder = MagicMock()
    embedder.available.return_value = available
    embedder.dim.return_value = dim
    if vectors is not None:
        embedder.embed_batch.return_value = vectors
        if vectors:
            embedder.embed.return_value = vectors[0]
        else:
            embedder.embed.return_value = None
    else:
        embedder.embed_batch.return_value = []
        embedder.embed.return_value = None
    return embedder


# High-similarity vectors (cosine sim close to 1.0)
_V1 = [1.0, 0.0, 0.0, 0.0]
_V2 = [0.99, 0.1, 0.0, 0.0]
_V3 = [0.98, 0.15, 0.0, 0.0]
_W1 = [0.0, 1.0, 0.0, 0.0]
_W2 = [0.1, 0.99, 0.0, 0.0]
_W3 = [0.15, 0.98, 0.0, 0.0]

# Low-similarity vector
_V_OUTLIER = [0.0, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# _redact_paths
# ---------------------------------------------------------------------------


class TestRedactPaths:
    def test_redacts_unix_home(self) -> None:
        result = _redact_paths("file at /home/user/project/file.py")
        assert "[REDACTED_PATH]" in result
        assert "/home/user" not in result

    def test_redacts_mnt_path(self) -> None:
        result = _redact_paths("stored at /mnt/c/Users/Tyler/project")
        assert "[REDACTED_PATH]" in result

    def test_redacts_windows_path(self) -> None:
        result = _redact_paths(r"path C:\Users\Tyler\project")
        assert "[REDACTED_PATH]" in result

    def test_no_paths_unchanged(self) -> None:
        text = "no filesystem paths here"
        assert _redact_paths(text) == text

    def test_redacts_tmp_path(self) -> None:
        result = _redact_paths("temp file /tmp/abc123")
        assert "[REDACTED_PATH]" in result


# ---------------------------------------------------------------------------
# _summarize_cluster_fallback
# ---------------------------------------------------------------------------


class TestSummarizeClusterFallback:
    def test_selects_longest_content(self) -> None:
        cluster = [
            _make_entry("e1", content="short", detail="x"),
            _make_entry("e2", content="much longer content here", detail="longer detail too"),
            _make_entry("e3", content="medium content", detail="ok"),
        ]
        result = _summarize_cluster_fallback(cluster)
        assert result["summary"] == "much longer content here"
        assert result["detail"] == "longer detail too"

    def test_returns_both_fields(self) -> None:
        cluster = [_make_entry("e1", content="c", detail="d")]
        result = _summarize_cluster_fallback(cluster)
        assert "summary" in result
        assert "detail" in result

    def test_single_entry_cluster(self) -> None:
        cluster = [_make_entry("e1", content="only entry", detail="some detail")]
        result = _summarize_cluster_fallback(cluster)
        assert result["summary"] == "only entry"
        assert result["detail"] == "some detail"

    def test_logs_fallback_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr(consolidation_module, "logger", mock_logger)
        cluster = [
            _make_entry("e1", content="short", detail="x"),
            _make_entry("e2", content="much longer content here", detail="longer detail too"),
        ]

        result = _summarize_cluster_fallback(cluster)

        assert result["summary"] == "much longer content here"
        mock_logger.info.assert_called_once_with(
            "consolidation_llm_fallback",
            cluster_size=2,
            selected_id="e2",
        )


# ---------------------------------------------------------------------------
# _create_consolidated_entry
# ---------------------------------------------------------------------------


class TestCreateConsolidatedEntry:
    def test_creates_entry_with_correct_fields(self) -> None:
        storage = _InMemoryBackend()
        cluster = [
            _make_entry("e1", importance=0.8, tags=["a", "b"], evidence=["ev1"], recurrence=2, q_value=0.7),
            _make_entry("e2", importance=0.6, tags=["b", "c"], evidence=["ev2"], recurrence=3, q_value=0.5),
        ]
        result = _create_consolidated_entry(cluster, "summary text", "detail text", storage)
        assert result.id.startswith("M-")
        assert result.content == "summary text"
        assert result.detail == "detail text"
        assert result.source == "consolidated"
        assert result.importance == 0.8  # max
        assert result.tags == ["a", "b", "c"]  # sorted union
        assert result.evidence == ["ev1", "ev2"]
        assert result.recurrence == 5  # sum
        assert result.q_value == 0.7  # max
        assert set(result.consolidated_from) == {"e1", "e2"}

    def test_entry_persisted_in_storage(self) -> None:
        storage = _InMemoryBackend()
        cluster = [_make_entry("e1"), _make_entry("e2"), _make_entry("e3")]
        result = _create_consolidated_entry(cluster, "c", "d", storage)
        assert storage.get(result.id) is not None

    def test_entry_status_is_active(self) -> None:
        storage = _InMemoryBackend()
        cluster = [_make_entry("e1"), _make_entry("e2")]
        result = _create_consolidated_entry(cluster, "c", "d", storage)
        assert str(result.status) == "active"

    def test_evidence_deduplication(self) -> None:
        storage = _InMemoryBackend()
        cluster = [
            _make_entry("e1", evidence=["ev1", "ev2"]),
            _make_entry("e2", evidence=["ev2", "ev3"]),
        ]
        result = _create_consolidated_entry(cluster, "c", "d", storage)
        # ev2 should appear only once, preserving insertion order
        assert result.evidence == ["ev1", "ev2", "ev3"]

    def test_namespace_passed_correctly(self) -> None:
        storage = _InMemoryBackend()
        cluster = [_make_entry("e1"), _make_entry("e2")]
        result = _create_consolidated_entry(cluster, "c", "d", storage, namespace="team")
        assert result.namespace == "team"

    def test_persists_vector_when_embedder_available(self) -> None:
        storage = _InMemoryBackend()
        cluster = [_make_entry("e1"), _make_entry("e2")]
        embedder = _make_embedder(vectors=[_V1])

        result = _create_consolidated_entry(cluster, "summary", "detail", storage, embedder=embedder)

        assert storage.get_stored_embeddings([result.id])[result.id] == _V1

    def test_rolls_back_entry_when_vector_write_fails(self) -> None:
        storage = _InMemoryBackend()
        cluster = [_make_entry("e1"), _make_entry("e2")]
        embedder = _make_embedder(vectors=[_V1])

        def _fail_vector(_entry_id: str, _embedding: list[float]) -> None:
            raise RuntimeError("vector write failed")

        setattr(storage, "upsert_vector", _fail_vector)

        with pytest.raises(StorageError, match="entry write was rolled back"):
            _create_consolidated_entry(cluster, "summary", "detail", storage, embedder=embedder)

        assert len(storage.list_entries()) == 0


# ---------------------------------------------------------------------------
# _archive_originals
# ---------------------------------------------------------------------------


class TestArchiveOriginals:
    def test_sets_consolidated_into_and_archived(self) -> None:
        storage = _InMemoryBackend()
        e1 = _make_entry("e1")
        e2 = _make_entry("e2")
        storage.store(e1)
        storage.store(e2)

        _archive_originals([e1, e2], "M-consolidated", storage)

        updated_e1 = storage.get("e1")
        updated_e2 = storage.get("e2")
        assert updated_e1 is not None
        assert updated_e2 is not None
        assert updated_e1.consolidated_into == "M-consolidated"
        assert updated_e2.consolidated_into == "M-consolidated"
        assert str(updated_e1.status) == "archived"
        assert str(updated_e2.status) == "archived"

    def test_raises_on_storage_error(self) -> None:
        storage = _InMemoryBackend()
        e1 = _make_entry("e1")
        # Don't store e1 — update on non-existent entry will return None,
        # but let's make update raise instead
        original_update = storage.update

        def _fail_update(entry_id: str, **fields: object) -> MemoryEntry | None:
            raise RuntimeError("storage error")

        storage.update = _fail_update  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            _archive_originals([e1], "M-cons", storage)


# ---------------------------------------------------------------------------
# find_clusters
# ---------------------------------------------------------------------------


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
        # All three are highly similar
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
        # 3 similar + 1 outlier
        embedder = _make_embedder(vectors=[_V1, _V2, _V3, _V_OUTLIER])
        result = find_clusters(storage, embedder, similarity_threshold=0.5, min_cluster_size=3)
        # Should get 1 cluster of 3 (outlier excluded)
        assert len(result) == 1
        cluster_ids = {e.id for e in result[0]}
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
        # Only return 3 vectors even though more entries exist
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])
        embedder.embed_batch.return_value = [_V1, _V2, _V3]
        result = find_clusters(storage, embedder, similarity_threshold=0.5, min_cluster_size=3, max_entries=3)
        # embed_batch called with at most max_entries entries
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


# ---------------------------------------------------------------------------
# _mean_pairwise_similarity
# ---------------------------------------------------------------------------


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
        # Only 2 valid vectors, should compute similarity between them
        assert result > 0.0

    def test_all_none_returns_zero(self) -> None:
        embedder = MagicMock()
        embedder.embed_batch.return_value = [None, None]
        cluster = [_make_entry("e1"), _make_entry("e2")]
        result = _mean_pairwise_similarity(cluster, embedder)
        assert result == 0.0


# ---------------------------------------------------------------------------
# consolidate_cycle
# ---------------------------------------------------------------------------


class TestConsolidateCycle:
    def test_dry_run_returns_clusters_no_writes(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", content=f"content {i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])

        result = consolidate_cycle(storage, embedder, dry_run=True, config=MemoryConfig())

        assert result["dry_run"] is True
        assert result["consolidated_count"] == 0
        assert result["skipped_reason"] == "dry_run"
        # No new entries created
        assert storage.count() == 3

    def test_dry_run_preserves_yaml_file_mtimes(self, tmp_path: Path) -> None:
        entries_dir = tmp_path / "entries"
        storage = YAMLBackend(entries_dir)
        for i in range(3):
            storage.store(_make_entry(f"e{i}", content=f"content {i}", detail=f"detail {i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])
        before = {path.name: path.stat().st_mtime_ns for path in entries_dir.glob("*.yaml")}

        result = consolidate_cycle(
            storage,
            embedder,
            dry_run=True,
            config=MemoryConfig(
                consolidation_similarity_threshold=0.5,
                consolidation_min_cluster=3,
            ),
        )
        after = {path.name: path.stat().st_mtime_ns for path in entries_dir.glob("*.yaml")}

        assert result["dry_run"] is True
        assert before == after

    def test_no_clusters_returns_no_clusters_status(self) -> None:
        storage = _InMemoryBackend()
        # Embedder unavailable — no clusters
        result = consolidate_cycle(storage, None, config=MemoryConfig())
        assert result["status"] == "no_clusters"
        assert result["clusters_found"] == 0
        assert result["consolidated_count"] == 0

    def test_consolidation_creates_entry_and_archives_originals(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", content=f"content {i}", detail=f"detail {i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])

        result = consolidate_cycle(
            storage,
            embedder,
            config=MemoryConfig(
                consolidation_similarity_threshold=0.5,
                consolidation_min_cluster=3,
            ),
        )

        assert result["status"] == "completed"
        assert result["consolidated_count"] == 1

        # Originals should be archived
        for i in range(3):
            entry = storage.get(f"e{i}")
            assert entry is not None
            assert str(entry.status) == "archived"
            assert entry.consolidated_into is not None

        # New consolidated entry exists
        all_entries = storage.list_entries()
        consolidated = [e for e in all_entries if e.source == "consolidated"]
        assert len(consolidated) == 1
        assert consolidated[0].id in storage.get_stored_embeddings([consolidated[0].id])

    def test_dry_run_includes_cluster_preview(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", content=f"content {i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])
        # embed_batch used for dry-run mean similarity too
        embedder.embed_batch.return_value = [_V1, _V2, _V3]

        result = consolidate_cycle(
            storage,
            embedder,
            dry_run=True,
            config=MemoryConfig(
                consolidation_similarity_threshold=0.5,
                consolidation_min_cluster=3,
            ),
        )
        clusters = result.get("clusters")
        assert isinstance(clusters, list)
        if clusters:
            cluster = clusters[0]
            assert "entry_ids" in cluster
            assert "count" in cluster
            assert "mean_similarity" in cluster

    def test_consolidation_uses_namespace(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", namespace="my-ns"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])

        result = consolidate_cycle(
            storage,
            embedder,
            namespace="my-ns",
            config=MemoryConfig(
                consolidation_similarity_threshold=0.5,
                consolidation_min_cluster=3,
            ),
        )
        assert result["consolidated_count"] == 1
        all_entries = storage.list_entries(namespace="my-ns")
        consolidated = [e for e in all_entries if e.source == "consolidated"]
        assert consolidated[0].namespace == "my-ns"

    def test_empty_storage_returns_no_clusters(self) -> None:
        storage = _InMemoryBackend()
        embedder = _make_embedder(vectors=[])
        result = consolidate_cycle(storage, embedder, config=MemoryConfig())
        assert result["clusters_found"] == 0
        assert result["consolidated_count"] == 0

    def test_error_in_cluster_recorded_and_rolled_back(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])

        # Make _archive_originals fail by breaking update
        original_update = storage.update

        call_count = [0]

        def _failing_update(entry_id: str, **fields: object) -> MemoryEntry | None:
            call_count[0] += 1
            if call_count[0] > 0:
                raise RuntimeError("simulated failure")
            return original_update(entry_id, **fields)

        storage.update = _failing_update  # type: ignore[method-assign]

        result = consolidate_cycle(
            storage,
            embedder,
            config=MemoryConfig(
                consolidation_similarity_threshold=0.5,
                consolidation_min_cluster=3,
            ),
        )
        # Should have recorded errors but not crashed
        assert result.get("errors") is not None
        consolidated = [entry for entry in storage.list_entries() if entry.source == "consolidated"]
        assert consolidated == []
        for i in range(3):
            entry = storage.get(f"e{i}")
            assert entry is not None
            assert str(entry.status) == "active"
            assert entry.consolidated_into is None

    def test_rollback_failure_raises_storage_error(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])

        original_delete = storage.delete
        delete_calls = [0]

        def _failing_update(entry_id: str, **fields: object) -> MemoryEntry | None:
            raise RuntimeError("archive failed")

        def _failing_delete(entry_id: str) -> bool:
            delete_calls[0] += 1
            if delete_calls[0] == 1:
                return False
            return original_delete(entry_id)

        storage.update = _failing_update  # type: ignore[method-assign]
        storage.delete = _failing_delete  # type: ignore[method-assign]

        with pytest.raises(StorageError, match="rollback failed"):
            consolidate_cycle(
                storage,
                embedder,
                config=MemoryConfig(
                    consolidation_similarity_threshold=0.5,
                    consolidation_min_cluster=3,
                ),
            )

    def test_disabled_config_skips_writes(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", content=f"content {i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])

        result = consolidate_cycle(
            storage,
            embedder,
            config=MemoryConfig(consolidation_enabled=False),
        )

        assert result["status"] == "disabled"
        assert result["skipped_reason"] == "consolidation_disabled"
        assert result["consolidated_count"] == 0
        assert len(storage.list_entries()) == 3

    def test_config_max_per_cycle_caps_scanned_entries(self) -> None:
        storage = _InMemoryBackend()
        for i in range(10):
            storage.store(_make_entry(f"e{i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])

        consolidate_cycle(
            storage,
            embedder,
            max_entries=10,
            dry_run=True,
            config=MemoryConfig(consolidation_max_per_cycle=3),
        )

        call_args = embedder.embed_batch.call_args
        assert call_args is not None
        texts_arg = call_args[0][0]
        assert len(texts_arg) <= 3

    def test_second_cycle_processes_zero_clusters(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", content=f"content {i}", detail=f"detail {i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])
        config = MemoryConfig(
            consolidation_similarity_threshold=0.5,
            consolidation_min_cluster=3,
        )

        first = consolidate_cycle(storage, embedder, config=config)
        second = consolidate_cycle(storage, embedder, config=config)

        assert first["consolidated_count"] == 1
        assert second["clusters_found"] == 0
        assert second["consolidated_count"] == 0

    def test_fallback_runs_for_each_detected_cluster(self) -> None:
        storage = _InMemoryBackend()
        for i in range(6):
            storage.store(_make_entry(f"e{i}", content=f"content {i}", detail=f"detail {i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3, _W1, _W2, _W3])

        result = consolidate_cycle(
            storage,
            embedder,
            config=MemoryConfig(
                consolidation_similarity_threshold=0.5,
                consolidation_min_cluster=3,
            ),
        )

        consolidated = [entry for entry in storage.list_entries() if entry.source == "consolidated"]
        assert result["clusters_found"] == 2
        assert result["consolidated_count"] == 2
        assert len(consolidated) == 2

    def test_config_defaults_used_when_none(self) -> None:
        storage = _InMemoryBackend()
        # No embedder means no clusters — but should not crash
        result = consolidate_cycle(storage, None, config=None)
        assert "status" in result or "dry_run" in result or "clusters_found" in result
