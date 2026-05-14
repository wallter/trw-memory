"""Focused consolidate-cycle tests for lifecycle/consolidation.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.lifecycle.consolidation import consolidate_cycle
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.yaml_backend import YAMLBackend

from ._test_consolidation_support import (
    _V1,
    _V2,
    _V3,
    _W1,
    _W2,
    _W3,
    _InMemoryBackend,
    _make_embedder,
    _make_entry,
)


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

        for i in range(3):
            entry = storage.get(f"e{i}")
            assert entry is not None
            assert str(entry.status) == "archived"
            assert entry.consolidated_into is not None

        all_entries = storage.list_entries()
        consolidated = [entry for entry in all_entries if entry.source == "consolidated"]
        assert len(consolidated) == 1
        assert consolidated[0].id in storage.get_stored_embeddings([consolidated[0].id])

    def test_dry_run_includes_cluster_preview(self) -> None:
        storage = _InMemoryBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", content=f"content {i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])
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
        consolidated = [entry for entry in all_entries if entry.source == "consolidated"]
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
        original_update = storage.update
        call_count = [0]

        def _failing_update(entry_id: str, fields: dict[str, object]) -> MemoryEntry | None:
            call_count[0] += 1
            del fields
            if call_count[0] > 0:
                raise RuntimeError("simulated failure")
            return original_update(entry_id)

        storage.update_override = _failing_update

        result = consolidate_cycle(
            storage,
            embedder,
            config=MemoryConfig(
                consolidation_similarity_threshold=0.5,
                consolidation_min_cluster=3,
            ),
        )
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

        def _failing_update(entry_id: str, fields: dict[str, object]) -> MemoryEntry | None:
            del entry_id, fields
            raise RuntimeError("archive failed")

        def _failing_delete(entry_id: str) -> bool:
            delete_calls[0] += 1
            if delete_calls[0] == 1:
                return False
            return original_delete(entry_id)

        storage.update_override = _failing_update
        storage.delete_override = _failing_delete

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
        result = consolidate_cycle(storage, None, config=None)
        assert "status" in result or "dry_run" in result or "clusters_found" in result
