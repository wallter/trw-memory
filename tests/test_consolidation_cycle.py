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
        assert result["clusters_found"] == len(result["clusters"]) == 1
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
            entry = storage.get(f"e{i}", namespace="default")
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
            entry = storage.get(f"e{i}", namespace="default")
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

    def test_namespace_none_on_multitenant_store_raises(self) -> None:
        """memory-lifecycle-4: refuse the cross-tenant consolidation path.

        With namespace=None on a store that owns >1 namespace, find_clusters
        would mix entries across tenants and the consolidated entry would land
        in a single namespace. Guard rejects this before any write.
        """

        class _MultiTenantBackend(_InMemoryBackend):
            def list_namespaces(self) -> list[str]:
                return ["tenant-a", "tenant-b"]

        storage = _MultiTenantBackend()
        # Seed entries in two tenants so a leak would actually be possible.
        for i in range(3):
            storage.store(_make_entry(f"a{i}", namespace="tenant-a"))
            storage.store(_make_entry(f"b{i}", namespace="tenant-b"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])

        with pytest.raises(ValueError, match="namespace required for consolidate_cycle"):
            consolidate_cycle(storage, embedder, config=MemoryConfig())

        # Nothing was clustered or relocated.
        consolidated = [e for e in storage.list_entries() if e.source == "consolidated"]
        assert consolidated == []

    def test_namespace_none_on_single_tenant_store_allowed(self) -> None:
        """The guard must NOT block the legitimate single-namespace path."""

        class _SingleTenantBackend(_InMemoryBackend):
            def list_namespaces(self) -> list[str]:
                return ["default"]

        storage = _SingleTenantBackend()
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
        assert result["consolidated_count"] == 1

    def test_namespace_none_fails_closed_when_list_namespaces_raises(self) -> None:
        """A namespace-enumeration failure must fail closed, not fall through.

        If list_namespaces() raises we cannot prove the store is single-tenant,
        so namespace=None must refuse rather than silently clustering across
        tenants into "default".
        """

        class _BrokenEnumBackend(_InMemoryBackend):
            def list_namespaces(self) -> list[str]:
                raise RuntimeError("transient enumeration failure")

        storage = _BrokenEnumBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}", content=f"content {i}", detail=f"detail {i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])

        with pytest.raises(ValueError, match="could not enumerate"):
            consolidate_cycle(storage, embedder, config=MemoryConfig())

        # Fail-closed: nothing was clustered or relocated.
        consolidated = [e for e in storage.list_entries() if e.source == "consolidated"]
        assert consolidated == []

    @pytest.mark.parametrize("delete_raises", [False, True])
    def test_partial_archival_failure_restores_originals_when_delete_fails(self, delete_raises: bool) -> None:
        """memory-lifecycle-7: rollback must restore originals even if the

        consolidated-entry delete itself fails. On a YAML backend the per-entry
        archival transaction() is a no-op, so when archival fails mid-loop some
        originals are already archived. The old rollback raised on the failed
        delete BEFORE reinstating those originals, stranding them in the
        archived state. The fix restores originals first, then surfaces the
        delete failure — so the originals are always ACTIVE again.
        """
        import contextlib
        from collections.abc import Iterator

        from trw_memory.storage.interface import StorageBackend

        class _NoOpTxnBackend(_InMemoryBackend):
            """Models a YAML backend whose transaction() does NOT roll back."""

            @contextlib.contextmanager
            def transaction(self) -> Iterator[StorageBackend]:
                yield self

        storage = _NoOpTxnBackend()
        for i in range(3):
            storage.store(_make_entry(f"e{i}"))
        embedder = _make_embedder(vectors=[_V1, _V2, _V3])

        def _real_update(entry_id: str, fields: dict[str, object]) -> MemoryEntry | None:
            # Apply the update without re-entering update_override (which would
            # recurse). Mirrors _InMemoryBackend.update's non-override branch.
            prev, storage.update_override = storage.update_override, None
            try:
                return storage.update(entry_id, **fields, namespace="default")
            finally:
                storage.update_override = prev

        def _update_then_fail(entry_id: str, fields: dict[str, object]) -> MemoryEntry | None:
            # e0 archives successfully (no-op transaction => persisted now);
            # any other original's archival raises mid-loop.
            if entry_id == "e0":
                return _real_update(entry_id, fields)
            raise RuntimeError("archive failed mid-loop")

        def _delete_always_fails(entry_id: str) -> bool:
            # Both a false return and an exception must happen only after the
            # originals are restored.
            del entry_id
            if delete_raises:
                raise StorageError("delete exploded")
            return False

        storage.update_override = _update_then_fail
        storage.delete_override = _delete_always_fails

        # consolidate_cycle re-raises a failed rollback as a StorageError.
        with pytest.raises(StorageError, match="rollback failed"):
            consolidate_cycle(
                storage,
                embedder,
                config=MemoryConfig(
                    consolidation_similarity_threshold=0.5,
                    consolidation_min_cluster=3,
                ),
            )

        # Despite the failed delete, every original is restored to ACTIVE with
        # consolidated_into cleared — including the one archived before failure.
        for i in range(3):
            entry = storage.get(f"e{i}", namespace="default")
            assert entry is not None, f"e{i} was lost"
            assert str(entry.status) == "active", f"e{i} left archived after rollback"
            assert entry.consolidated_into is None
