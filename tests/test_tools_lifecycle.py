"""Tests for lifecycle MCP tool implementations: store, consolidate, status.

Tests the *_impl functions directly without requiring a running FastMCP server.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

from trw_memory.graph import wait_for_graph_updates
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.consolidate import memory_consolidate_impl
from trw_memory.tools.status import memory_status_impl
from trw_memory.tools.store import memory_store_impl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    entry_id: str = "M-001",
    content: str = "test memory",
    namespace: str = "project:default",
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content,
        namespace=namespace,
        status=status,
    )


def _mock_backend(entries: list[MemoryEntry] | None = None) -> MagicMock:
    backend = MagicMock()
    entries = entries or []
    backend.list_entries.return_value = entries
    backend.get_stored_embeddings.return_value = {}
    backend.search.return_value = entries
    backend.get.return_value = entries[0] if entries else None
    backend.delete.return_value = True
    backend.count.return_value = len(entries)
    backend.store.return_value = None
    backend.update.return_value = entries[0] if entries else None
    return backend


# ---------------------------------------------------------------------------
# memory_store_impl
# ---------------------------------------------------------------------------


class TestMemoryStoreImpl:
    def test_returns_memory_id_on_success(self) -> None:
        backend = _mock_backend()
        result = memory_store_impl("learned something", "project:default", backend=backend)
        assert "memory_id" in result
        assert str(result["memory_id"]).startswith("M-")
        assert result["status"] == "stored"
        assert result["namespace"] == "project:default"

    def test_empty_content_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_store_impl("", "project:default", backend=backend)
        assert result["status"] == "invalid"
        assert "error" in result

    def test_whitespace_only_content_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_store_impl("   ", "project:default", backend=backend)
        assert result["status"] == "invalid"
        assert "error" in result

    def test_invalid_namespace_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_store_impl("some content", "INVALID!!", backend=backend)
        assert result["status"] == "invalid"
        assert "error" in result

    def test_global_namespace_valid(self) -> None:
        backend = _mock_backend()
        result = memory_store_impl("global tip", "global", backend=backend)
        assert result["status"] == "stored"
        assert result["namespace"] == "global"

    def test_tags_passed_to_entry(self) -> None:
        backend = _mock_backend()
        result = memory_store_impl("tagged content", "project:default", backend=backend, tags=["python", "async"])
        assert result["status"] == "stored"
        # Verify store was called
        assert backend.store.called
        stored_entry: MemoryEntry = backend.store.call_args[0][0]
        assert "python" in stored_entry.tags
        assert "async" in stored_entry.tags

    def test_importance_passed_to_entry(self) -> None:
        backend = _mock_backend()
        memory_store_impl("important tip", "project:default", backend=backend, importance=0.9)
        stored_entry: MemoryEntry = backend.store.call_args[0][0]
        assert stored_entry.importance == 0.9

    def test_importance_out_of_range_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_store_impl("content", "project:default", backend=backend, importance=1.5)
        assert result["status"] == "invalid"
        assert "error" in result

    def test_detail_passed_to_entry(self) -> None:
        backend = _mock_backend()
        memory_store_impl("tip", "project:default", backend=backend, detail="extended explanation")
        stored_entry: MemoryEntry = backend.store.call_args[0][0]
        assert stored_entry.detail == "extended explanation"

    def test_metadata_passed_to_entry(self) -> None:
        backend = _mock_backend()
        memory_store_impl("meta tip", "project:default", backend=backend, metadata={"source": "test"})
        stored_entry: MemoryEntry = backend.store.call_args[0][0]
        assert stored_entry.metadata["source"] == "test"
        assert stored_entry.metadata["provenance_author"] == "tool"
        assert stored_entry.metadata["provenance_content_hash"]
        assert stored_entry.metadata["provenance_signature"]

    def test_storage_error_returns_error_dict(self) -> None:
        backend = _mock_backend()
        backend.store.side_effect = RuntimeError("disk full")
        result = memory_store_impl("content", "project:default", backend=backend)
        assert result["status"] == "error"
        assert "error" in result

    def test_content_is_stripped(self) -> None:
        backend = _mock_backend()
        memory_store_impl("  padded content  ", "project:default", backend=backend)
        stored_entry: MemoryEntry = backend.store.call_args[0][0]
        assert stored_entry.content == "padded content"

    def test_sqlite_store_populates_similarity_and_tag_edges(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path), embedding_dim=4)

        with create_backend_from_config(cfg, "project:default") as storage:
            backend = cast("SQLiteBackend", storage)
            backend.store(
                MemoryEntry(
                    id="M-existing",
                    content="existing memory",
                    namespace="project:default",
                    tags=["python", "async", "sqlite"],
                )
            )
            backend.upsert_vector("M-existing", [1.0, 0.0, 0.0, 0.0], namespace="default")

            fake_embedder = MagicMock()
            fake_embedder.embed.return_value = [1.0, 0.0, 0.0, 0.0]

            with patch("trw_memory.tools.store.get_local_embedder", return_value=fake_embedder):
                result = memory_store_impl(
                    "new memory",
                    "project:default",
                    backend=backend,
                    tags=["python", "async", "graph"],
                    config=cfg,
                )
                wait_for_graph_updates()

            assert result["status"] == "stored"
            edge_rows = backend._conn.execute(
                "SELECT edge_type, COUNT(*) FROM memory_graph_edges GROUP BY edge_type ORDER BY edge_type"
            ).fetchall()
            # PRD-CORE-245 FR07: tag_cooccurrence is derived, never written.
            expected = [("similarity", 2)] if backend._vec_available else []
            assert [tuple(row) for row in edge_rows] == expected

    def test_team_namespace_store_registers_lifecycle_row(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path), embedding_dim=4)

        with create_backend_from_config(cfg, "team:sprint-37") as storage:
            backend = cast("SQLiteBackend", storage)
            with patch("trw_memory.tools.store.get_local_embedder", return_value=None):
                result = memory_store_impl(
                    "team discovery",
                    "team:sprint-37",
                    backend=backend,
                    tags=["team"],
                    config=cfg,
                )

            assert result["status"] == "stored"
            row = backend._conn.execute(
                "SELECT team_id, expires_at, status FROM memory_namespaces WHERE namespace_id = ?",
                ("team:sprint-37",),
            ).fetchone()
            assert tuple(row) == ("sprint-37", None, "active")

    def test_store_cross_validates_matching_project_entries(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path), embedding_dim=4)

        with (
            create_backend_from_config(cfg, "project:default") as current_storage,
            create_backend_from_config(cfg, "project:other") as remote_storage,
        ):
            current_backend = cast("SQLiteBackend", current_storage)
            remote_backend = cast("SQLiteBackend", remote_storage)

            remote_backend.store(
                MemoryEntry(
                    id="M-remote",
                    content="shared operational lesson",
                    namespace="project:other",
                    importance=0.6,
                )
            )
            remote_vectors: dict[str, list[float]] = {"M-remote": [1.0, 0.0, 0.0, 0.0]}

            fake_embedder = MagicMock()
            fake_embedder.embed.return_value = [1.0, 0.0, 0.0, 0.0]

            @contextmanager
            def fake_discover(_cfg: MemoryConfig) -> Iterator[object]:
                yield [(["project:other"], remote_backend)]

            with (
                patch.object(remote_backend, "get_stored_embeddings", return_value=remote_vectors),
                patch("trw_memory.tools.store.get_local_embedder", return_value=fake_embedder),
                patch("trw_memory.integrations._backend.discover_namespace_backends", fake_discover),
            ):
                result = memory_store_impl(
                    "shared operational lesson",
                    "project:default",
                    backend=current_backend,
                    config=cfg,
                    importance=0.6,
                )
                wait_for_graph_updates()

            assert result["status"] == "stored"
            stored_entry = current_backend.get(cast("str", result["memory_id"]), namespace="project:default")
            remote_entry = remote_backend.get("M-remote", namespace="project:other")
            assert stored_entry is not None
            assert remote_entry is not None
            assert stored_entry.cross_validated is True
            assert remote_entry.cross_validated is True
            assert stored_entry.importance == 0.65
            assert remote_entry.importance == 0.65
            assert any("cross_validated:project_id=other" in item for item in stored_entry.outcome_history)
            assert any("cross_validated:project_id=default" in item for item in remote_entry.outcome_history)

    def test_store_primes_tier_runtime(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path), embedding_dim=4)
        with create_backend_from_config(cfg, "project:default") as storage:
            backend = cast("SQLiteBackend", storage)
            with patch("trw_memory.tools.store.get_local_embedder", return_value=None):
                result = memory_store_impl(
                    "tier primed entry",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )

            manager = get_tier_manager(cfg, "project:default")
            warm_ids = [str(item["id"]) for item in manager.warm_search(["tier"], None)]
            assert result["status"] == "stored"
            assert cast("str", result["memory_id"]) in warm_ids


# ---------------------------------------------------------------------------
# memory_consolidate_impl
# ---------------------------------------------------------------------------


class TestMemoryConsolidateImpl:
    def test_dry_run_returns_plan_dict(self) -> None:
        backend = _mock_backend()
        result = memory_consolidate_impl("project:default", backend=backend, dry_run=True)
        assert "clusters_found" in result
        assert "entries_consolidated" in result
        assert "dry_run" in result
        assert result["dry_run"] is True
        assert "clusters" in result
        assert result["skipped_reason"] == "dry_run"

    def test_invalid_namespace_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_consolidate_impl("INVALID!!", backend=backend)
        assert result["status"] == "invalid"
        assert "error" in result

    def test_no_clusters_returns_zero(self) -> None:
        backend = _mock_backend()
        # No embedder → no clusters
        result = memory_consolidate_impl("project:default", backend=backend)
        assert result["clusters_found"] == 0
        assert result["entries_consolidated"] == 0

    def test_global_namespace_valid(self) -> None:
        backend = _mock_backend()
        result = memory_consolidate_impl("global", backend=backend, dry_run=True)
        assert "clusters_found" in result
        assert "dry_run" in result

    def test_entries_consolidated_key_present(self) -> None:
        backend = _mock_backend()
        result = memory_consolidate_impl("project:default", backend=backend)
        assert "entries_consolidated" in result
        assert isinstance(result["entries_consolidated"], int)

    @patch("trw_memory.tools.consolidate.consolidate_cycle")
    @patch("trw_memory.tools.consolidate.get_local_embedder")
    def test_disabled_result_passes_through_status(
        self,
        mock_get_local_embedder: MagicMock,
        mock_consolidate_cycle: MagicMock,
    ) -> None:
        backend = _mock_backend()
        mock_get_local_embedder.return_value = MagicMock()
        mock_consolidate_cycle.return_value = {
            "status": "disabled",
            "clusters_found": 0,
            "consolidated_count": 0,
            "skipped_reason": "consolidation_disabled",
        }

        result = memory_consolidate_impl("project:default", backend=backend)

        assert result["status"] == "disabled"
        assert result["skipped_reason"] == "consolidation_disabled"

    @patch("trw_memory.tools.consolidate.consolidate_cycle")
    @patch("trw_memory.tools.consolidate.get_local_embedder")
    def test_resolves_embedder_from_config(
        self,
        mock_get_local_embedder: MagicMock,
        mock_consolidate_cycle: MagicMock,
    ) -> None:
        backend = _mock_backend()
        fake_embedder = MagicMock()
        mock_get_local_embedder.return_value = fake_embedder
        mock_consolidate_cycle.return_value = {
            "clusters_found": 0,
            "consolidated_count": 0,
            "dry_run": False,
        }

        memory_consolidate_impl("project:default", backend=backend)

        kwargs = mock_consolidate_cycle.call_args.kwargs
        assert kwargs["embedder"] is fake_embedder
        cfg = kwargs["config"]
        assert isinstance(cfg, MemoryConfig)
        mock_get_local_embedder.assert_called_once_with(
            model_name=cfg.embedding_model,
            dim=cfg.embedding_dim,
        )


# ---------------------------------------------------------------------------
# memory_status_impl
# ---------------------------------------------------------------------------


class TestMemoryStatusImpl:
    def test_returns_total_entries_key(self) -> None:
        backend = _mock_backend([_make_entry()])
        result = memory_status_impl(None, backend=backend)
        assert "total_entries" in result
        assert result["total_entries"] == 1

    def test_returns_namespaces_and_config_keys(self) -> None:
        backend = _mock_backend()
        result = memory_status_impl(None, backend=backend)
        assert "namespaces" in result
        assert "config" in result
        assert isinstance(result["namespaces"], dict)
        assert isinstance(result["config"], dict)

    def test_config_contains_storage_backend(self) -> None:
        backend = _mock_backend()
        result = memory_status_impl(None, backend=backend)
        config = result["config"]
        assert isinstance(config, dict)
        assert "storage_backend" in config

    def test_invalid_namespace_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_status_impl("INVALID!!", backend=backend)
        assert result["status"] == "invalid"
        assert "error" in result

    def test_namespace_scoped_count(self) -> None:
        backend = _mock_backend()
        backend.count.return_value = 5
        result = memory_status_impl("project:default", backend=backend)
        assert result["total_entries"] == 5
        # count called with namespace
        backend.count.assert_called_with(namespace="project:default")

    def test_global_namespace_valid(self) -> None:
        backend = _mock_backend()
        result = memory_status_impl("global", backend=backend)
        assert "total_entries" in result
        assert "error" not in result

    def test_storage_error_returns_error_dict(self) -> None:
        backend = _mock_backend()
        backend.count.side_effect = RuntimeError("db gone")
        result = memory_status_impl(None, backend=backend)
        assert result["status"] == "error"
        assert "error" in result

    def test_dedup_and_consolidation_in_config(self) -> None:
        backend = _mock_backend()
        result = memory_status_impl(None, backend=backend)
        config = result["config"]
        assert isinstance(config, dict)
        assert "dedup_enabled" in config
        assert "consolidation_enabled" in config

    def test_degraded_security_posture_is_compactly_reported(self) -> None:
        backend = _mock_backend()
        cfg = MemoryConfig(enable_recall_filter=False, provenance_required=False)
        with patch("trw_memory.tools.status.list_quarantined_entries", return_value=[]):
            result = memory_status_impl("project:default", backend=backend, config=cfg)

        posture = result["security_posture"]
        assert isinstance(posture, dict)
        assert posture["status"] == "degraded"
        assert posture["recall_filter_mode"] == "disabled"
        assert posture["provenance_mode"] == "optional"

    def test_positive_quarantine_count_reports_security_posture(self) -> None:
        backend = _mock_backend()
        cfg = MemoryConfig()
        with patch("trw_memory.tools.status.list_quarantined_entries", return_value=[_make_entry()]):
            result = memory_status_impl("project:default", backend=backend, config=cfg)

        posture = result["security_posture"]
        assert isinstance(posture, dict)
        assert posture["status"] == "degraded"
        assert posture["quarantine_count"] == 1

    def test_quarantine_count_failure_reports_degraded_posture(self) -> None:
        backend = _mock_backend()
        cfg = MemoryConfig()
        with patch("trw_memory.tools.status.list_quarantined_entries", side_effect=RuntimeError("quarantine db down")):
            result = memory_status_impl("project:default", backend=backend, config=cfg)

        posture = result["security_posture"]
        assert isinstance(posture, dict)
        assert posture["status"] == "degraded"
        assert posture["quarantine_count"] is None

    def test_status_introspection_lists_live_recovery_and_security_config(self) -> None:
        backend = _mock_backend()
        result = memory_status_impl(None, backend=backend)

        introspection = result["introspection"]
        assert isinstance(introspection, dict)
        assert introspection["tool"] == "memory_status"
        assert "memory_recovery_inline_max_bytes" in introspection["recovery_config_fields"]
        assert "security_maintenance_inline" in introspection["security_config_fields"]
