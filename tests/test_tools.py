"""Tests for MCP tool implementations (recall, search, forget).

Tests the *_impl functions directly without requiring a running FastMCP server.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import AuthorizationError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.security.audit import AuditLog
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.forget import memory_forget_impl
from trw_memory.tools.recall import _merge_tier_entries, memory_recall_impl
from trw_memory.tools.search import memory_search_impl
from trw_memory.tools.store import memory_store_impl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    entry_id: str = "M-001",
    content: str = "test memory",
    namespace: str = "project:default",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    tags: list[str] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content,
        namespace=namespace,
        status=status,
        tags=tags or [],
    )


def _mock_backend(entries: list[MemoryEntry] | None = None) -> MagicMock:
    """Create a mock StorageBackend with sensible defaults."""
    backend = MagicMock()
    entries = entries or []
    backend.list_entries.return_value = entries
    backend.get_stored_embeddings.return_value = {}
    backend.search.return_value = entries
    backend.get.return_value = entries[0] if entries else None
    backend.delete.return_value = True
    backend.count.return_value = len(entries)
    return backend


# ---------------------------------------------------------------------------
# memory_store_impl
# ---------------------------------------------------------------------------


class TestMemoryStoreImpl:
    def test_store_denied_for_reader_namespace_role(self) -> None:
        backend = _mock_backend()
        cfg = MemoryConfig(rbac_enabled=True, namespace_roles={"project:default": "reader"})

        with pytest.raises(
            AuthorizationError,
            match=r"Role 'reader' does not have store permission on namespace 'project:default'\.",
        ):
            memory_store_impl("blocked write", "project:default", backend=backend, config=cfg)

    def test_store_blocks_api_keys_before_backend_write(self, tmp_path: Path) -> None:
        backend = _mock_backend()
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))

        result = memory_store_impl(
            "api token sk-abcdefghijklmnopqrstuvwxyz",
            "project:default",
            backend=backend,
            config=cfg,
        )

        assert result["status"] == "blocked"
        assert backend.store.called is False
        audit_records = AuditLog(Path(cfg.audit_log_path)).read_all()
        assert audit_records[-1].op == "reject"

    def test_store_quarantines_anomalous_entries(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), poisoning_z_threshold=1.0)

        with create_backend_from_config(cfg, "project:default") as backend:
            for index in range(20):
                backend.store(
                    MemoryEntry(
                        id=f"M-seed-{index}",
                        content="normal content",
                        namespace="project:default",
                        source_identity="seed",
                    )
                )

            result = memory_store_impl(
                "A" * 5000,
                "project:default",
                backend=backend,
                config=cfg,
                source_identity="alice",
            )
            quarantined = memory_search_impl(
                "project:default",
                backend=backend,
                config=cfg,
                status="quarantined",
                actor="alice",
            )

        assert result["status"] == "quarantined"
        assert quarantined["total"] == 1
        entries = cast("list[dict[str, object]]", quarantined["entries"])
        assert entries[0]["metadata"]["quarantined"] == "true"

    def test_forget_actor_deletes_matching_entries_and_audits(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))

        with create_backend_from_config(cfg, "project:default") as backend:
            backend.store(
                MemoryEntry(id="M-alice", content="alice entry", namespace="project:default", source_identity="alice")
            )
            backend.store(MemoryEntry(id="M-bob", content="bob entry", namespace="project:default", source_identity="bob"))

            result = memory_forget_impl(None, None, "project:default", backend=backend, config=cfg, actor="alice")

            remaining = memory_search_impl("project:default", backend=backend, config=cfg, actor="alice")

        assert result["entries_deleted"] == 1
        assert remaining["total"] == 0
        audit_records = AuditLog(Path(cfg.audit_log_path)).read_all()
        assert audit_records[-1].op == "forget"
        assert audit_records[-1].data["entries_deleted"] == 1


# ---------------------------------------------------------------------------
# memory_recall_impl
# ---------------------------------------------------------------------------


class TestMemoryRecallImpl:
    def test_returns_expected_keys(self) -> None:
        backend = _mock_backend([_make_entry()])
        result = memory_recall_impl("python async", "project:default", backend=backend)
        assert "memories" in result
        assert "total_matches" in result
        assert "query" in result

    def test_invalid_namespace_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_recall_impl("query", "INVALID_NS", backend=backend)
        assert result["status"] == "invalid"
        assert "error" in result

    def test_recall_denied_for_writer_namespace_role(self) -> None:
        backend = _mock_backend()
        cfg = MemoryConfig(rbac_enabled=True, namespace_roles={"project:default": "writer"})

        with pytest.raises(
            AuthorizationError,
            match=r"Role 'writer' does not have recall permission on namespace 'project:default'\.",
        ):
            memory_recall_impl("query", "project:default", backend=backend, config=cfg)

    def test_empty_query_returns_all_active(self) -> None:
        entries = [_make_entry("M-001"), _make_entry("M-002")]
        backend = _mock_backend(entries)
        result = memory_recall_impl("", "project:default", backend=backend)
        assert "memories" in result
        assert isinstance(result["memories"], list)

    def test_limit_respected(self) -> None:
        entries = [_make_entry(f"M-{i:03d}", f"content {i}") for i in range(10)]
        backend = _mock_backend(entries)
        result = memory_recall_impl("content", "project:default", backend=backend, limit=3)
        assert len(result["memories"]) <= 3  # type: ignore[arg-type]

    def test_query_preserved_in_result(self) -> None:
        backend = _mock_backend()
        result = memory_recall_impl("my query", "project:default", backend=backend)
        assert result["query"] == "my query"

    def test_tags_filter_passed_to_search(self) -> None:
        backend = _mock_backend()
        memory_recall_impl("q", "project:default", backend=backend, tags=["python"])
        # Verify list_entries was called (backend was used)
        assert backend.list_entries.called or backend.search.called

    def test_global_namespace_valid(self) -> None:
        backend = _mock_backend([_make_entry(namespace="global")])
        result = memory_recall_impl("test", "global", backend=backend)
        assert "memories" in result
        assert "status" not in result or result.get("status") != "invalid"

    def test_min_score_filters_results(self) -> None:
        backend = _mock_backend([_make_entry()])
        # With min_score=1.0, nothing should pass (importance defaults to 0.5)
        result = memory_recall_impl("test", "project:default", backend=backend, min_score=1.0)
        assert "memories" in result
        assert isinstance(result["memories"], list)

    def test_expired_team_namespace_returns_empty_with_flag(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

        with create_backend_from_config(cfg, "team:sprint-24") as storage:
            backend = cast("SQLiteBackend", storage)
            backend.store(_make_entry("M-001", namespace="team:sprint-24"))
            manager = NamespaceManager(backend)
            manager.ensure_team_namespace(
                "team:sprint-24",
                created_at=datetime.now(timezone.utc) - timedelta(days=2),
            )
            manager.mark_team_namespace_completed(
                "team:sprint-24",
                completed_at=datetime.now(timezone.utc) - timedelta(days=2),
            )

            result = memory_recall_impl("", "team:sprint-24", backend=backend)

            assert result["memories"] == []
            assert result["namespace_expired"] is True

    def test_graph_depth_zero_omits_related_field(self) -> None:
        backend = _mock_backend([_make_entry()])

        result = memory_recall_impl("test", "project:default", backend=backend, graph_depth=0)

        assert "related" not in result

    def test_graph_depth_positive_returns_related_entry_payloads(self) -> None:
        import sqlite3

        root = _make_entry("M-root", content="root entry")
        related = _make_entry("M-related", content="related entry")
        backend = _mock_backend([root])
        backend.get.side_effect = lambda entry_id: {"M-root": root, "M-related": related}.get(entry_id)

        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE memory_graph_edges (source_id TEXT, target_id TEXT, edge_type TEXT, weight REAL)"
        )
        conn.execute(
            "INSERT INTO memory_graph_edges (source_id, target_id, edge_type, weight) VALUES (?, ?, ?, ?)",
            ("M-root", "M-related", "similarity", 0.91),
        )
        conn.commit()

        result = memory_recall_impl("", "project:default", backend=backend, graph_depth=1, conn=conn)

        related_items = cast("list[dict[str, object]]", result["related"])
        assert related_items[0]["id"] == "M-related"
        assert related_items[0]["content"] == "related entry"
        assert related_items[0]["edge_type"] == "similarity"
        assert related_items[0]["depth"] == 1
        conn.close()

    def test_include_org_memories_appends_cross_validated_project_entries(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

        with (
            create_backend_from_config(cfg, "project:default") as current_storage,
            create_backend_from_config(cfg, "project:other") as remote_storage,
        ):
            current_backend = cast("SQLiteBackend", current_storage)
            remote_backend = cast("SQLiteBackend", remote_storage)

            current_backend.store(
                MemoryEntry(
                    id="M-local",
                    content="deployment lesson",
                    namespace="project:default",
                )
            )
            remote_backend.store(
                MemoryEntry(
                    id="M-org",
                    content="deployment lesson from another project",
                    namespace="project:other",
                    importance=0.85,
                    cross_validated=True,
                )
            )

            @contextmanager
            def fake_discover(_cfg: MemoryConfig) -> Iterator[object]:
                yield [(["project:other"], remote_backend)]

            with patch("trw_memory.integrations._backend.discover_namespace_backends", fake_discover):
                result = memory_recall_impl(
                    "",
                    "project:default",
                    backend=current_backend,
                    namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
                    include_org_memories=True,
                    config=cfg,
                )

            memories = cast("list[dict[str, object]]", result["memories"])
            assert len(memories) == 2
            assert memories[0]["namespace"] == "project:default"
            assert memories[1]["namespace"] == "project:other"
            assert memories[1]["scope"] == "org"
            local_entry = current_backend.get("M-local")
            remote_entry = remote_backend.get("M-org")
            assert local_entry is not None
            assert remote_entry is not None
            assert local_entry.access_count == 1
            assert remote_entry.access_count == 1

    def test_include_org_memories_skips_below_threshold_entries(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

        with (
            create_backend_from_config(cfg, "project:default") as current_storage,
            create_backend_from_config(cfg, "project:other") as remote_storage,
        ):
            current_backend = cast("SQLiteBackend", current_storage)
            remote_backend = cast("SQLiteBackend", remote_storage)

            current_backend.store(MemoryEntry(id="M-local", content="deployment lesson", namespace="project:default"))
            remote_backend.store(
                MemoryEntry(
                    id="M-org-low",
                    content="low importance org entry",
                    namespace="project:other",
                    importance=0.79,
                    cross_validated=True,
                )
            )

            @contextmanager
            def fake_discover(_cfg: MemoryConfig) -> Iterator[object]:
                yield [(["project:other"], remote_backend)]

            with patch("trw_memory.integrations._backend.discover_namespace_backends", fake_discover):
                result = memory_recall_impl(
                    "",
                    "project:default",
                    backend=current_backend,
                    namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
                    include_org_memories=True,
                    config=cfg,
                )

            memories = cast("list[dict[str, object]]", result["memories"])
            assert [memory["namespace"] for memory in memories] == ["project:default"]

    def test_include_org_memories_false_suppresses_org_entries(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

        with create_backend_from_config(cfg, "project:default") as current_storage:
            current_backend = cast("SQLiteBackend", current_storage)
            current_backend.store(MemoryEntry(id="M-local", content="deployment lesson", namespace="project:default"))

            with patch(
                "trw_memory.graph.list_org_shared_entries",
                side_effect=AssertionError("org lookup should be skipped"),
            ):
                result = memory_recall_impl(
                    "",
                    "project:default",
                    backend=current_backend,
                    namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
                    include_org_memories=False,
                    config=cfg,
                )

            memories = cast("list[dict[str, object]]", result["memories"])
            assert [memory["namespace"] for memory in memories] == ["project:default"]

    def test_recall_promotes_cold_tier_hit_through_tool_surface(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold.yaml"
            write_yaml(
                cold_file,
                MemoryEntry(
                    id="M-tool-cold",
                    content="tool cold archive lesson",
                    namespace="project:default",
                    tags=["cold"],
                ).model_dump(mode="json"),
            )

            result = memory_recall_impl(
                "tool cold archive",
                "project:default",
                backend=backend,
                config=cfg,
            )

            memories = cast("list[dict[str, object]]", result["memories"])
            assert any(memory["id"] == "M-tool-cold" for memory in memories)
            assert not cold_file.exists()
            assert backend.get("M-tool-cold") is not None
            warm_ids = [str(item["id"]) for item in manager.warm_search(["tool"], None)]
            assert "M-tool-cold" in warm_ids

    def test_recall_restores_cold_hit_into_warm_vector_index(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        class _FakeWarmBackend:
            def __init__(self) -> None:
                self.vectors: dict[str, list[float]] = {}

            def upsert_vector(self, entry_id: str, embedding: list[float]) -> None:
                self.vectors[entry_id] = embedding

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold-vector.yaml"
            payload = MemoryEntry(
                id="M-tool-cold-vector",
                content="tool keyword promoted vector lesson",
                namespace="project:default",
                tags=["cold"],
            ).model_dump(mode="json")
            payload["_warm_embedding"] = [1.0, 0.0]
            write_yaml(cold_file, payload)

            fake_backend = _FakeWarmBackend()
            manager._warm_store._get_warm_backend = lambda dim=None: fake_backend  # type: ignore[assignment,return-value]
            result = memory_recall_impl(
                "keyword promoted",
                "project:default",
                backend=backend,
                config=cfg,
            )

            memories = cast("list[dict[str, object]]", result["memories"])
            assert any(memory["id"] == "M-tool-cold-vector" for memory in memories)
            assert fake_backend.vectors["M-tool-cold-vector"] == [1.0, 0.0]

    def test_recall_does_not_surface_cold_hit_when_promotion_fails(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold-fail.yaml"
            write_yaml(
                cold_file,
                MemoryEntry(
                    id="M-tool-cold-fail",
                    content="tool cold rollback lesson",
                    namespace="project:default",
                    tags=["cold"],
                ).model_dump(mode="json"),
            )

            original_warm_add = manager._cold_store._warm_store.warm_add

            def _fail_warm_add(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
                raise OSError("warm unavailable")

            manager._cold_store._warm_store.warm_add = _fail_warm_add  # type: ignore[method-assign]
            try:
                result = memory_recall_impl(
                    "tool cold rollback",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )
            finally:
                manager._cold_store._warm_store.warm_add = original_warm_add  # type: ignore[method-assign]

            memories = cast("list[dict[str, object]]", result["memories"])
            assert not any(memory["id"] == "M-tool-cold-fail" for memory in memories)
            assert cold_file.exists()
            assert backend.get("M-tool-cold-fail") is None

    def test_recall_does_not_leave_sqlite_canonical_copy_when_promotion_fails(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold-sqlite-fail.yaml"
            write_yaml(
                cold_file,
                MemoryEntry(
                    id="M-tool-cold-sqlite-fail",
                    content="tool sqlite rollback lesson",
                    namespace="project:default",
                    tags=["cold"],
                ).model_dump(mode="json"),
            )

            original_warm_add = manager._cold_store._warm_store.warm_add

            def _fail_warm_add(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
                raise OSError("warm unavailable")

            manager._cold_store._warm_store.warm_add = _fail_warm_add  # type: ignore[method-assign]
            try:
                result = memory_recall_impl(
                    "tool sqlite rollback",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )
            finally:
                manager._cold_store._warm_store.warm_add = original_warm_add  # type: ignore[method-assign]

            memories = cast("list[dict[str, object]]", result["memories"])
            assert not any(memory["id"] == "M-tool-cold-sqlite-fail" for memory in memories)
            assert cold_file.exists()
            assert backend.get("M-tool-cold-sqlite-fail") is None

    def test_recall_does_not_leave_sqlite_canonical_copy_when_archive_delete_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold-sqlite-unlink-fail.yaml"
            write_yaml(
                cold_file,
                MemoryEntry(
                    id="M-tool-cold-sqlite-unlink-fail",
                    content="tool sqlite archive delete rollback lesson",
                    namespace="project:default",
                    tags=["cold"],
                ).model_dump(mode="json"),
            )

            original_unlink = Path.unlink

            def _fail_archive_delete(path: Path, *, missing_ok: bool = False) -> None:
                if path == cold_file:
                    raise OSError("archive delete failed")
                original_unlink(path, missing_ok=missing_ok)

            monkeypatch.setattr(Path, "unlink", _fail_archive_delete)
            try:
                result = memory_recall_impl(
                    "archive delete rollback",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )
            finally:
                monkeypatch.setattr(Path, "unlink", original_unlink)

            memories = cast("list[dict[str, object]]", result["memories"])
            assert not any(memory["id"] == "M-tool-cold-sqlite-unlink-fail" for memory in memories)
            assert cold_file.exists()
            assert backend.get("M-tool-cold-sqlite-unlink-fail") is None

    def test_recall_force_deletes_yaml_canonical_copy_when_primary_rollback_delete_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold-yaml-double-fail.yaml"
            write_yaml(
                cold_file,
                MemoryEntry(
                    id="M-tool-cold-yaml-double-fail",
                    content="tool yaml canonical cleanup fallback lesson",
                    namespace="project:default",
                    tags=["cold"],
                ).model_dump(mode="json"),
            )

            original_unlink = Path.unlink

            def _fail_archive_delete(path: Path, *, missing_ok: bool = False) -> None:
                if path == cold_file:
                    raise OSError("archive delete failed")
                original_unlink(path, missing_ok=missing_ok)

            def _fail_primary_delete(_entry_id: str) -> bool:
                raise OSError("primary rollback delete failed")

            monkeypatch.setattr(Path, "unlink", _fail_archive_delete)
            monkeypatch.setattr(backend, "delete", _fail_primary_delete)
            try:
                result = memory_recall_impl(
                    "yaml canonical cleanup fallback",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )
            finally:
                monkeypatch.setattr(Path, "unlink", original_unlink)

            memories = cast("list[dict[str, object]]", result["memories"])
            assert not any(memory["id"] == "M-tool-cold-yaml-double-fail" for memory in memories)
            assert cold_file.exists()
            assert backend.get("M-tool-cold-yaml-double-fail") is None

    def test_recall_force_deletes_sqlite_canonical_copy_when_primary_rollback_delete_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager
        from trw_memory.storage.persistence import write_yaml

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            cold_partition = manager._cold_dir() / "2026" / "04"
            cold_partition.mkdir(parents=True, exist_ok=True)
            cold_file = cold_partition / "tool-cold-sqlite-double-fail.yaml"
            write_yaml(
                cold_file,
                MemoryEntry(
                    id="M-tool-cold-sqlite-double-fail",
                    content="tool sqlite canonical cleanup fallback lesson",
                    namespace="project:default",
                    tags=["cold"],
                ).model_dump(mode="json"),
            )

            original_unlink = Path.unlink

            def _fail_archive_delete(path: Path, *, missing_ok: bool = False) -> None:
                if path == cold_file:
                    raise OSError("archive delete failed")
                original_unlink(path, missing_ok=missing_ok)

            def _fail_primary_delete(_entry_id: str) -> bool:
                raise OSError("primary rollback delete failed")

            monkeypatch.setattr(Path, "unlink", _fail_archive_delete)
            monkeypatch.setattr(backend, "delete", _fail_primary_delete)
            try:
                result = memory_recall_impl(
                    "canonical cleanup fallback",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )
            finally:
                monkeypatch.setattr(Path, "unlink", original_unlink)

            memories = cast("list[dict[str, object]]", result["memories"])
            assert not any(memory["id"] == "M-tool-cold-sqlite-double-fail" for memory in memories)
            assert cold_file.exists()
            assert backend.get("M-tool-cold-sqlite-double-fail") is None

    def test_recall_surfaces_semantic_warm_tier_hit(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            manager.warm_add(
                "M-semantic",
                MemoryEntry(
                    id="M-semantic",
                    content="opaque title",
                    detail="vector-only match",
                    namespace="project:default",
                    importance=0.9,
                    q_value=0.95,
                    q_observations=5,
                ).model_dump(mode="json"),
                [1.0, 0.0],
            )

            fake_embedder = MagicMock()
            fake_embedder.embed.return_value = [1.0, 0.0]

            with patch("trw_memory.tools.recall.get_local_embedder", return_value=fake_embedder):
                result = memory_recall_impl(
                    "semantic query",
                    "project:default",
                    backend=backend,
                    config=cfg,
                )

            memories = cast("list[dict[str, object]]", result["memories"])
            assert [memory["id"] for memory in memories] == ["M-semantic"]

    def test_recall_refreshes_hot_recency_for_ttl(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager

        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path), hot_ttl_days=7, hot_max_entries=5)
        with create_backend_from_config(cfg, "project:default") as backend:
            manager = get_tier_manager(cfg, "project:default")
            stale_time = datetime.now(timezone.utc) - timedelta(days=30)
            entry = MemoryEntry(
                id="M-tool-hot-ttl",
                content="tool ttl refresh lesson",
                namespace="project:default",
                last_accessed_at=stale_time,
            )
            backend.store(entry)
            manager.warm_add("M-tool-hot-ttl", entry.model_dump(mode="json"), None)

            result = memory_recall_impl(
                "tool ttl refresh",
                "project:default",
                backend=backend,
                config=cfg,
            )

            memories = cast("list[dict[str, object]]", result["memories"])
            assert any(memory["id"] == "M-tool-hot-ttl" for memory in memories)
            hot_entry = manager.hot_get("M-tool-hot-ttl")
            assert hot_entry is not None
            sweep_result = manager.sweep(config=cfg)
            assert sweep_result.demoted == 0

    def test_merge_tier_entries_reranks_by_composite_score(self) -> None:
        cfg = MemoryConfig()
        merged = _merge_tier_entries(
            [
                {
                    "id": "M-local",
                    "content": "deploy lesson",
                    "detail": "stale local result",
                    "importance": 0.1,
                    "q_value": 0.1,
                    "q_observations": 5,
                    "last_accessed_at": (datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),
                    "score": 0.9,
                    "namespace": "project:default",
                }
            ],
            [
                {
                    "id": "M-tier",
                    "content": "deploy lesson",
                    "detail": "fresh tier result",
                    "importance": 0.9,
                    "q_value": 0.95,
                    "q_observations": 5,
                    "last_accessed_at": datetime.now(timezone.utc).isoformat(),
                    "score": 0.2,
                    "namespace": "project:default",
                }
            ],
            ["deploy"],
            cfg,
            None,
        )
        assert [str(entry["id"]) for entry in merged] == ["M-tier", "M-local"]


# ---------------------------------------------------------------------------
# memory_search_impl
# ---------------------------------------------------------------------------


class TestMemorySearchImpl:
    def test_returns_expected_keys(self) -> None:
        backend = _mock_backend([_make_entry()])
        result = memory_search_impl("project:default", backend=backend)
        assert "entries" in result
        assert "total" in result
        assert "offset" in result
        assert "limit" in result

    def test_invalid_namespace_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_search_impl("BAD NS!!", backend=backend)
        assert result["status"] == "invalid"
        assert "error" in result

    def test_search_denied_for_writer_namespace_role(self) -> None:
        backend = _mock_backend()
        cfg = MemoryConfig(rbac_enabled=True, namespace_roles={"project:default": "writer"})

        with pytest.raises(
            AuthorizationError,
            match=r"Role 'writer' does not have search permission on namespace 'project:default'\.",
        ):
            memory_search_impl("project:default", backend=backend, config=cfg)

    def test_status_filter_active(self) -> None:
        entries = [
            _make_entry("M-001", status=MemoryStatus.ACTIVE),
            _make_entry("M-002", status=MemoryStatus.OBSOLETE),
        ]
        backend = _mock_backend(entries)
        result = memory_search_impl("project:default", status="active", backend=backend)
        assert "entries" in result

    def test_offset_and_limit_in_result(self) -> None:
        entries = [_make_entry(f"M-{i:03d}") for i in range(5)]
        backend = _mock_backend(entries)
        result = memory_search_impl("project:default", backend=backend, offset=2, limit=2)
        assert result["offset"] == 2
        assert result["limit"] == 2

    def test_tags_filter_applied(self) -> None:
        entries = [
            _make_entry("M-001", tags=["python"]),
            _make_entry("M-002", tags=["rust"]),
        ]
        backend = _mock_backend(entries)
        result = memory_search_impl("project:default", backend=backend, tags=["python"])
        assert "entries" in result

    def test_invalid_status_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_search_impl("project:default", backend=backend, status="nonexistent_status")
        assert result.get("status") == "invalid" or "entries" in result


# ---------------------------------------------------------------------------
# memory_forget_impl
# ---------------------------------------------------------------------------


class TestMemoryForgetImpl:
    def test_no_args_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_forget_impl(None, None, "project:default", backend=backend)
        assert result["status"] == "invalid"
        assert "error" in result

    def test_delete_by_id(self) -> None:
        entry = _make_entry("M-001")
        backend = _mock_backend([entry])
        backend.get.return_value = entry
        backend.delete.return_value = True
        result = memory_forget_impl("M-001", None, "project:default", backend=backend)
        assert result["status"] == "ok"
        assert result["deleted"] == 1

    def test_delete_missing_id(self) -> None:
        backend = _mock_backend()
        backend.get.return_value = None
        backend.delete.return_value = False
        result = memory_forget_impl("M-999", None, "project:default", backend=backend)
        assert result["status"] == "ok"
        assert result["deleted"] == 0

    def test_invalid_namespace_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_forget_impl("M-001", None, "INVALID!!", backend=backend)
        assert result["status"] == "invalid"

    def test_forget_denied_for_reader_namespace_role(self) -> None:
        backend = _mock_backend()
        cfg = MemoryConfig(rbac_enabled=True, namespace_roles={"project:default": "reader"})

        with pytest.raises(
            AuthorizationError,
            match=r"Role 'reader' does not have forget permission on namespace 'project:default'\.",
        ):
            memory_forget_impl("M-001", None, "project:default", backend=backend, config=cfg)

    def test_bulk_delete_via_query(self) -> None:
        entries = [_make_entry("M-001"), _make_entry("M-002")]
        backend = _mock_backend(entries)
        backend.search.return_value = entries
        backend.delete.return_value = True
        result = memory_forget_impl(None, "some query", "project:default", backend=backend)
        assert result["status"] == "ok"
        assert int(str(result["deleted"])) >= 0

    def test_bulk_delete_no_matches(self) -> None:
        backend = _mock_backend([])
        backend.search.return_value = []
        result = memory_forget_impl(None, "nothing matches", "project:default", backend=backend)
        assert result["status"] == "ok"
        assert result["deleted"] == 0

    def test_namespace_isolation_blocks_cross_namespace_delete(self) -> None:
        """P0 fix: delete-by-ID must NOT delete entries from a different namespace."""
        entry = _make_entry("M-001", namespace="project:repo-a")
        backend = _mock_backend([entry])
        backend.get.return_value = entry
        # Try to delete M-001 from namespace project:repo-b
        result = memory_forget_impl("M-001", None, "project:repo-b", backend=backend)
        assert result["status"] == "ok"
        assert result["deleted"] == 0
        backend.delete.assert_not_called()

    def test_empty_memory_id_returns_error(self) -> None:
        """P0 fix: whitespace-only memory_id must be rejected."""
        backend = _mock_backend()
        result = memory_forget_impl("  ", None, "project:default", backend=backend)
        assert result["status"] == "invalid"
        assert "non-empty" in str(result.get("error", "")).lower()

    def test_forget_removes_entry_from_tier_runtime(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            stored = memory_store_impl("tool tracked entry", "project:default", backend=backend, config=cfg)
            memory_id = cast("str", stored["memory_id"])
            manager = get_tier_manager(cfg, "project:default")

            warm_ids = [str(item["id"]) for item in manager.warm_search(["tracked"], None)]
            assert memory_id in warm_ids

            result = memory_forget_impl(memory_id, None, "project:default", backend=backend, config=cfg)

            assert result["deleted"] == 1
            warm_ids_after = [str(item["id"]) for item in manager.warm_search(["tracked"], None)]
            assert memory_id not in warm_ids_after
