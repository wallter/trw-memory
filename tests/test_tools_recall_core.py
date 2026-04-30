from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from trw_memory.exceptions import AuthorizationError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.recall import _apply_sec001_recall_policy, memory_recall_impl

from ._test_tools_support import _make_entry, _mock_backend


class TestMemoryRecallImpl:
    def test_apply_sec001_recall_policy_maps_filtered_entries_back_to_results(self) -> None:
        cfg = MemoryConfig(enable_recall_filter=True, recall_filter_mode="observe")

        secured = _apply_sec001_recall_policy(
            [{"id": "M-001", "content": "alpha memory", "namespace": "project:default"}],
            config=cfg,
        )

        assert [entry["id"] for entry in secured] == ["M-001"]

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
        assert backend.list_entries.called or backend.search.called

    def test_global_namespace_valid(self) -> None:
        backend = _mock_backend([_make_entry(namespace="global")])
        result = memory_recall_impl("test", "global", backend=backend)
        assert "memories" in result
        assert "status" not in result or result.get("status") != "invalid"

    def test_min_score_filters_results(self) -> None:
        backend = _mock_backend([_make_entry()])
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

    def test_expired_team_namespace_returns_empty_with_flag_for_yaml(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))

        with create_backend_from_config(cfg, "team:sprint-24") as backend:
            from trw_memory.tools.store import memory_store_impl

            memory_store_impl("team discovery", "team:sprint-24", backend=backend, config=cfg)
            manager = NamespaceManager(backend)
            manager.mark_team_namespace_completed(
                "team:sprint-24",
                completed_at=datetime.now(timezone.utc) - timedelta(days=2),
            )

            result = memory_recall_impl("", "team:sprint-24", backend=backend, config=cfg)

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
        conn.execute("CREATE TABLE memory_graph_edges (source_id TEXT, target_id TEXT, edge_type TEXT, weight REAL)")
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
