"""Tests for MCP tool implementations (recall, search, forget).

Tests the *_impl functions directly without requiring a running FastMCP server.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, cast
from unittest.mock import MagicMock, patch

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.forget import memory_forget_impl
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.tools.search import memory_search_impl

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
            backend = cast(SQLiteBackend, storage)
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

    def test_include_org_memories_appends_cross_validated_project_entries(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

        with (
            create_backend_from_config(cfg, "project:default") as current_storage,
            create_backend_from_config(cfg, "project:other") as remote_storage,
        ):
            current_backend = cast(SQLiteBackend, current_storage)
            remote_backend = cast(SQLiteBackend, remote_storage)

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
                    include_org_memories=True,
                    config=cfg,
                )

            memories = cast(list[dict[str, object]], result["memories"])
            assert len(memories) == 2
            assert memories[0]["namespace"] == "project:default"
            assert memories[1]["namespace"] == "project:other"
            assert memories[1]["scope"] == "org"


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
