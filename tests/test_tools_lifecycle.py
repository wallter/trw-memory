"""Tests for lifecycle MCP tool implementations: store, consolidate, status.

Tests the *_impl functions directly without requiring a running FastMCP server.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from trw_memory.models.memory import MemoryEntry, MemoryStatus
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
        assert stored_entry.metadata == {"source": "test"}

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
