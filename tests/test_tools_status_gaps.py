"""Wave 12: targeted tests for uncovered branches in tools/status.py."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.status import memory_status_impl


def _make_backend() -> SQLiteBackend:
    return SQLiteBackend(Path(":memory:"))


def _ctx_factory(backend):
    @contextmanager
    def _cm(*args, **kwargs):
        yield backend
    return _cm


# ---------------------------------------------------------------------------
# namespace branch — backend.count raises (lines 127-129)
# ---------------------------------------------------------------------------


class TestMemoryStatusImplNamespaceBranchErrors:
    def test_namespace_count_error_returns_error_dict(self) -> None:
        """backend.count() raises → returns error dict."""
        mock_backend = MagicMock()
        mock_backend.count.side_effect = RuntimeError("DB gone")

        result = memory_status_impl("project:default", backend=mock_backend)

        assert result.get("status") == "error"
        assert "storage error" in str(result.get("error"))

    def test_invalid_namespace_returns_invalid(self) -> None:
        """Invalid namespace → validate_namespace raises ConfigError → invalid dict."""
        mock_backend = MagicMock()

        result = memory_status_impl("INVALID!!", backend=mock_backend)

        assert result.get("status") == "invalid"

    def test_valid_namespace_returns_entry_count(self) -> None:
        """Valid namespace → returns total_entries and namespace breakdown."""
        backend = _make_backend()
        try:
            backend.store(MemoryEntry(id="S-001", content="test", namespace="project:default"))
            result = memory_status_impl("project:default", backend=backend)
            assert result.get("total_entries") == 1
            ns = result.get("namespaces", {})
            assert isinstance(ns, dict)
            assert "project:default" in ns
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# no-config branch — individual namespace count fails (lines 143-144)
# ---------------------------------------------------------------------------


class TestMemoryStatusImplNoConfigBranchErrors:
    def test_total_count_error_returns_error(self) -> None:
        """backend.count(namespace=None) raises → returns error dict."""
        mock_backend = MagicMock()
        mock_backend.count.side_effect = RuntimeError("Total count failed")

        result = memory_status_impl(None, backend=mock_backend)

        assert result.get("status") == "error"

    def test_namespace_count_failure_skipped(self) -> None:
        """Per-namespace count raises → that namespace is skipped (best-effort)."""
        mock_backend = MagicMock()
        call_count = 0

        def _flaky_count(namespace=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 5  # total count succeeds
            raise RuntimeError("ns count failed")  # per-namespace fails

        mock_backend.count.side_effect = _flaky_count
        mock_backend.list_entries.return_value = []

        result = memory_status_impl(None, backend=mock_backend)

        # Should still succeed with total_entries
        assert result.get("total_entries") == 5
        # Namespace keys may be partial or absent — no error status
        assert result.get("status") != "error"

    def test_active_count_failure_skipped(self) -> None:
        """list_entries raises → active count skipped (best-effort, no error)."""
        mock_backend = MagicMock()
        mock_backend.count.return_value = 3
        mock_backend.list_entries.side_effect = RuntimeError("list_entries failed")

        result = memory_status_impl(None, backend=mock_backend)

        # Should still succeed without __active__ key
        assert result.get("total_entries") == 3
        assert result.get("status") != "error"

    def test_success_with_real_backend(self) -> None:
        """Integration: real backend with entries returns valid status."""
        backend = _make_backend()
        try:
            backend.store(MemoryEntry(id="S-002", content="hello"))
            result = memory_status_impl(None, backend=backend)
            assert result.get("total_entries", 0) >= 1
            assert "namespaces" in result
            assert "config" in result
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# config-provided branch — discover_namespace_backends fails (lines 173-175)
# ---------------------------------------------------------------------------


class TestMemoryStatusImplConfigBranchErrors:
    def test_discover_backends_error_returns_error(self) -> None:
        """discover_namespace_backends raises → returns error dict."""
        cfg = MemoryConfig()
        mock_backend = MagicMock()

        with patch(
            "trw_memory.tools.status.discover_namespace_backends",
            side_effect=RuntimeError("backends unavailable"),
        ):
            result = memory_status_impl(None, backend=mock_backend, config=cfg)

        assert result.get("status") == "error"
        assert "storage error" in str(result.get("error"))


# ---------------------------------------------------------------------------
# register_status_tool (lines 213-231)
# ---------------------------------------------------------------------------


class TestRegisterStatusTool:
    def test_register_calls_mcp_tool(self) -> None:
        from trw_memory.tools.status import register_status_tool

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda f: f

        register_status_tool(mock_mcp)

        mock_mcp.tool.assert_called_once()

    async def test_registered_function_delegates_to_impl(self) -> None:
        from trw_memory.tools.status import register_status_tool

        registered = {}
        mock_mcp = MagicMock()

        def capture(f):
            registered["fn"] = f
            return f

        mock_mcp.tool.return_value = capture
        register_status_tool(mock_mcp)

        mock_backend = MagicMock()
        mock_backend.count.return_value = 0
        mock_backend.list_entries.return_value = []

        with patch(
            "trw_memory.integrations._backend.create_backend_from_config",
            new=_ctx_factory(mock_backend),
        ):
            result = await registered["fn"](namespace=None)

        assert "total_entries" in result
