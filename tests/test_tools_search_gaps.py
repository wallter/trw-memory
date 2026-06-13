"""Wave 14: coverage gap-fill for tools/search.py (lines 62, 64, 82, 145-184)."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from trw_memory.tools.search import memory_search_impl, register_search_tool

from ._test_tools_support import _mock_backend


class TestSearchLimitValidation:
    def test_limit_zero_returns_invalid(self) -> None:
        """limit < 1 → error response (line 62)."""
        backend = _mock_backend()
        result = memory_search_impl("project:default", backend=backend, limit=0)
        assert result["status"] == "invalid"
        assert "limit" in str(result.get("error", ""))

    def test_limit_exceeds_max_returns_invalid(self) -> None:
        """limit > _MAX_SEARCH_LIMIT (500) → error response (line 62)."""
        backend = _mock_backend()
        result = memory_search_impl("project:default", backend=backend, limit=501)
        assert result["status"] == "invalid"
        assert "limit" in str(result.get("error", ""))


class TestSearchOffsetValidation:
    def test_negative_offset_returns_invalid(self) -> None:
        """offset < 0 → error response (line 64)."""
        backend = _mock_backend()
        result = memory_search_impl("project:default", backend=backend, offset=-1)
        assert result["status"] == "invalid"
        assert "offset" in str(result.get("error", ""))


class TestSearchQuarantinedBranch:
    def test_quarantined_status_calls_list_quarantined_entries(self) -> None:
        """status='quarantined' → list_quarantined_entries branch (line 82)."""
        from trw_memory.models.memory import MemoryEntry

        backend = _mock_backend()
        entries = [MemoryEntry(id="M-001", content="quarantined item")]

        with patch(
            "trw_memory.tools.search.list_quarantined_entries",
            return_value=entries,
        ) as mock_list:
            result = memory_search_impl(
                "project:default",
                backend=backend,
                status="quarantined",
            )

        assert mock_list.call_count == 1
        assert isinstance(result.get("entries"), list)


class TestRegisterSearchTool:
    def test_registered_function_delegates_to_impl(self) -> None:
        """register_search_tool wires memory_search to impl (lines 145-184)."""
        registered: dict[str, object] = {}
        mock_mcp = MagicMock()

        def _capture(f):
            registered["fn"] = f
            return f

        mock_mcp.tool.return_value = _capture
        register_search_tool(mock_mcp)

        assert "fn" in registered

        mock_backend = _mock_backend()

        def _ctx_factory(backend: MagicMock):
            @contextmanager
            def _cm(*_a, **_kw):
                yield backend
            return _cm

        with patch(
            "trw_memory.integrations._backend.create_backend_from_config",
            new=_ctx_factory(mock_backend),
        ):
            result = asyncio.run(
                registered["fn"]("project:default")  # type: ignore[operator]
            )

        assert isinstance(result, dict)
        assert "entries" in result
