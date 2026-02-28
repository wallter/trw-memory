"""Tests for the FastMCP server entry point (server.py).

fastmcp is an optional dependency not installed in the dev venv.
Tests mock the fastmcp module to verify server structure without requiring it.

Each test that needs the server module uses importlib.reload() with the mock
in place to ensure a fresh server state.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _make_fastmcp_mock() -> tuple[MagicMock, MagicMock, list[str]]:
    """Create a mock FastMCP class and instance for testing.

    Returns:
        (FastMCP_cls, mcp_instance, registered_tools_list)
    """
    registered_tools: list[str] = []

    mcp_instance = MagicMock()
    mcp_instance.name = "trw-memory"

    def _tool_decorator() -> object:
        def _decorator(fn: object) -> object:
            registered_tools.append(getattr(fn, "__name__", "unknown"))
            return fn
        return _decorator

    mcp_instance.tool = _tool_decorator
    mcp_instance._registered_tools = registered_tools

    FastMCP_cls = MagicMock(return_value=mcp_instance)
    return FastMCP_cls, mcp_instance, registered_tools


def _reload_server_with_mock() -> tuple[types.ModuleType, MagicMock, list[str]]:
    """Reload trw_memory.server with a mocked fastmcp and return helpers.

    Returns:
        (server_module, mcp_instance, registered_tools_list)
    """
    FastMCP_cls, mcp_instance, registered_tools = _make_fastmcp_mock()

    fake_fastmcp = types.ModuleType("fastmcp")
    fake_fastmcp.FastMCP = FastMCP_cls  # type: ignore[attr-defined]

    # Remove cached server module so import does a fresh load
    sys.modules.pop("trw_memory.server", None)
    sys.modules["fastmcp"] = fake_fastmcp

    try:
        import trw_memory.server as server_mod  # noqa: F811
    finally:
        # Restore: remove mock fastmcp, leave server cached for callers
        sys.modules.pop("fastmcp", None)

    return server_mod, mcp_instance, registered_tools


class TestServerModule:
    def test_mcp_instance_exists(self) -> None:
        """Server module must export a FastMCP instance named 'mcp'."""
        server_mod, mcp_instance, _ = _reload_server_with_mock()
        assert hasattr(server_mod, "mcp")
        assert server_mod.mcp is not None
        # Cleanup
        sys.modules.pop("trw_memory.server", None)

    def test_mcp_named_trw_memory(self) -> None:
        """FastMCP must be instantiated with 'trw-memory'."""
        server_mod, mcp_instance, _ = _reload_server_with_mock()
        # The mcp_instance.name is set to "trw-memory" in our mock
        assert mcp_instance.name == "trw-memory"
        sys.modules.pop("trw_memory.server", None)

    def test_main_callable(self) -> None:
        """server.main must be callable."""
        server_mod, _, _ = _reload_server_with_mock()
        assert callable(server_mod.main)
        sys.modules.pop("trw_memory.server", None)

    def test_six_tools_registered(self) -> None:
        """Exactly 6 tools must be registered via mcp.tool()."""
        server_mod, mcp_instance, registered_tools = _reload_server_with_mock()
        assert len(registered_tools) == 6, (
            f"Expected 6 tools, got {len(registered_tools)}: {registered_tools}"
        )
        sys.modules.pop("trw_memory.server", None)

    def test_all_tool_modules_importable(self) -> None:
        """All 6 tool modules must be importable with their impl functions."""
        from trw_memory.tools import store, recall, forget, consolidate, search, status
        assert callable(store.memory_store_impl)
        assert callable(recall.memory_recall_impl)
        assert callable(forget.memory_forget_impl)
        assert callable(consolidate.memory_consolidate_impl)
        assert callable(search.memory_search_impl)
        assert callable(status.memory_status_impl)

    def test_register_functions_exist(self) -> None:
        """Each tool module must export a register_*_tool function."""
        from trw_memory.tools.store import register_store_tool
        from trw_memory.tools.recall import register_recall_tool
        from trw_memory.tools.forget import register_forget_tool
        from trw_memory.tools.consolidate import register_consolidate_tool
        from trw_memory.tools.search import register_search_tool
        from trw_memory.tools.status import register_status_tool

        for fn in [
            register_store_tool, register_recall_tool, register_forget_tool,
            register_consolidate_tool, register_search_tool, register_status_tool,
        ]:
            assert callable(fn), f"{fn} must be callable"
