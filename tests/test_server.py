"""Tests for the FastMCP server entry point (server.py).

fastmcp is an optional dependency not installed in the dev venv.
Tests mock the fastmcp module to verify server structure without requiring it.

Each test that needs the server module uses importlib.reload() with the mock
in place to ensure a fresh server state.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import EncryptionUnavailableError, LocalOnlyViolationError


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
        import trw_memory.server as server_mod
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

    def test_main_preflights_local_only_embedder(self) -> None:
        """Local-only startup should fail before mcp.run when embeddings require download."""
        server_mod, mcp_instance, _ = _reload_server_with_mock()
        with (
            patch(
                "trw_memory.models.config.MemoryConfig",
                return_value=SimpleNamespace(
                    encryption_enabled=False,
                    local_only=True,
                    embedding_model="test-model",
                    embedding_dim=384,
                ),
            ),
            patch("trw_memory.embeddings.get_local_embedder", side_effect=LocalOnlyViolationError("blocked")),
            pytest.raises(LocalOnlyViolationError, match="blocked"),
        ):
            server_mod.main()
        mcp_instance.run.assert_not_called()
        sys.modules.pop("trw_memory.server", None)

    def test_main_preflights_sqlcipher_when_encryption_enabled(self) -> None:
        """Encrypted startup should fail before mcp.run when SQLCipher is unavailable."""
        server_mod, mcp_instance, _ = _reload_server_with_mock()
        with (
            patch(
                "trw_memory.models.config.MemoryConfig",
                return_value=SimpleNamespace(
                    encryption_enabled=True,
                    local_only=False,
                    embedding_model="test-model",
                    embedding_dim=384,
                ),
            ),
            patch(
                "trw_memory.storage.sqlite_backend._import_sqlcipher_driver",
                side_effect=EncryptionUnavailableError("sqlcipher missing"),
            ),
            pytest.raises(EncryptionUnavailableError, match="sqlcipher missing"),
        ):
            server_mod.main()
        mcp_instance.run.assert_not_called()
        sys.modules.pop("trw_memory.server", None)

    def test_eight_tools_registered(self) -> None:
        """All expected MCP tools must be registered via mcp.tool().

        Asserts the exact set (not a magic count) so additions/removals are
        caught explicitly. The code-index + wiki-lint tools were added after
        the original 8.
        """
        server_mod, mcp_instance, registered_tools = _reload_server_with_mock()
        expected = {
            "memory_store", "memory_recall", "memory_audit", "memory_review",
            "memory_forget", "memory_consolidate", "memory_search", "memory_status",
            "memory_wiki_lint", "memory_code_index", "memory_code_search", "memory_code_symbol",
        }
        assert set(registered_tools) == expected, (
            f"registered tool set drift: {sorted(set(registered_tools) ^ expected)}"
        )
        sys.modules.pop("trw_memory.server", None)

    def test_all_tool_modules_importable(self) -> None:
        """All tool modules must be importable with their impl functions."""
        from trw_memory.tools import audit, consolidate, forget, recall, review, search, status, store

        assert callable(store.memory_store_impl)
        assert callable(recall.memory_recall_impl)
        assert callable(audit.memory_audit_impl)
        assert callable(review.memory_review_impl)
        assert callable(forget.memory_forget_impl)
        assert callable(consolidate.memory_consolidate_impl)
        assert callable(search.memory_search_impl)
        assert callable(status.memory_status_impl)

    def test_register_functions_exist(self) -> None:
        """Each tool module must export a register_*_tool function."""
        from trw_memory.tools.audit import register_audit_tool
        from trw_memory.tools.consolidate import register_consolidate_tool
        from trw_memory.tools.forget import register_forget_tool
        from trw_memory.tools.recall import register_recall_tool
        from trw_memory.tools.review import register_review_tool
        from trw_memory.tools.search import register_search_tool
        from trw_memory.tools.status import register_status_tool
        from trw_memory.tools.store import register_store_tool

        for fn in [
            register_store_tool,
            register_recall_tool,
            register_audit_tool,
            register_review_tool,
            register_forget_tool,
            register_consolidate_tool,
            register_search_tool,
            register_status_tool,
        ]:
            assert callable(fn), f"{fn} must be callable"
