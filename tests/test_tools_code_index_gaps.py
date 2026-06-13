"""Wave 12: coverage gap-fill for tools/code_index.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from trw_memory.tools.code_index import (
    memory_code_index_impl,
    memory_code_search_impl,
    memory_code_symbol_impl,
)


class TestMemoryCodeSearchImplErrorPaths:
    def test_invalid_root_file_returns_error(self, tmp_path: Path) -> None:
        file = tmp_path / "file.txt"
        file.write_text("text")

        result = memory_code_search_impl(str(file), "query")

        assert result["status"] == "failed"
        assert result["error_code"] == "invalid_root"

    def test_os_error_returns_error(self, tmp_path: Path) -> None:
        with patch(
            "trw_memory.tools.code_index.code_search",
            side_effect=OSError("disk error"),
        ):
            result = memory_code_search_impl(str(tmp_path), "query")

        assert result["status"] == "failed"
        assert result["error_code"] == "search_failed"
        assert "results" in result

    def test_auto_indexes_when_store_empty(self, tmp_path: Path) -> None:
        """Search auto-indexes when store has no files."""
        src = tmp_path / "mod.py"
        src.write_text("def hello(): pass")

        result = memory_code_search_impl(str(tmp_path), "hello", namespace="auto-idx")
        assert result["status"] == "ok"


class TestMemoryCodeSymbolImplErrorPaths:
    def test_invalid_root_file_returns_error(self, tmp_path: Path) -> None:
        file = tmp_path / "file.txt"
        file.write_text("text")

        result = memory_code_symbol_impl(str(file), "my_func")

        assert result["status"] == "failed"
        assert result["error_code"] == "invalid_root"

    def test_os_error_returns_error(self, tmp_path: Path) -> None:
        with patch(
            "trw_memory.tools.code_index.lookup_symbols",
            side_effect=OSError("disk error"),
        ):
            result = memory_code_symbol_impl(str(tmp_path), "my_func")

        assert result["status"] == "failed"
        assert result["error_code"] == "symbol_failed"
        assert "results" in result

    def test_auto_indexes_when_store_empty(self, tmp_path: Path) -> None:
        """Symbol lookup auto-indexes when store has no files."""
        src = tmp_path / "mod.py"
        src.write_text("def my_func(): pass")

        result = memory_code_symbol_impl(str(tmp_path), "my_func", namespace="auto-sym")
        assert result["status"] == "ok"


class TestMemoryCodeIndexImplOSError:
    def test_os_error_returns_error(self, tmp_path: Path) -> None:
        with patch(
            "trw_memory.tools.code_index.CodeIndexer",
            side_effect=OSError("permission denied"),
        ):
            result = memory_code_index_impl(str(tmp_path))

        assert result["status"] == "failed"
        assert result["error_code"] == "index_failed"


class TestRegisterCodeIndexTools:
    def test_register_tools_calls_mcp_tool_three_times(self) -> None:
        from trw_memory.tools.code_index import register_code_index_tools

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda f: f

        register_code_index_tools(mock_mcp)

        assert mock_mcp.tool.call_count == 3

    async def test_registered_index_tool_delegates_to_impl(self, tmp_path: Path) -> None:
        from trw_memory.tools.code_index import register_code_index_tools

        registered = {}
        mock_mcp = MagicMock()
        call_order = []

        def capture(f):
            call_order.append(f.__name__)
            registered[f.__name__] = f
            return f

        mock_mcp.tool.return_value = capture
        register_code_index_tools(mock_mcp)

        src = tmp_path / "mod.py"
        src.write_text("def hello(): pass")
        result = await registered["memory_code_index"](str(tmp_path))
        assert result["status"] == "ok"

    async def test_registered_search_tool_delegates_to_impl(self, tmp_path: Path) -> None:
        from trw_memory.tools.code_index import register_code_index_tools

        registered = {}
        mock_mcp = MagicMock()

        def capture(f):
            registered[f.__name__] = f
            return f

        mock_mcp.tool.return_value = capture
        register_code_index_tools(mock_mcp)

        src = tmp_path / "mod.py"
        src.write_text("def hello(): pass")
        memory_code_index_impl(str(tmp_path), namespace="reg-s")
        result = await registered["memory_code_search"](str(tmp_path), "hello", namespace="reg-s")
        assert result["status"] == "ok"

    async def test_registered_symbol_tool_delegates_to_impl(self, tmp_path: Path) -> None:
        from trw_memory.tools.code_index import register_code_index_tools

        registered = {}
        mock_mcp = MagicMock()

        def capture(f):
            registered[f.__name__] = f
            return f

        mock_mcp.tool.return_value = capture
        register_code_index_tools(mock_mcp)

        src = tmp_path / "mod.py"
        src.write_text("def hello(): pass")
        memory_code_index_impl(str(tmp_path), namespace="reg-sym")
        result = await registered["memory_code_symbol"](str(tmp_path), "hello", namespace="reg-sym")
        assert result["status"] == "ok"
