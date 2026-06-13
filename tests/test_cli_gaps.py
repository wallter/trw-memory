"""Wave 15: coverage gap-fill for cli.py.

Target lines: 69, 80, 163, 168, 173-184, 189-190, 195-208, 213-225, 247-249, 270, 277.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from trw_memory.cli import _dispatch, main


# ---------------------------------------------------------------------------
# lines 69, 80: SystemExit re-raise in _cli_error_boundary (async + sync)
# ---------------------------------------------------------------------------

class TestCliErrorBoundarySystemExitReRaise:
    async def test_async_handler_systemexit_is_reraised(self) -> None:
        """async inner fn raises SystemExit → async_wrapper re-raises (line 69).

        Patch the inner `handle_store` (not the wrapped `_handle_store`) so
        the SystemExit bubbles through async_wrapper's re-raise path, then
        _dispatch catches it and converts to return code.
        """
        args = argparse.Namespace(command="store")
        with patch("trw_memory.cli.handle_store", side_effect=SystemExit(42)):
            # _dispatch catches the re-raised SystemExit and returns its code
            rc = await _dispatch(args)
        assert rc == 42

    def test_sync_handler_systemexit_is_reraised(self) -> None:
        """sync handler raises SystemExit → re-raised not caught (line 80)."""
        from trw_memory.cli import _cli_error_boundary

        @_cli_error_boundary
        def _raises_sys_exit() -> int:
            raise SystemExit(7)

        with pytest.raises(SystemExit) as exc_info:
            _raises_sys_exit()  # type: ignore[call-arg]
        assert exc_info.value.code == 7

    def test_sync_handler_exception_becomes_systemexit_1(self) -> None:
        """sync handler raises generic Exception → wrapped into SystemExit(1)."""
        from trw_memory.cli import _cli_error_boundary

        @_cli_error_boundary
        def _raises_error() -> int:
            raise RuntimeError("boom")

        with pytest.raises(SystemExit) as exc_info:
            _raises_error()  # type: ignore[call-arg]
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# lines 163, 168: _handle_restore and _handle_snapshot via _dispatch
# ---------------------------------------------------------------------------

class TestHandleRestoreAndSnapshot:
    async def test_restore_handler_dispatched(self, tmp_path: Path) -> None:
        """_dispatch('restore') → _handle_restore called (line 163)."""
        args = argparse.Namespace(command="restore")
        with patch("trw_memory.cli.handle_restore", return_value=0) as mock_restore:
            with patch("trw_memory.cli.MemoryConfig"):
                rc = await _dispatch(args)
        mock_restore.assert_called_once()
        assert rc == 0

    async def test_snapshot_handler_dispatched(self, tmp_path: Path) -> None:
        """_dispatch('snapshot') → _handle_snapshot called (line 168)."""
        args = argparse.Namespace(command="snapshot")
        with patch("trw_memory.cli.handle_snapshot", return_value=0) as mock_snapshot:
            with patch("trw_memory.cli.MemoryConfig"):
                rc = await _dispatch(args)
        mock_snapshot.assert_called_once()
        assert rc == 0


# ---------------------------------------------------------------------------
# lines 173-184: _handle_wiki_lint
# ---------------------------------------------------------------------------

class TestHandleWikiLint:
    async def test_wiki_lint_valid_pages(self, tmp_path: Path, capsys) -> None:
        """_dispatch('wiki-lint') with valid JSON file → prints lint result (lines 173-184)."""
        pages_file = tmp_path / "pages.json"
        pages_file.write_text(json.dumps([{"title": "Page 1", "content": "body"}]))
        args = argparse.Namespace(command="wiki-lint", path=str(pages_file), top_limit=10)
        with patch("trw_memory.cli.memory_wiki_lint_impl", return_value={"ok": True}) as mock_lint:
            rc = await _dispatch(args)
        assert rc == 0
        mock_lint.assert_called_once()
        captured = capsys.readouterr()
        assert "ok" in captured.out

    async def test_wiki_lint_non_dict_item_returns_1(self, tmp_path: Path, capsys) -> None:
        """wiki-lint with non-dict item → JsonInputError → return 1 (line 182)."""
        pages_file = tmp_path / "pages.json"
        pages_file.write_text(json.dumps(["not a dict"]))
        args = argparse.Namespace(command="wiki-lint", path=str(pages_file), top_limit=10)
        rc = await _dispatch(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err


# ---------------------------------------------------------------------------
# lines 189-190: _handle_code_index
# ---------------------------------------------------------------------------

class TestHandleCodeIndex:
    async def test_code_index_dispatched(self, tmp_path: Path, capsys) -> None:
        """_dispatch('code-index') → memory_code_index_impl called (lines 189-190)."""
        args = argparse.Namespace(command="code-index", root=str(tmp_path), namespace="project:default")
        with patch("trw_memory.cli.memory_code_index_impl", return_value={"indexed": 0}) as mock_idx:
            rc = await _dispatch(args)
        assert rc == 0
        mock_idx.assert_called_once_with(str(tmp_path), namespace="project:default")
        captured = capsys.readouterr()
        assert "indexed" in captured.out


# ---------------------------------------------------------------------------
# lines 195-208: _handle_code_search
# ---------------------------------------------------------------------------

class TestHandleCodeSearch:
    async def test_code_search_dispatched(self, tmp_path: Path, capsys) -> None:
        """_dispatch('code-search') → memory_code_search_impl called (lines 195-208)."""
        args = argparse.Namespace(
            command="code-search",
            root=str(tmp_path),
            query="def main",
            namespace="project:default",
            path_glob=None,
            language=None,
            limit=10,
        )
        with patch("trw_memory.cli.memory_code_search_impl", return_value=[]) as mock_srch:
            rc = await _dispatch(args)
        assert rc == 0
        mock_srch.assert_called_once()
        captured = capsys.readouterr()
        assert "[]" in captured.out


# ---------------------------------------------------------------------------
# lines 213-225: _handle_code_symbol
# ---------------------------------------------------------------------------

class TestHandleCodeSymbol:
    async def test_code_symbol_dispatched(self, tmp_path: Path, capsys) -> None:
        """_dispatch('code-symbol') → memory_code_symbol_impl called (lines 213-225)."""
        args = argparse.Namespace(
            command="code-symbol",
            root=str(tmp_path),
            name="my_func",
            namespace="project:default",
            kind=None,
            path=None,
        )
        with patch("trw_memory.cli.memory_code_symbol_impl", return_value={"symbols": []}) as mock_sym:
            rc = await _dispatch(args)
        assert rc == 0
        mock_sym.assert_called_once()
        captured = capsys.readouterr()
        assert "symbols" in captured.out


# ---------------------------------------------------------------------------
# lines 247-249: unknown command in _dispatch
# ---------------------------------------------------------------------------

class TestDispatchUnknownCommand:
    async def test_unknown_command_returns_1(self, capsys) -> None:
        """_dispatch with unknown command → prints error, returns 1 (lines 247-249)."""
        args = argparse.Namespace(command="not-a-real-command")
        rc = await _dispatch(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "Unknown command" in captured.err


# ---------------------------------------------------------------------------
# line 270: quiet mode sets verbosity = -1
# ---------------------------------------------------------------------------

class TestMainQuietMode:
    def test_quiet_flag_sets_verbosity_minus_one(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main(['--quiet', 'store', ...]) → verbosity=-1 passed to configure_logging (line 270)."""
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_STORAGE_SQLITE_PATH", str(tmp_path / "mem.db"))
        monkeypatch.setenv("MEMORY_EMBEDDINGS_ENABLED", "0")

        captured_verbosity: list[int] = []

        def _mock_configure_logging(verbosity: int, log_level: str | None = None) -> None:
            captured_verbosity.append(verbosity)

        with patch("trw_memory.cli._dispatch", new=AsyncMock(return_value=0)):
            with patch("trw_memory._logging.configure_logging", side_effect=_mock_configure_logging):
                rc = main(["--quiet", "store", "--summary", "test"])

        assert rc == 0
        assert captured_verbosity == [-1]
