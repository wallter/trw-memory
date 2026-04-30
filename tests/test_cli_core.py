"""Tests for trw_memory.cli parser and client-backed commands."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trw_memory.cli import main
from trw_memory.cli_parser import build_parser

from ._test_cli_support import _CLI, _mock_client


class TestBuildParser:
    def test_no_command_returns_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_store_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["store", "--summary", "test"])
        assert args.command == "store"
        assert args.summary == "test"
        assert args.importance == 0.5
        assert args.namespace == "default"

    def test_store_all_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "store",
                "--summary",
                "test",
                "--detail",
                "details",
                "--tags",
                "py",
                "--tags",
                "test",
                "--importance",
                "0.9",
                "--namespace",
                "myns",
            ]
        )
        assert args.summary == "test"
        assert args.detail == "details"
        assert args.tags == ["py", "test"]
        assert args.importance == 0.9
        assert args.namespace == "myns"

    def test_recall_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["recall", "my query"])
        assert args.command == "recall"
        assert args.query == "my query"
        assert args.limit == 10
        assert args.fmt == "table"

    def test_recall_with_format(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["recall", "q", "--format", "json"])
        assert args.fmt == "json"

    def test_search_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["search", "--tags", "py", "--min-importance", "0.5"])
        assert args.command == "search"
        assert args.tags == ["py"]
        assert args.min_importance == 0.5

    def test_consolidate_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["consolidate", "--dry-run"])
        assert args.command == "consolidate"
        assert args.dry_run is True

    def test_export_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["export", "--format", "yaml", "--output", "/tmp/out.yaml"])
        assert args.command == "export"
        assert args.fmt == "yaml"
        assert args.output == "/tmp/out.yaml"

    def test_import_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["import", "/tmp/data.json", "--merge"])
        assert args.command == "import"
        assert args.path == "/tmp/data.json"
        assert args.merge is True

    def test_status_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "--format", "json"])
        assert args.command == "status"
        assert args.fmt == "json"

    def test_forget_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["forget", "M-abc123"])
        assert args.command == "forget"
        assert args.memory_id == "M-abc123"


class TestMainNoCommand:
    def test_no_args_returns_1(self) -> None:
        assert main([]) == 1

    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        main([])
        captured = capsys.readouterr()
        assert "trw-memory" in captured.out


class TestStoreCommand:
    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_store_success(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["store", "--summary", "Test content"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Stored:" in captured.out
        assert "M-abc12345" in captured.out
        client.store.assert_awaited_once()
        client.close.assert_awaited_once()

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_store_with_tags(self, mock_cls: MagicMock) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        main(["store", "--summary", "test", "--tags", "py", "--tags", "sql"])
        call_kwargs = client.store.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("tags") == ["py", "sql"]

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_store_with_importance(self, mock_cls: MagicMock) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        main(["store", "--summary", "test", "--importance", "0.9"])
        call_kwargs = client.store.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("importance") == 0.9

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_store_error(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.store = AsyncMock(side_effect=ValueError("empty content"))
        mock_cls.return_value = client
        ret = main(["store", "--summary", ""])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_store_missing_summary(self) -> None:
        with pytest.raises(SystemExit):
            main(["store"])


class TestRecallCommand:
    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_recall_table(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["recall", "test query"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "M-abc12345" in captured.out
        client.recall.assert_awaited_once()

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_recall_json(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["recall", "test query", "--format", "json"])
        assert ret == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_recall_compact(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["recall", "q", "--format", "compact"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "score=" in captured.out

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_recall_with_tags(self, mock_cls: MagicMock) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        main(["recall", "q", "--tags", "py"])
        call_kwargs = client.recall.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("tags") == ["py"]

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_recall_with_limit(self, mock_cls: MagicMock) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        main(["recall", "q", "--limit", "5"])
        call_kwargs = client.recall.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("limit") == 5

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_recall_error(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.recall = AsyncMock(side_effect=RuntimeError("backend down"))
        mock_cls.return_value = client
        ret = main(["recall", "q"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err


class TestSearchCommand:
    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_search_success(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["search", "--tags", "py"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "M-abc12345" in captured.out

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_search_with_since(self, mock_cls: MagicMock) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["search", "--since", "2026-01-01T00:00:00"])
        assert ret == 0

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_search_invalid_since(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["search", "--since", "not-a-date"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_search_json_format(self, mock_cls: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["search", "--format", "json"])
        assert ret == 0
        parsed = json.loads(capsys.readouterr().out)
        assert isinstance(parsed, list)

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_search_with_min_importance(self, mock_cls: MagicMock) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        main(["search", "--min-importance", "0.8"])
        call_kwargs = client.search.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("min_importance") == 0.8


class TestForgetCommand:
    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_forget_success(
        self,
        mock_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = _mock_client()
        mock_cls.return_value = client
        ret = main(["forget", "M-abc123"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Deleted:" in captured.out
        client.forget.assert_awaited_once_with("M-abc123")

    @patch(f"{_CLI}.MemoryClient", autospec=False)
    def test_forget_not_found(
        self,
        mock_cls: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from trw_memory.exceptions import MemoryNotFoundError

        client = _mock_client()
        client.forget = AsyncMock(side_effect=MemoryNotFoundError("not found"))
        mock_cls.return_value = client
        ret = main(["forget", "M-nonexistent"])
        assert ret == 1
        assert "Error:" in capsys.readouterr().err

    def test_forget_missing_id(self) -> None:
        with pytest.raises(SystemExit):
            main(["forget"])
