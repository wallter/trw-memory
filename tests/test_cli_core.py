"""Tests for trw_memory.cli parser and client-backed commands."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from trw_memory.cli import main
from trw_memory.cli_parser import build_parser

from ._test_cli_support import _DAEMON_CLIENT, _mock_client


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
        assert args.namespace is None, "the daemon verbs default to this checkout's project identity"

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
        args = parser.parse_args(["search", "--tags", "py", "--status", "active"])
        assert args.command == "search"
        assert args.tags == ["py"]
        assert args.status == "active"

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


@pytest.fixture
def daemon(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """The daemon client every daemon verb reaches."""
    client = _mock_client()
    monkeypatch.setattr(_DAEMON_CLIENT, lambda: client)
    return client


class TestStoreCommand:
    def test_store_success(self, daemon: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        ret = main(["store", "--summary", "Test content", "--namespace", "project:a-11111111"])
        assert ret == 0
        assert "Stored: M-abc12345" in capsys.readouterr().out
        daemon.store.assert_awaited_once_with(
            "Test content", "project:a-11111111", tags=None, importance=0.5, detail=""
        )

    def test_store_with_tags_and_importance(self, daemon: MagicMock) -> None:
        main(["store", "--summary", "t", "--tags", "py", "--tags", "sql", "--importance", "0.9", "--namespace", "n"])
        assert daemon.store.call_args.kwargs["tags"] == ["py", "sql"]
        assert daemon.store.call_args.kwargs["importance"] == 0.9

    def test_the_namespace_defaults_to_the_project_identity(self, daemon: MagicMock) -> None:
        from trw_memory.namespaces.identity import resolve_project_identity

        main(["store", "--summary", "t"])
        assert daemon.store.call_args.args[1] == resolve_project_identity().namespace

    def test_a_refused_store_exits_1_without_a_traceback(
        self, daemon: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        daemon.store = AsyncMock(return_value={"status": "invalid", "error": "content must be non-empty"})
        ret = main(["store", "--summary", "", "--namespace", "n"])
        err = capsys.readouterr().err
        assert ret == 1
        assert "content must be non-empty" in err
        assert "Traceback" not in err

    def test_store_missing_summary(self) -> None:
        with pytest.raises(SystemExit):
            main(["store"])


class TestRecallCommand:
    def test_recall_table(self, daemon: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        ret = main(["recall", "test query", "--namespace", "n"])
        assert ret == 0
        assert "M-abc12345" in capsys.readouterr().out
        daemon.recall.assert_awaited_once_with("test query", "n", limit=10, tags=None)

    def test_recall_json(self, daemon: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["recall", "q", "--format", "json", "--namespace", "n"]) == 0
        assert isinstance(json.loads(capsys.readouterr().out), list)

    def test_recall_compact(self, daemon: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["recall", "q", "--format", "compact", "--namespace", "n"]) == 0
        assert "score=" in capsys.readouterr().out

    def test_recall_with_tags_and_limit(self, daemon: MagicMock) -> None:
        main(["recall", "q", "--tags", "py", "--limit", "5", "--namespace", "n"])
        assert daemon.recall.call_args.kwargs == {"limit": 5, "tags": ["py"]}

    def test_an_unreachable_daemon_names_the_start_command(
        self, daemon: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from trw_memory.daemon import DAEMON_START_COMMAND
        from trw_memory.exceptions import DaemonUnreachableError

        daemon.recall = AsyncMock(
            side_effect=DaemonUnreachableError(f"unreachable. Start it with: {DAEMON_START_COMMAND}")
        )
        ret = main(["recall", "q", "--namespace", "n"])
        err = capsys.readouterr().err
        assert ret == 1
        assert DAEMON_START_COMMAND in err
        assert "Traceback" not in err


class TestSearchCommand:
    def test_search_success(self, daemon: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        ret = main(["search", "--tags", "py", "--status", "active", "--namespace", "n"])
        assert ret == 0
        assert "M-abc12345" in capsys.readouterr().out
        daemon.search.assert_awaited_once_with("n", tags=["py"], status="active", limit=50)

    def test_search_json_format(self, daemon: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["search", "--format", "json", "--namespace", "n"]) == 0
        assert isinstance(json.loads(capsys.readouterr().out), list)

    def test_the_local_only_filters_are_gone(self) -> None:
        with pytest.raises(SystemExit):
            main(["search", "--min-importance", "0.8"])


class TestForgetCommand:
    def test_forget_success(self, daemon: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        ret = main(["forget", "M-abc123", "--namespace", "n"])
        assert ret == 0
        assert "Deleted: M-abc123" in capsys.readouterr().out
        daemon.forget.assert_awaited_once_with("M-abc123", "n")

    def test_forget_not_found(self, daemon: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        daemon.forget = AsyncMock(return_value={"deleted": 0, "status": "not_found"})
        assert main(["forget", "M-nonexistent", "--namespace", "n"]) == 1
        assert "not_found" in capsys.readouterr().err

    def test_forget_missing_id(self) -> None:
        with pytest.raises(SystemExit):
            main(["forget"])
