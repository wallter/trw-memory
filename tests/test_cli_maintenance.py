"""Tests for trw_memory.cli maintenance commands."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trw_memory.cli import main
from trw_memory.exceptions import DaemonUnreachableError

from ._test_cli_support import _DAEMON_CLIENT, _mock_client


class TestConsolidateCommand:
    def test_consolidate_runs_over_the_daemon(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = _mock_client()
        monkeypatch.setattr(_DAEMON_CLIENT, lambda: client)

        assert main(["consolidate", "--dry-run", "--namespace", "project:a-11111111"]) == 0

        client.consolidate.assert_awaited_once_with("project:a-11111111", dry_run=True)
        assert json.loads(capsys.readouterr().out)["entries_consolidated"] == 3

    def test_a_refused_consolidate_exits_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = _mock_client()
        client.consolidate = AsyncMock(return_value={"status": "invalid", "error": "Invalid namespace"})
        monkeypatch.setattr(_DAEMON_CLIENT, lambda: client)

        assert main(["consolidate", "--namespace", "../../escape"]) == 1
        assert "Invalid namespace" in capsys.readouterr().err


class TestStatusCommand:
    def _client(self, total: int) -> MagicMock:
        client = _mock_client()
        client.status = AsyncMock(return_value={"total_entries": total, "config": {"storage_backend": "sqlite"}})
        return client

    def test_status_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch(_DAEMON_CLIENT, return_value=self._client(42)):
            ret = main(["status", "--namespace", "default"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "42" in captured.out
        assert "Memory System Status" in captured.out

    def test_status_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch(_DAEMON_CLIENT, return_value=self._client(5)):
            ret = main(["status", "--namespace", "default", "--format", "json"])
        assert ret == 0
        parsed = json.loads(capsys.readouterr().out)
        assert (parsed["entry_count"], parsed["storage_path"]) == (5, "/daemon/memory.db")

    def test_status_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        client = _mock_client()
        client.status = AsyncMock(side_effect=DaemonUnreachableError("the daemon is gone"))
        with patch(_DAEMON_CLIENT, return_value=client):
            ret = main(["status", "--namespace", "default"])
        assert ret == 1
        assert "Error:" in capsys.readouterr().err
