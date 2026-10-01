"""A verb refused because the daemon owns the store names how the USER stops it, and tells agents not to (INC-129 d)."""

from __future__ import annotations

import pytest

from trw_memory import cli_client
from trw_memory.daemon import DaemonInfo
from trw_memory.daemon._discovery import AGENT_MUST_NOT_STOP


def _info() -> DaemonInfo:
    return DaemonInfo(pid=4242, url="http://127.0.0.1:41234/mcp", started_at="2026-09-30T00:00:00Z", version="5.1.1")


def test_the_refusal_names_the_pid_and_the_user_only_stop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli_client, "read_live_discovery", lambda _paths: _info())
    assert cli_client.refused_beside_daemon("backup create") is True
    err = capsys.readouterr().err
    assert "pid 4242" in err and "kill 4242" in err  # how the user stops it
    assert "next memory call starts" in err  # and what happens after
    assert AGENT_MUST_NOT_STOP in err  # CONSTITUTION HB-2


def test_no_daemon_means_no_refusal(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from trw_memory.daemon import DiscoveryAbsent

    monkeypatch.setattr(cli_client, "read_live_discovery", lambda _paths: DiscoveryAbsent(reason="none"))
    assert cli_client.refused_beside_daemon("backup create") is False
    assert capsys.readouterr().err == ""
