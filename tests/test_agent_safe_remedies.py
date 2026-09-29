"""Every remedy that names a process to stop is addressed to the user, never an instruction to the calling agent.

CONSTITUTION HB-2: agents never kill a process they did not start. The 5.0.0 ``daemon_version_mismatch`` message
said "Stop process <pid> and retry", and an opencode agent ran ``kill <pid>`` on a daemon it did not own; only a
bash:ask permission stopped it (DoD-5 run, 2026-09-26).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from trw_memory.daemon._discovery import AGENT_MUST_NOT_STOP, DaemonInfo

pytestmark = pytest.mark.unit

_IMPERATIVE = re.compile(r"\bkill\b|\bstop (it|process)\b[^.]*\band retry\b|^stop\b", re.IGNORECASE)


def _assert_user_addressed(message: str, pid: int) -> None:
    assert not _IMPERATIVE.search(message), f"an agent-actionable imperative survives: {message!r}"
    assert str(pid) in message, "the pid stays, for the human"
    assert AGENT_MUST_NOT_STOP in message


def _info(process_start: str | None) -> DaemonInfo:
    return DaemonInfo(
        pid=4242,
        url="http://127.0.0.1:1/mcp",
        started_at="2026-09-26T00:00:00+00:00",
        version="4.0.1",
        process_start=process_start,
    )


@pytest.mark.parametrize("process_start", ["123.0", None], ids=["start-proven", "start-unknown"])
def test_the_stop_remedy_is_addressed_to_the_user(process_start: str | None) -> None:
    _assert_user_addressed(_info(process_start).stop_remedy(Path("/tmp/daemon.json")), 4242)


def test_the_version_mismatch_error_is_addressed_to_the_user(monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.daemon import client as client_mod
    from trw_memory.daemon.client import DaemonClient, DaemonVersionMismatchError

    monkeypatch.setattr(client_mod, "_package_version", lambda: "5.0.0")
    with pytest.raises(DaemonVersionMismatchError) as raised:
        DaemonClient._compatible(DaemonClient.__new__(DaemonClient), _info("1.0"))

    message = str(raised.value)
    _assert_user_addressed(message, 4242)
    assert "No memory was read or written" in message
    assert "restarting the MCP client alone leaves it running" in message, "restart advice alone would not resolve it"
