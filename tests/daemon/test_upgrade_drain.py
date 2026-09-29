# ruff: noqa: F811
"""``drain_daemon``: the drain handshake with no major-older gate (HOTRELOAD-DAEMON-DRAIN-API)."""

from __future__ import annotations

import os
import subprocess

import httpx
import pytest

from tests.test_daemon_auto_restart import _MINE, _MINE_MAJOR, _NAMESPACE, _start, paths, started  # noqa: F401
from trw_memory.daemon import DaemonPaths, mint_grant
from trw_memory.daemon._discovery import VERSION_HEADER, DaemonInfo, DiscoveryAbsent, read_live_discovery
from trw_memory.daemon._upgrade import drain_daemon

pytest.importorskip("fastmcp")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process identity")


def test_drain_daemon_replaces_same_major(paths: DaemonPaths, started: list[subprocess.Popen[bytes]]) -> None:
    process, info = _start(paths, started, version=_MINE, capabilities="drain")

    reason = drain_daemon(paths, token=mint_grant(paths, [_NAMESPACE]), mine=_MINE)

    assert reason == ""
    assert process.wait(timeout=30) is not None
    assert isinstance(read_live_discovery(paths), DiscoveryAbsent)
    assert info.pid == process.pid


def test_drain_daemon_refuses_a_record_that_is_not_its_process(
    paths: DaemonPaths,
    started: list[subprocess.Popen[bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trw_memory.daemon import _upgrade

    process, info = _start(paths, started, version=_MINE, capabilities="drain")
    forged = info.model_copy(update={"process_start": "not-its-start"})
    monkeypatch.setattr(_upgrade, "read_live_discovery", lambda _paths: forged)
    reason = drain_daemon(paths, token=mint_grant(paths, [_NAMESPACE]), mine=_MINE)
    assert "could not be proven" in reason
    assert process.poll() is None, "a refused drain must leave the daemon serving"
    assert isinstance(read_live_discovery(paths), DaemonInfo)


def test_drain_daemon_with_no_daemon_returns_a_reason(paths: DaemonPaths) -> None:
    assert drain_daemon(paths, token="t", mine=_MINE) != ""


async def test_a_same_major_drain_passes_the_gate_but_not_without_the_drain_key(
    paths: DaemonPaths,
    started: list[subprocess.Popen[bytes]],
) -> None:
    from tests.test_daemon_auto_restart import _drain

    process, info = _start(paths, started, version=_MINE, capabilities="drain")

    answer = await _drain(info, mint_grant(paths, [_NAMESPACE]), 1.0, version=_MINE, key="0" * 64)

    assert isinstance(answer, dict) and answer["status"] == "declined", answer
    assert "same or a newer major" not in str(answer["detail"]), "the version gate refused a same-major caller"
    assert "0" * 64 not in str(answer["detail"])
    assert process.poll() is None and isinstance(read_live_discovery(paths), DaemonInfo)


@pytest.mark.parametrize("timeout", [0, -1.0, float("nan"), float("inf")])
def test_drain_daemon_rejects_a_timeout_that_is_not_finite_and_positive(paths: DaemonPaths, timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        drain_daemon(paths, token="t", mine=_MINE, timeout=timeout)


@pytest.mark.parametrize("header", ["unknown", f"{_MINE_MAJOR}rc1", ""])
async def test_a_non_drain_call_with_an_unparseable_client_version_is_refused(
    paths: DaemonPaths, started: list[subprocess.Popen[bytes]], header: str
) -> None:
    process, info = _start(paths, started, version=_MINE, capabilities="drain")
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "memory_recall", "arguments": {}}}
    headers = {
        "authorization": f"Bearer {mint_grant(paths, [_NAMESPACE])}",
        "accept": "application/json, text/event-stream",
        VERSION_HEADER: header,
    }
    async with httpx.AsyncClient(timeout=60) as http:
        response = await http.post(info.url, headers=headers, json=body)

    assert "daemon_version_mismatch" in response.text, response.text
