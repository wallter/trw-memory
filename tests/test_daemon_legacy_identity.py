"""E2E-INC-134: an upgrade stops a 4.0 daemon whose record carries no OS start, but only on full proof.

The record is 4.0-shaped (no ``process_start``). The proof is the daemon's own socket (a real loopback
server answering with the 4.0.1 version gate's exact refusal text), the record's pid, and ``ps``.
Every missing piece keeps today's refusal and signals nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from trw_memory.daemon import DaemonPaths, stop_outdated_daemon
from trw_memory.daemon import _legacy_identity as legacy
from trw_memory.daemon._discovery import VERSION_HEADER, DaemonInfo
from trw_memory.daemon._paths import write_secret_file

_INSTALLED = "9.9.1"
_SERVED = "4.0.1"
#: The 4.0.1 gate's refusal, byte for byte (git show 791a394ea4:.../_version_gate.py).
_GATE_TEXT = (
    "daemon_version_mismatch: this trw-memory daemon (pid {pid}) serves {version}, but the calling client is "
    "trw-memory 9.9.1; their tool signatures differ, so nothing was read or written. Upgrade the client "
    "(pip install -U trw-mcp trw-memory, then reconnect the MCP server), or stop process {pid} so the "
    "client's own version starts a daemon."
)
_SERVER_COMMAND = "/usr/bin/python3 -m trw_memory.server serve http"


class _FakeDaemon(BaseHTTPRequestHandler):
    reported_pid = 0
    reported_version = _SERVED
    seen: list[dict[str, str]] = []

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        type(self).seen.append({k.lower(): v for k, v in self.headers.items()})
        text = _GATE_TEXT.format(pid=self.reported_pid, version=self.reported_version)
        body = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": {"isError": True, "content": [{"type": "text", "text": text}]}}
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


@pytest.fixture
def socket_daemon() -> Iterator[type[_FakeDaemon]]:
    handler = type("Handler", (_FakeDaemon,), {"seen": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    handler.url = f"http://127.0.0.1:{server.server_port}/mcp"  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield handler
    server.shutdown()
    server.server_close()


@pytest.fixture
def paths(tmp_path: Path) -> DaemonPaths:
    return DaemonPaths(user_memory_dir=tmp_path / "memory")


@pytest.fixture
def sleeper() -> Iterator[subprocess.Popen[bytes]]:
    """A live process standing in for the 4.0 daemon (its real command line is not the daemon's: ps is faked)."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield process
    process.kill()
    process.wait()


def _plant(paths: DaemonPaths, pid: int, url: str) -> None:
    paths.user_memory_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = DaemonInfo(
        pid=pid, url=url, started_at=datetime.now(timezone.utc).isoformat(), version=_SERVED, process_start=None
    )
    write_secret_file(paths.discovery, info.model_dump_json())


def _ps(monkeypatch: pytest.MonkeyPatch, answer: tuple[int, str] | None) -> None:
    monkeypatch.setattr(legacy, "process_identity", lambda _pid: answer)


def test_a_4_0_daemon_with_a_matching_socket_and_ps_is_stopped(
    paths: DaemonPaths,
    sleeper: subprocess.Popen[bytes],
    socket_daemon: type[_FakeDaemon],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket_daemon.reported_pid = sleeper.pid
    _plant(paths, sleeper.pid, socket_daemon.url)  # type: ignore[attr-defined]
    _ps(monkeypatch, (os.getuid(), _SERVER_COMMAND))

    result = stop_outdated_daemon(paths, _INSTALLED, token="grant-token")

    assert result.outcome == "stopped", result.detail
    sleeper.wait(timeout=10)  # raises TimeoutExpired: the proven 4.0 daemon is still running
    sent = socket_daemon.seen[0]
    assert sent["authorization"] == "Bearer grant-token"
    assert sent[VERSION_HEADER] == _INSTALLED


@pytest.mark.parametrize(
    ("reported_pid_offset", "reported_version", "ps_answer", "why"),
    [
        (1, _SERVED, "ok", "socket reports another pid"),
        (0, "4.0.0", "ok", "socket reports another version"),
        (0, _SERVED, None, "ps cannot show the process"),
        (0, _SERVED, "other-user", "another user's process"),
        (0, _SERVED, "other-command", "not python -m trw_memory.server"),
    ],
)
def test_any_missing_proof_keeps_the_refusal_and_signals_nothing(
    paths: DaemonPaths,
    sleeper: subprocess.Popen[bytes],
    socket_daemon: type[_FakeDaemon],
    monkeypatch: pytest.MonkeyPatch,
    reported_pid_offset: int,
    reported_version: str,
    ps_answer: str | None,
    why: str,
) -> None:
    socket_daemon.reported_pid = sleeper.pid + reported_pid_offset
    socket_daemon.reported_version = reported_version
    _plant(paths, sleeper.pid, socket_daemon.url)  # type: ignore[attr-defined]
    answers = {
        "ok": (os.getuid(), _SERVER_COMMAND),
        None: None,
        "other-user": (os.getuid() + 1, _SERVER_COMMAND),
        "other-command": (os.getuid(), "/usr/bin/python3 -c import time"),
    }
    _ps(monkeypatch, answers[ps_answer])

    result = stop_outdated_daemon(paths, _INSTALLED, token="grant-token")

    assert result.outcome == "unproven", why
    assert f"process {sleeper.pid}" in result.detail, "the exact manual remedy is kept"
    assert sleeper.poll() is None, f"signalled despite: {why}"


def test_a_socket_that_does_not_answer_is_not_proof(
    paths: DaemonPaths, sleeper: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plant(paths, sleeper.pid, "http://127.0.0.1:9/mcp")  # nothing listens on the discard port
    _ps(monkeypatch, (os.getuid(), _SERVER_COMMAND))

    result = stop_outdated_daemon(paths, _INSTALLED, token="grant-token")

    assert result.outcome == "unproven"
    assert "did not report a pid and version" in result.detail
    assert sleeper.poll() is None


def test_an_older_only_run_leaves_a_proven_4_0_daemon_that_is_not_older(
    paths: DaemonPaths,
    sleeper: subprocess.Popen[bytes],
    socket_daemon: type[_FakeDaemon],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket_daemon.reported_pid = sleeper.pid
    _plant(paths, sleeper.pid, socket_daemon.url)  # type: ignore[attr-defined]
    _ps(monkeypatch, (os.getuid(), _SERVER_COMMAND))

    result = stop_outdated_daemon(paths, "3.0.0", older_only=True, token="grant-token")

    assert result.outcome == "not_older"
    assert sleeper.poll() is None
