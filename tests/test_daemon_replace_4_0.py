"""E2E-INC-136: a client's own replacement of a major-older daemon (update-project, canary, uv/pipx) reaches a 4.0 daemon.

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

from trw_memory.daemon import DaemonPaths
from trw_memory.daemon import _legacy_identity as legacy
from trw_memory.daemon._discovery import DaemonInfo, read_live_discovery
from trw_memory.daemon._paths import write_secret_file
from trw_memory.daemon._upgrade import replace_older_daemon
from trw_memory.models.config import MemoryConfig

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


def _replace(paths: DaemonPaths, info: DaemonInfo) -> str:
    return replace_older_daemon(
        info, paths, MemoryConfig(memory_daemon_autostart=True), "grant-token", pinned=False, mine="9.9.1"
    )


def _info(paths: DaemonPaths) -> DaemonInfo:
    found = read_live_discovery(paths)
    assert isinstance(found, DaemonInfo)
    return found


def test_a_client_replaces_a_proven_4_0_daemon_that_offers_no_handshake(
    paths: DaemonPaths,
    sleeper: subprocess.Popen[bytes],
    socket_daemon: type[_FakeDaemon],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket_daemon.reported_pid = sleeper.pid
    _plant(paths, sleeper.pid, socket_daemon.url)  # type: ignore[attr-defined]
    _ps(monkeypatch, (os.getuid(), _SERVER_COMMAND))

    assert _replace(paths, _info(paths)) == "", "the slot is free once the proven 4.0 daemon is stopped"

    sleeper.wait(timeout=10)  # raises TimeoutExpired: the proven daemon is still running


@pytest.mark.parametrize(
    "ps_answer", [None, (os.getuid() + 1, _SERVER_COMMAND), (os.getuid(), "/usr/bin/python3 -c x")]
)
def test_a_4_0_daemon_whose_identity_is_not_proven_keeps_the_refusal(
    paths: DaemonPaths,
    sleeper: subprocess.Popen[bytes],
    socket_daemon: type[_FakeDaemon],
    monkeypatch: pytest.MonkeyPatch,
    ps_answer: tuple[int, str] | None,
) -> None:
    socket_daemon.reported_pid = sleeper.pid
    _plant(paths, sleeper.pid, socket_daemon.url)  # type: ignore[attr-defined]
    _ps(monkeypatch, ps_answer)

    reason = _replace(paths, _info(paths))

    assert "identity is not proven" in reason
    assert sleeper.poll() is None


def test_a_record_with_a_start_but_no_handshake_is_still_not_stopped_here(
    paths: DaemonPaths, sleeper: subprocess.Popen[bytes], socket_daemon: type[_FakeDaemon]
) -> None:
    """5.0.0 and older WITH an OS start keep the 'does not offer the drain handshake' refusal."""
    from trw_memory.storage._pid_liveness import process_start

    paths.user_memory_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = DaemonInfo(
        pid=sleeper.pid,
        url="http://127.0.0.1:9/mcp",
        started_at=datetime.now(timezone.utc).isoformat(),
        version=_SERVED,
        process_start=process_start(sleeper.pid),
    )
    write_secret_file(paths.discovery, info.model_dump_json())

    assert "does not offer the drain handshake" in _replace(paths, info)
    assert sleeper.poll() is None
