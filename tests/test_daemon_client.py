"""PRD-CORE-253 FR08 — the client fails closed, and says how to fix it.

Unreachability is produced for real here, not simulated: a discovery record
naming a LIVE process (this one) and a port nothing is listening on is exactly
the state a killed daemon or a reused pid leaves behind, and it is the state in
which a fail-open client would quietly return an empty recall.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from trw_memory.daemon import (
    DaemonInfo,
    DaemonPaths,
    DiscoveryAbsent,
    ensure_token,
    read_discovery,
    read_live_discovery,
)
from trw_memory.daemon.client import DAEMON_START_COMMAND, DaemonClient
from trw_memory.exceptions import DaemonAuthError, DaemonUnreachableError
from trw_memory.models.config import MemoryConfig

pytest.importorskip("fastmcp")

_START_DEADLINE_SECONDS = 60.0
_TEST_IDLE_SECONDS = 15.0


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DaemonPaths:
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "userhome"))
    return DaemonPaths.resolve()


@pytest.fixture
def config() -> MemoryConfig:
    """A 1-second startup deadline so an unstartable daemon fails promptly."""
    return MemoryConfig(memory_daemon_startup_timeout_seconds=1.0)


def _closed_port() -> int:
    """A port that was bound and released, so nothing is listening on it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _record_an_unreachable_daemon(paths: DaemonPaths) -> None:
    """Publish a discovery record naming a live pid and a dead endpoint."""
    paths.user_memory_dir.mkdir(parents=True, exist_ok=True)
    info = DaemonInfo(
        pid=os.getpid(),
        url=f"http://127.0.0.1:{_closed_port()}/mcp",
        token="recorded-token",
        started_at="2026-09-03T00:00:00+00:00",
        version="test",
    )
    paths.discovery.write_text(info.model_dump_json(), encoding="utf-8")


@pytest.fixture
def running_daemon(paths: DaemonPaths) -> Iterator[DaemonInfo]:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "trw_memory.server",
            "serve",
            "http",
            "--idle-shutdown-seconds",
            str(_TEST_IDLE_SECONDS),
        ],
        env={**os.environ, "TRW_USER_DIR": str(paths.user_memory_dir.parent)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + _START_DEADLINE_SECONDS
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail("daemon exited during startup")
            info = read_discovery(paths)
            if info is not None:
                yield info
                return
            time.sleep(0.05)
        pytest.fail("daemon never published a discovery file")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)


async def test_daemon_unreachable_fails_closed_with_actionable_error(paths: DaemonPaths, config: MemoryConfig) -> None:
    """FR08: BOTH a read and a write fail; no store is created; the remedy is named."""
    _record_an_unreachable_daemon(paths)
    client = DaemonClient(config=config, paths=paths)

    with pytest.raises(DaemonUnreachableError) as write_failure:
        await client.store("a conclusion that must not be silently dropped", "project:closed-aaaaaaaa")

    with pytest.raises(DaemonUnreachableError) as read_failure:
        await client.recall("anything", "project:closed-aaaaaaaa")

    for failure in (write_failure, read_failure):
        message = str(failure.value)
        assert str(paths.discovery) in message, "the error must name the discovery file"
        assert "daemon.json" in message
        assert DAEMON_START_COMMAND in message, "the error must name the start command"

    # FR08 clause 4: nothing anywhere resembling a store was created.
    assert not list(paths.user_memory_dir.glob("**/*.db"))
    assert not paths.store.exists()


async def test_a_failed_call_is_attempted_exactly_twice(paths: DaemonPaths, config: MemoryConfig) -> None:
    """FR08 clause 1: try once, retry exactly once, then fail. Never a third."""
    _record_an_unreachable_daemon(paths)
    client = DaemonClient(config=config, paths=paths)

    with structlog.testing.capture_logs() as logs:
        with pytest.raises(DaemonUnreachableError):
            await client.recall("anything", "project:closed-aaaaaaaa")

    attempts = [entry["attempt"] for entry in logs if entry.get("event") == "daemon_call_failed"]
    assert attempts == [1, 2], f"expected exactly two attempts, saw {attempts}"


async def test_a_rejected_token_fails_closed_without_regenerating(
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo
) -> None:
    """FR08 clause 3: rejection is a distinct, non-retried, non-rotating failure.

    Automatic rotation would let any local process force one by corrupting the
    file, so a corrupted token must survive the failed call byte for byte.
    """
    wrong = "a-token-this-daemon-never-issued"
    paths.token.write_text(wrong, encoding="utf-8")
    client = DaemonClient(config=config, paths=paths)

    with structlog.testing.capture_logs() as logs:
        with pytest.raises(DaemonAuthError, match="NOT regenerated"):
            await client.recall("anything", "project:auth-bbbbbbbb")

    assert paths.token.read_text(encoding="utf-8") == wrong, "the token was rotated on rejection"
    assert [entry for entry in logs if entry.get("event") == "daemon_call_failed"] == [], "a rejection was retried"
    assert wrong not in str(logs), "the token leaked into a log event"


async def test_a_missing_token_is_generated_at_0600_and_the_attach_succeeds(
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo
) -> None:
    """FR08 clause 2: first run is not an error."""
    token_before = paths.token.read_text(encoding="utf-8")
    paths.token.unlink()

    client = DaemonClient(config=config, paths=paths)
    ensure_token(paths)

    assert paths.token.exists()
    assert paths.token.stat().st_mode & 0o777 == 0o600
    # A freshly minted token is not the running daemon's, so the attach fails
    # closed on AUTH -- proving the generation happened and that the client did
    # not paper over the mismatch.
    with pytest.raises(DaemonAuthError):
        await client.recall("anything", "project:fresh-cccccccc")

    paths.token.write_text(token_before, encoding="utf-8")
    result = await client.recall("anything", "project:fresh-cccccccc")
    assert isinstance(result, dict)


async def test_a_live_daemon_serves_reads_and_writes_over_loopback(
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo
) -> None:
    """The positive path: the same client that fails closed also works."""
    client = DaemonClient(config=config, paths=paths)
    namespace = "project:roundtrip-dddddddd"

    stored = await client.store("a learning written through the daemon client", namespace)
    assert stored["status"] == "stored"

    recalled = await client.recall("learning written through", namespace)
    assert isinstance(recalled, dict)


def test_reading_the_record_never_starts_a_daemon(paths: DaemonPaths) -> None:
    """A reachability probe must not be the thing that starts one."""
    assert isinstance(read_live_discovery(paths), DiscoveryAbsent)
    assert not paths.discovery.exists()
    assert not paths.token.exists()


def test_reading_the_record_reports_a_running_daemon(paths: DaemonPaths, running_daemon: DaemonInfo) -> None:
    """And it does report one that is genuinely there."""
    probed = read_live_discovery(paths)

    assert isinstance(probed, DaemonInfo)
    assert probed.pid == running_daemon.pid
    assert probed.url == running_daemon.url
