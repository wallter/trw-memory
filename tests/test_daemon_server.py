"""PRD-CORE-253 FR03/NFR02/NFR03 — the loopback daemon, on the real path.

These tests start a real ``trw-memory-server serve http`` subprocess against a
throwaway ``TRW_USER_DIR``, read the discovery file it publishes, and speak
streamable-HTTP to it with the fastmcp client. Nothing about the transport, the
token check, the lock or the shutdown is simulated: an in-process double would
not be able to fail the way the second-start and idle-shutdown requirements
describe.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.daemon import (
    LOOPBACK_HOST,
    DaemonInfo,
    DaemonPaths,
    bind_loopback_socket,
    claim_single_instance,
    ensure_token,
    read_discovery,
    require_loopback,
    tokens_match,
)
from trw_memory.exceptions import ConfigError, DaemonAlreadyRunningError, DaemonSecretUnreadableError

pytest.importorskip("fastmcp")

#: Generous ceiling for a cold daemon start in CI. The product deadline is the
#: configurable 10s startup timeout; this only bounds the test.
_START_DEADLINE_SECONDS = 60.0
#: Idle window the test daemon is started with. Short enough to observe, long
#: enough that a slow first tool call does not trip it mid-conversation.
_TEST_IDLE_SECONDS = 6.0
#: Ceiling on how long the idle shutdown may take once the last call is done.
_SHUTDOWN_DEADLINE_SECONDS = 60.0


def _spawn_daemon(user_dir: Path, *, idle_seconds: float = _TEST_IDLE_SECONDS) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "trw_memory.server",
            "serve",
            "http",
            "--idle-shutdown-seconds",
            str(idle_seconds),
        ],
        env={**os.environ, "TRW_USER_DIR": str(user_dir)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_exit(proc: subprocess.Popen[str]) -> int | None:
    """Block until the daemon exits on its idle window, or the ceiling passes."""
    try:
        return proc.wait(timeout=_SHUTDOWN_DEADLINE_SECONDS)
    except subprocess.TimeoutExpired:
        return None


def _await_discovery(paths: DaemonPaths, proc: subprocess.Popen[str]) -> DaemonInfo:
    deadline = time.monotonic() + _START_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"daemon exited early: {proc.communicate()[0]}")
        info = read_discovery(paths)
        if info is not None:
            return info
        time.sleep(0.05)
    proc.kill()
    pytest.fail("daemon never published a discovery file")


@pytest.fixture
def user_dir(tmp_path: Path) -> Path:
    return tmp_path / "userhome"


@pytest.fixture
def paths(user_dir: Path, monkeypatch: pytest.MonkeyPatch) -> DaemonPaths:
    monkeypatch.setenv("TRW_USER_DIR", str(user_dir))
    return DaemonPaths.resolve()


@pytest.fixture
def running_daemon(user_dir: Path, paths: DaemonPaths) -> Iterator[tuple[subprocess.Popen[str], DaemonInfo]]:
    proc = _spawn_daemon(user_dir)
    try:
        yield proc, _await_discovery(paths, proc)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)


async def _call(info: DaemonInfo, name: str, arguments: dict[str, object], *, token: str | None = None) -> object:
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    transport = StreamableHttpTransport(url=info.url, auth=token if token is not None else info.token)
    async with Client(transport) as client:
        result = await client.call_tool(name, arguments)
    return result.data


async def test_loopback_daemon_single_instance_token_and_idle_shutdown(
    user_dir: Path,
    paths: DaemonPaths,
    running_daemon: tuple[subprocess.Popen[str], DaemonInfo],
) -> None:
    """FR03 end to end: discovery, token, first call, second start, idle exit."""
    proc, info = running_daemon

    # Property 1 + 2: a loopback URL on an OS-assigned port, published at 0600.
    assert info.url.startswith(f"http://{LOOPBACK_HOST}:")
    assert info.url.split(":")[2].split("/")[0] != "0"
    assert paths.discovery.stat().st_mode & 0o777 == 0o600
    assert paths.token.stat().st_mode & 0o777 == 0o600
    assert paths.user_memory_dir.stat().st_mode & 0o777 == 0o700

    # The first served call succeeds over the transport.
    stored = await _call(
        info,
        "memory_store",
        {"content": "a learning served over loopback", "namespace": "project:daemon-aaaaaaaa"},
    )
    assert isinstance(stored, dict) and stored["status"] == "stored"

    # Property 3: a wrong token is rejected before any tool body runs.
    before = paths.token.read_text(encoding="utf-8")
    with pytest.raises(Exception, match="401"):
        await _call(info, "memory_status", {}, token="definitely-not-the-token")
    assert paths.token.read_text(encoding="utf-8") == before, "a rejection must not rotate the token"

    # FR01: EVERY namespace landed in the ONE user-space file. This is the
    # assertion the review asked for, and it is the one that fails if the daemon
    # stops pinning ``memory_single_store_path``: without it each namespace gets
    # its own SQLite file under the user directory and ``DaemonPaths.store`` is
    # a path no write path ever opens.
    for namespace in ("project:second-bbbbbbbb", "user:local"):
        landed = await _call(info, "memory_store", {"content": f"row for {namespace}", "namespace": namespace})
        assert isinstance(landed, dict) and landed["status"] == "stored"
    # Scoped to the MEMORY store filename: the tier layer keeps its own
    # ``warm.db`` sidecars, which are not the corpus this requirement is about.
    stores = sorted(path.relative_to(paths.user_memory_dir) for path in paths.user_memory_dir.rglob("memory.db"))
    assert stores == [Path("memory.db")], f"expected exactly one memory store, found {stores}"
    assert paths.store.exists()

    # Property 4: a second start neither binds nor clobbers the discovery file.
    second = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "trw_memory.server", "serve", "http"],
        env={**os.environ, "TRW_USER_DIR": str(user_dir)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "already running" in second.stderr
    assert read_discovery(paths) == info, "the second start rewrote the first daemon's record"

    # Property 5: the idle window elapses, the process exits and cleans up.
    exit_code = await asyncio.to_thread(_wait_for_exit, proc)
    assert exit_code == 0, "the daemon did not exit on its idle window"
    assert read_discovery(paths) is None
    assert not paths.discovery.exists()


async def test_a_wrong_token_reads_and_writes_nothing(
    paths: DaemonPaths,
    running_daemon: tuple[subprocess.Popen[str], DaemonInfo],
) -> None:
    """FR03 property 3: rejection happens before the tool body, so no row moves."""
    _proc, info = running_daemon
    namespace = "project:rejected-bbbbbbbb"

    with pytest.raises(Exception, match="401"):
        await _call(info, "memory_store", {"content": "must not land", "namespace": namespace}, token="wrong")

    # The store directory for that namespace was never created, which is only
    # possible if no tool body ran.
    assert not (paths.user_memory_dir / namespace.replace(":", "_")).exists()


def test_stale_lock_reaped_only_when_pid_is_dead(paths: DaemonPaths) -> None:
    """NFR02: reaping is conditional on evidence the recorded process is gone.

    ``claim_single_instance`` treats a record naming THIS pid as its own (which
    is what lets one process restart its own daemon), so the live-holder branch
    is exercised with the parent pid: a process that is definitely alive and
    definitely not us.
    """
    token = ensure_token(paths)

    live = claim_single_instance(paths, port=0, token=token, version="test")
    try:
        held_by_another = live.info.model_copy(update={"pid": os.getppid()})
        paths.discovery.write_text(held_by_another.model_dump_json(), encoding="utf-8")

        with pytest.raises(DaemonAlreadyRunningError, match="already running"):
            claim_single_instance(paths, port=0, token=token, version="test")
        assert read_discovery(paths) == held_by_another, "a refused start must not touch the record"
    finally:
        live.sock.close()

    # Now record a pid that cannot be alive; the next claim reaps and rebinds.
    paths.discovery.write_text(live.info.model_copy(update={"pid": _unused_pid()}).model_dump_json(), encoding="utf-8")

    reclaimed = claim_single_instance(paths, port=0, token=token, version="test")
    try:
        assert reclaimed.info.pid == os.getpid()
    finally:
        reclaimed.sock.close()


def _unused_pid() -> int:
    """Return a pid no live process can hold: a short-lived child, reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    return proc.pid


def test_a_non_loopback_bind_is_refused_before_any_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """NFR03: the address never reaches ``bind``, and the error names it."""
    created: list[str] = []
    real_socket = socket.socket

    def _tracking_socket(*args: object, **kwargs: object) -> socket.socket:
        created.append("socket")
        return real_socket(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket, "socket", _tracking_socket)

    for rejected in ("0.0.0.0", "192.168.1.10", "::", "example.com"):
        with pytest.raises(ConfigError, match=rejected):
            bind_loopback_socket(0, host=rejected)
    assert created == [], "a socket was created for a non-loopback address"

    assert require_loopback(LOOPBACK_HOST) == LOOPBACK_HOST
    assert require_loopback("127.0.0.5") == "127.0.0.5"


def test_bound_socket_reports_an_ephemeral_loopback_port() -> None:
    """port=0 must yield a real, readable port before the server starts."""
    sock = bind_loopback_socket(0)
    try:
        host, port = sock.getsockname()[:2]
        assert host == LOOPBACK_HOST
        assert port > 0
    finally:
        sock.close()


def test_token_is_generated_once_at_0600_and_compared_in_constant_time(paths: DaemonPaths) -> None:
    """FR03 property 3 + FR08 clause 2."""
    assert not paths.token.exists()

    token = ensure_token(paths)

    assert len(token) >= 32
    assert paths.token.stat().st_mode & 0o777 == 0o600
    assert ensure_token(paths) == token, "a second call must not mint a new token"
    assert tokens_match(token, token)
    assert not tokens_match(token, "wrong")
    assert not tokens_match("", token)


def test_discovery_read_is_defensive(paths: DaemonPaths) -> None:
    """Malformed, wrong-schema and absent records are all 'no daemon'."""
    assert read_discovery(paths) is None

    paths.user_memory_dir.mkdir(parents=True, exist_ok=True)
    paths.discovery.write_text("not json{", encoding="utf-8")
    assert read_discovery(paths) is None

    paths.discovery.write_text(json.dumps({"schema_version": 99, "pid": 1}), encoding="utf-8")
    assert read_discovery(paths) is None

    paths.discovery.write_text(json.dumps({"schema_version": 1, "pid": -1}), encoding="utf-8")
    assert read_discovery(paths) is None


def test_discovery_repr_redacts_the_token() -> None:
    """NFR03: the token is never printed by a diagnostic path."""
    info = DaemonInfo(pid=1, url="http://127.0.0.1:1/mcp", token="s3cr3t", started_at="now", version="0")

    assert "s3cr3t" not in repr(info)
    assert "s3cr3t" not in str(info)
    assert "s3cr3t" not in f"{info}"


def test_secret_files_are_written_through_an_exclusive_temp_and_renamed(paths: DaemonPaths, tmp_path: Path) -> None:
    """NFR03: a pre-planted symlink at the destination is replaced, not followed."""
    from trw_memory.daemon._paths import read_secret_file, write_secret_file

    paths.user_memory_dir.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("original", encoding="utf-8")
    target = paths.user_memory_dir / "planted"
    target.symlink_to(victim)

    write_secret_file(target, "secret")

    assert victim.read_text(encoding="utf-8") == "original", "the write followed a symlink"
    assert not target.is_symlink()
    assert target.read_text(encoding="utf-8") == "secret"
    assert target.stat().st_mode & 0o777 == 0o600
    # And the read path refuses a symlink rather than following one -- loudly.
    # ``None`` is reserved for "absent", because every caller answers absent by
    # creating the file, which over a planted link would mint a new secret.
    target.unlink()
    target.symlink_to(victim)
    with pytest.raises(DaemonSecretUnreadableError, match="symlink"):
        read_secret_file(target)
    assert read_secret_file(paths.user_memory_dir / "never-written") is None


def test_the_daemon_refuses_to_start_under_encryption(monkeypatch: pytest.MonkeyPatch, user_dir: Path) -> None:
    """FR09 preflight: refuse at STARTUP, not at the second namespace.

    The daemon pins ``memory_single_store_path``, and a per-namespace SQLCipher
    key cannot open a shared file. Without this refusal the failure would land
    inside a served tool call for whichever namespace happened to be second --
    an unopenable store reported far from its cause.
    """
    import trw_memory.server as server_mod
    from trw_memory.exceptions import ConfigError

    monkeypatch.setenv("TRW_USER_DIR", str(user_dir))
    monkeypatch.setenv("MEMORY_ENCRYPTION_ENABLED", "true")
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)
    monkeypatch.setattr(server_mod, "_preflight", lambda _config: None)

    with pytest.raises(ConfigError) as refusal:
        server_mod._serve_http(None, None)

    message = str(refusal.value)
    assert "encryption_enabled" in message
    assert "FR09" in message
    assert "serve stdio" in message, "the refusal must name the mode that still works"
    # Nothing was claimed: no discovery file, no token, no store.
    paths = DaemonPaths.resolve(create=False)
    assert not paths.discovery.exists()
    assert not paths.token.exists()
