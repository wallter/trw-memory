"""DAEMON-AUTO-RESTART-ON-UPGRADE: a client replaces a major-older daemon only through its drain handshake.

Every daemon here is a real ``trw_memory.server serve http`` process under a
throwaway ``TRW_USER_DIR``, owner-tagged by the session conftest and reaped at
the end of each test. A daemon is made to look older by overriding the version
(and the capabilities) it publishes, never by installing an old wheel. The
safety bound is CONSTITUTION HB-2: nothing is stopped that is not proven, and
another session's in-flight write is never traded for the restart.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from trw_memory.daemon import DaemonPaths, _session, mint_grant
from trw_memory.daemon import client as client_module
from trw_memory.daemon._discovery import (
    AGENT_MUST_NOT_STOP,
    VERSION_HEADER,
    DaemonInfo,
    DiscoveryAbsent,
    read_discovery_result,
    read_live_discovery,
    refused_while_draining,
)
from trw_memory.daemon._paths import write_secret_file
from trw_memory.daemon._spawn import SpawnedDaemon
from trw_memory.daemon.client import DaemonClient, _package_version
from trw_memory.exceptions import DaemonVersionMismatchError
from trw_memory.models.config import MemoryConfig
from trw_memory.storage._pid_liveness import process_start

pytest.importorskip("fastmcp")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process identity")

_MINE = _package_version()
_MINE_MAJOR = int(_MINE.split(".")[0])
_OLDER = f"{_MINE_MAJOR - 1}.9.0"
_NEWER = f"{_MINE_MAJOR + 1}.0.0"
_NAMESPACE = "project:autorestart-aaaaaaaa"
_START_DEADLINE_SECONDS = 60.0

#: A real daemon whose published version and capabilities are overridden before it serves. The trailing
#: ``trw_memory.server serve http`` argv is only a marker: it puts the daemon's argv mark on the command line,
#: so the session reaper recognises this process as a daemon too.
_FAKED = textwrap.dedent(
    """
    import os
    import trw_memory._version as version_module
    import trw_memory.daemon._discovery as discovery
    version_module.__version__ = os.environ["TRW_TEST_DAEMON_VERSION"]
    discovery.DAEMON_CAPABILITIES = tuple(c for c in os.environ["TRW_TEST_DAEMON_CAPABILITIES"].split(",") if c)
    from trw_memory.server import main
    main(["serve", "http", "--idle-shutdown-seconds", "60"])
    """
)


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DaemonPaths:
    """A scratch user dir; the successor a client auto-starts inherits this environment."""
    user_dir = tmp_path / "userhome"
    model_dir = user_dir.resolve() / "unavailable-local-model"  # keyword-only: no model load, no Hub lookup
    model_dir.mkdir(parents=True)
    monkeypatch.setenv("TRW_USER_DIR", str(user_dir))
    monkeypatch.setenv("MEMORY_EMBEDDING_MODEL", str(model_dir))
    monkeypatch.setenv("MEMORY_DAEMON_IDLE_SHUTDOWN_SECONDS", "60")
    return DaemonPaths.resolve()


@pytest.fixture
def started(paths: DaemonPaths) -> Iterator[list[subprocess.Popen[bytes]]]:
    """Every daemon this test started, plus whatever serves at the end (a successor), stopped by pid."""
    processes: list[subprocess.Popen[bytes]] = []
    yield processes
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)
    successor = read_live_discovery(paths)
    if isinstance(successor, DaemonInfo):
        SpawnedDaemon(successor.pid, successor.process_start, paths.lock).stop()


def _start(
    paths: DaemonPaths, started: list[subprocess.Popen[bytes]], *, version: str, capabilities: str
) -> tuple[subprocess.Popen[bytes], DaemonInfo]:
    env = {**os.environ, "TRW_TEST_DAEMON_VERSION": version, "TRW_TEST_DAEMON_CAPABILITIES": capabilities}
    log = (paths.user_memory_dir.parent / f"daemon-{len(started)}.log").open("wb")
    process = subprocess.Popen(
        [sys.executable, "-c", _FAKED, "trw_memory.server", "serve", "http"], env=env, stdout=log, stderr=log
    )
    log.close()
    started.append(process)
    deadline = time.monotonic() + _START_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        assert process.poll() is None, f"the scratch daemon exited early (log in {log.name})"
        found = read_discovery_result(paths)
        if isinstance(found, DaemonInfo) and found.pid == process.pid:
            return process, found
        time.sleep(0.05)
    raise AssertionError("the scratch daemon never published")


def _client(paths: DaemonPaths, *, autostart: bool = True, instance: tuple[int, str] | None = None) -> DaemonClient:
    config = MemoryConfig(memory_daemon_autostart=autostart, memory_daemon_startup_timeout_seconds=30.0)
    return DaemonClient(mint_grant(paths, [_NAMESPACE]), config=config, paths=paths, instance=instance)


@pytest.fixture
def spawns(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Every auto-start the client makes, still performed for real."""
    calls: list[object] = []
    real = client_module.start_daemon_detached

    def record(paths: DaemonPaths) -> SpawnedDaemon:
        calls.append(paths)
        return real(paths)

    monkeypatch.setattr(client_module, "start_daemon_detached", record)
    return calls


def _assert_untouched(process: subprocess.Popen[bytes], info: DaemonInfo, paths: DaemonPaths) -> None:
    assert process.poll() is None, "a daemon the client may not replace was stopped"
    now = read_discovery_result(paths)
    assert isinstance(now, DaemonInfo)
    assert (now.pid, now.started_at) == (info.pid, info.started_at), "the record changed under a refusal"


async def _refused(client: DaemonClient) -> str:
    with pytest.raises(DaemonVersionMismatchError) as refused:
        await client.get("M-none", _NAMESPACE)
    message = str(refused.value)
    assert AGENT_MUST_NOT_STOP in message, "the refusal must stay addressed to the user"
    assert "kill" not in message.lower()
    return message


# (b) ------------------------------------------------------------------------------------------------------------
async def test_a_pinned_client_refuses_before_any_restart_action(
    paths: DaemonPaths, started: list[subprocess.Popen[bytes]], spawns: list[object]
) -> None:
    process, info = _start(paths, started, version=_OLDER, capabilities="drain")

    message = await _refused(_client(paths, instance=(info.pid, info.started_at)))

    assert "pinned" in message
    _assert_untouched(process, info, paths)
    assert spawns == []


# (c) ------------------------------------------------------------------------------------------------------------
async def test_autostart_off_refuses_and_reports_without_stopping(
    paths: DaemonPaths, started: list[subprocess.Popen[bytes]], spawns: list[object]
) -> None:
    """A supervisor owns the daemon when auto-start is off: the client reports and never drains it."""
    process, info = _start(paths, started, version=_OLDER, capabilities="drain")

    message = await _refused(_client(paths, autostart=False))

    assert "MEMORY_DAEMON_AUTOSTART=false" in message
    _assert_untouched(process, info, paths)
    assert spawns == []


# (d) ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("version", "capabilities", "reason"),
    [
        (_OLDER, "", "drain handshake"),  # every 4.x/5.0.0 daemon: no capability advertised
        (_NEWER, "drain", "not a major version older"),  # a newer daemon is never stopped
    ],
)
async def test_a_daemon_that_cannot_be_drained_keeps_todays_refusal(
    paths: DaemonPaths,
    started: list[subprocess.Popen[bytes]],
    spawns: list[object],
    version: str,
    capabilities: str,
    reason: str,
) -> None:
    process, info = _start(paths, started, version=version, capabilities=capabilities)

    message = await _refused(_client(paths))

    assert f"serves {version}" in message
    assert reason in message
    _assert_untouched(process, info, paths)
    assert spawns == []


# (e) ------------------------------------------------------------------------------------------------------------
async def test_a_proven_older_daemon_with_drain_is_drained_and_a_new_daemon_serves_the_call(
    paths: DaemonPaths, started: list[subprocess.Popen[bytes]], spawns: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.daemon import _upgrade

    process, info = _start(paths, started, version=_OLDER, capabilities="drain")
    assert info.capabilities == ["drain"]
    answers: list[dict[str, object]] = []
    real_drain = _upgrade._request_drain

    def recorded(*args: object) -> dict[str, object]:
        answers.append(real_drain(*args))  # type: ignore[arg-type]
        return answers[-1]

    monkeypatch.setattr(_upgrade, "_request_drain", recorded)

    answer = await _client(paths).get("M-none", _NAMESPACE)

    assert isinstance(answer, dict), answer
    assert process.wait(timeout=30) is not None, "the drained daemon did not exit"
    successor = read_live_discovery(paths)
    assert isinstance(successor, DaemonInfo)
    assert successor.version == _MINE
    assert successor.pid != info.pid
    assert len(spawns) == 1, "exactly one successor is started, through the normal auto-start path"
    assert [(a["status"], a["pid"]) for a in answers] == [("drained", info.pid)], "the drain itself was not answered"


# (f) ------------------------------------------------------------------------------------------------------------
def _store_request(token: str, entry_id: str) -> tuple[dict[str, str], bytes]:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "memory_store",
            "arguments": {"content": "the other session's write", "namespace": _NAMESPACE, "entry_id": entry_id},
        },
    }
    headers = {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
        VERSION_HEADER: _MINE,
    }
    return headers, json.dumps(body).encode()


async def _held_store(info: DaemonInfo, token: str, entry_id: str, release: asyncio.Event) -> httpx.Response:
    """A write from another session that stays in flight until *release*: its body arrives in two parts."""
    headers, body = _store_request(token, entry_id)

    async def parts() -> AsyncIterator[bytes]:
        yield body[:10]
        await release.wait()
        yield body[10:]

    async with httpx.AsyncClient(timeout=90) as http:
        return await http.post(info.url, headers=headers, content=parts())


def _key_file(paths: DaemonPaths) -> Path:
    """Where the daemon keeps its drain key: spelled out, so the tests do not trust the code under test."""
    return paths.user_memory_dir / "drain.key"


async def _drain(
    info: DaemonInfo, token: str, deadline_seconds: float, *, version: str = _NEWER, key: str | None = None
) -> object:
    """The production drain request, from a client of *version* holding *key* (default: the daemon's own)."""
    from trw_memory.daemon._drain_key import read_drain_key
    from trw_memory.daemon._upgrade import _request_drain

    admin_key = key if key is not None else read_drain_key(DaemonPaths.resolve())
    return await asyncio.to_thread(_request_drain, info, token, version, deadline_seconds, admin_key or "")


async def _raw_drain(info: DaemonInfo, token: str, arguments: dict[str, object]) -> dict[str, object]:
    """``memory_drain`` as any namespace-token holder can send it, with a newer-major header."""
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "memory_drain", "arguments": arguments},
    }
    headers = {
        "authorization": f"Bearer {token}",
        "accept": "application/json, text/event-stream",
        VERSION_HEADER: _NEWER,
    }
    async with httpx.AsyncClient(timeout=60) as http:
        response = await http.post(info.url, headers=headers, json=body)
    result = response.json().get("result") or {}
    if result.get("isError"):
        return {"status": "declined", "detail": json.dumps(result.get("content"))}
    answer = result.get("structuredContent")
    return answer if isinstance(answer, dict) else {"status": "unparseable", "detail": response.text}


async def test_drain_waits_for_an_in_flight_write_before_exiting(
    paths: DaemonPaths, started: list[subprocess.Popen[bytes]]
) -> None:
    process, info = _start(paths, started, version=_MINE, capabilities="drain")
    token = mint_grant(paths, [_NAMESPACE])
    release = asyncio.Event()
    store = asyncio.create_task(_held_store(info, token, "M-inflightwrite01", release))
    await asyncio.sleep(1.0)  # the write is in flight: its body is half sent

    drain = asyncio.create_task(_drain(info, token, 30.0))
    await asyncio.sleep(1.5)
    assert not drain.done(), "the drain returned while another session's write was still in flight"
    release.set()
    written = await store
    drained = await drain

    assert written.status_code == 200 and '"stored"' in written.text, written.text
    assert isinstance(drained, dict) and drained["status"] == "drained", drained
    process.wait(timeout=30)
    assert isinstance(read_live_discovery(paths), DiscoveryAbsent), "the drained daemon left its record"
    assert not _key_file(paths).exists(), "the drained daemon left its drain key"
    survived = await _client(paths).get("M-inflightwrite01", _NAMESPACE)  # served by an auto-started successor
    assert "the other session's write" in json.dumps(survived)


async def test_a_drain_whose_deadline_passes_with_a_write_in_flight_resumes_serving(
    paths: DaemonPaths, started: list[subprocess.Popen[bytes]]
) -> None:
    """HB-2: the drain never trades another session's write for the restart; it gives up and keeps serving."""
    process, info = _start(paths, started, version=_MINE, capabilities="drain")
    token = mint_grant(paths, [_NAMESPACE])
    release = asyncio.Event()
    store = asyncio.create_task(_held_store(info, token, "M-inflightwrite02", release))
    await asyncio.sleep(1.0)

    drained = await _drain(info, token, 1.0)
    release.set()
    written = await store

    assert isinstance(drained, dict) and drained["status"] == "busy", drained
    assert written.status_code == 200 and '"stored"' in written.text, written.text
    _assert_untouched(process, info, paths)
    assert "the other session's write" in json.dumps(await _client(paths).get("M-inflightwrite02", _NAMESPACE))


@pytest.mark.parametrize("caller", [_OLDER, "unversioned", ""])
async def test_a_drain_from_an_older_or_unversioned_client_is_refused(
    paths: DaemonPaths, started: list[subprocess.Popen[bytes]], caller: str
) -> None:
    """The daemon's own gate: an older or unversioned client can never retire it, whatever the client decides."""
    process, info = _start(paths, started, version=_MINE, capabilities="drain")

    answer = await _drain(info, mint_grant(paths, [_NAMESPACE]), 1.0, version=caller)

    assert isinstance(answer, dict) and answer["status"] == "declined", answer
    assert "drain_refused" in str(answer["detail"])
    _assert_untouched(process, info, paths)


# (g) ------------------------------------------------------------------------------------------------------------
@pytest.fixture
def sleeper() -> Iterator[subprocess.Popen[bytes]]:
    """A live process standing in for a daemon whose record does not withdraw after its drain."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield process
    process.kill()
    process.wait()


def test_the_bounded_wait_refuses_when_the_record_does_not_withdraw(
    paths: DaemonPaths, sleeper: subprocess.Popen[bytes], spawns: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.daemon import _upgrade

    paths.user_memory_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = DaemonInfo(
        pid=sleeper.pid,
        url="http://127.0.0.1:9/mcp",
        started_at=datetime.now(timezone.utc).isoformat(),
        version=_OLDER,
        process_start=process_start(sleeper.pid),
        capabilities=["drain"],
    )
    write_secret_file(paths.discovery, info.model_dump_json())
    write_secret_file(_key_file(paths), "a" * 64)
    drains: list[str] = []

    def answered_drained(found: DaemonInfo, *_args: object) -> dict[str, object]:
        drains.append(found.url)
        return {"status": "drained"}  # the daemon said yes, and then its record stayed

    monkeypatch.setattr(_upgrade, "_request_drain", answered_drained)
    config = MemoryConfig(memory_daemon_startup_timeout_seconds=0.5)
    client = DaemonClient(mint_grant(paths, [_NAMESPACE]), config=config, paths=paths)

    started_at = time.monotonic()
    with pytest.raises(DaemonVersionMismatchError, match="did not withdraw") as refused:
        client._attach()

    assert time.monotonic() - started_at < 10, "the wait for withdrawal is not bounded"
    assert drains == [info.url]
    assert spawns == [], "a daemon was started beside a live record"
    assert sleeper.poll() is None, "the client signalled the process instead of waiting"
    assert AGENT_MUST_NOT_STOP in str(refused.value)


# (h) ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arguments", [{"deadline_seconds": 1.0}, {"deadline_seconds": 1.0, "admin_key": "0" * 64}])
async def test_a_namespace_token_cannot_drain_without_the_daemon_key(
    paths: DaemonPaths, started: list[subprocess.Popen[bytes]], arguments: dict[str, object]
) -> None:
    """Review r1 P1: a checkout grant plus a forged newer-major header must not stop the shared daemon."""
    process, info = _start(paths, started, version=_MINE, capabilities="drain")
    token = mint_grant(paths, [_NAMESPACE])

    answer = await _raw_drain(info, token, arguments)

    assert answer["status"] == "declined", answer
    assert "drain_refused" in str(answer["detail"])
    assert "0" * 64 not in str(answer["detail"]), "the refusal echoed the key it was given"
    _assert_untouched(process, info, paths)
    assert isinstance(await _client(paths).get("M-none", _NAMESPACE), dict), "the daemon stopped serving"


async def test_an_in_process_drain_without_the_key_changes_nothing() -> None:
    """The reviewer's reproduction: no HTTP request, so no version gate; the key alone must refuse."""
    import types

    from trw_memory.daemon import _drain

    door = types.SimpleNamespace(in_flight=1, draining=False)
    server = types.SimpleNamespace(should_exit=False)
    _drain.arm_drain(door, server, "k" * 64, None)
    try:
        for wrong in ("", "x" * 64):
            with pytest.raises(_drain.DrainRefusedError, match="drain_refused"):
                await _drain.drain(1.0, wrong)
    finally:
        _drain.disarm_drain()
    assert (door.draining, server.should_exit) == (False, False)


# (i) ------------------------------------------------------------------------------------------------------------
async def test_the_drain_key_is_user_only_and_removed_on_exit(
    paths: DaemonPaths, started: list[subprocess.Popen[bytes]]
) -> None:
    import stat

    process, info = _start(paths, started, version=_MINE, capabilities="drain")
    key = _key_file(paths)

    assert info.capabilities == ["drain"]
    assert not key.is_symlink() and stat.S_IMODE(key.stat().st_mode) == 0o600
    assert len(key.read_text(encoding="utf-8").strip()) == 64
    process.terminate()
    process.wait(timeout=30)
    assert not key.exists(), "the daemon left its drain key behind"
    assert isinstance(read_live_discovery(paths), DiscoveryAbsent)


async def test_a_daemon_without_drain_writes_no_key(paths: DaemonPaths, started: list[subprocess.Popen[bytes]]) -> None:
    _process, info = _start(paths, started, version=_OLDER, capabilities="")

    assert info.capabilities == []
    assert not _key_file(paths).exists()


# (j) ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tamper", ["absent", "group-readable", "symlink"])
async def test_a_client_that_cannot_read_the_key_keeps_the_refusal(
    paths: DaemonPaths, started: list[subprocess.Popen[bytes]], spawns: list[object], tamper: str
) -> None:
    process, info = _start(paths, started, version=_OLDER, capabilities="drain")
    key = _key_file(paths)
    if tamper == "absent":
        key.unlink()
    elif tamper == "group-readable":
        key.chmod(0o640)
    else:
        real = key.with_name("elsewhere.key")
        key.rename(real)
        key.symlink_to(real)

    message = await _refused(_client(paths))

    assert "drain key" in message
    _assert_untouched(process, info, paths)
    assert spawns == []


def test_a_corrupt_drain_key_reads_as_no_key(tmp_path: Path) -> None:
    """Review r2: an owner-only key file holding non-ASCII bytes must refuse, not raise out of attach."""
    from trw_memory.daemon._drain_key import read_drain_key

    corrupt_paths = DaemonPaths(user_memory_dir=tmp_path)
    corrupt_paths.drain_key.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(corrupt_paths.drain_key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(b"\xff\xfe not a key")

    assert read_drain_key(corrupt_paths) is None


# (i) ------------------------------------------------------------------------------------------------------------
async def test_a_write_refused_by_a_draining_daemon_waits_for_the_successor_instead_of_may_have_been_applied(
    paths: DaemonPaths, started: list[subprocess.Popen[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """DRAIN-503-NEVER-SENT: the drain door answers 503 before the app sees the request, so it was never applied.

    A non-replayable write (``forget``) sent while another session's drain waits on an in-flight write used to
    fail at once with "may have been applied". It is refused at the door, so the client waits for the draining
    record to withdraw and retries on the successor it auto-starts.
    """
    process, info = _start(paths, started, version=_MINE, capabilities="drain")
    token = mint_grant(paths, [_NAMESPACE])
    seeded = await _client(paths).store("seeded before the drain", _NAMESPACE, entry_id="M-drainforget0001")
    assert seeded is not None
    release = asyncio.Event()
    store = asyncio.create_task(_held_store(info, token, "M-inflightwrite03", release))
    await asyncio.sleep(1.0)
    drain = asyncio.create_task(_drain(info, token, 30.0))
    await asyncio.sleep(1.0)  # the door is closed; the drain waits on the held write

    refusals: list[bool] = []
    real_never_sent = _session.never_sent

    def observed(exc: BaseException) -> bool:
        refusals.append(any(refused_while_draining(e) for e in _session.chain(exc)))
        return real_never_sent(exc)

    monkeypatch.setattr(client_module, "never_sent", observed)  # the name client.call_tool resolves
    assert not drain.done(), "the drain finished before the write was sent: nothing would be refused"
    forget = asyncio.create_task(_client(paths).forget("M-drainforget0001", _NAMESPACE))
    await asyncio.sleep(1.0)
    assert refusals and refusals[0], "the write was never refused at the drain door, so the test proves nothing"
    release.set()
    assert (await store).status_code == 200
    assert (await drain)["status"] == "drained"  # type: ignore[index]

    await forget  # served by the successor; a "may have been applied" DaemonUnreachableError fails the test
    process.wait(timeout=30)
    successor = read_live_discovery(paths)
    assert isinstance(successor, DaemonInfo) and successor.pid != info.pid
    assert (await _client(paths).get("M-drainforget0001", _NAMESPACE))["status"] == "not_found", "forget did not land"


def _door_503(method: str | None, *, marked: bool = True) -> RuntimeError:
    """A transport failure wrapping a 503 on a request whose JSON-RPC *method* is given (None: no body)."""
    body = b"" if method is None else json.dumps({"jsonrpc": "2.0", "id": 1, "method": method}).encode()
    request = httpx.Request("POST", "http://127.0.0.1:9/mcp", content=body)
    response = httpx.Response(503, headers={"x-trw-memory-draining": "1"} if marked else {}, request=request)
    wrapped = RuntimeError("transport failed")
    wrapped.__cause__ = httpx.HTTPStatusError("503", request=request, response=response)
    return wrapped


@pytest.mark.parametrize(
    ("error", "never_sent"),
    [
        (_door_503("tools/call"), True),  # the door refused the call itself: nothing ran
        (_door_503("initialize"), True),  # refused before any call in the session
        (_door_503("notifications/initialized"), True),
        # DRAIN-503-REQUEST-SCOPE: mcp's ClientSession sends tools/list AFTER a successful tools/call when the
        # output schema is not cached; a refusal there means the call before it WAS applied.
        (_door_503("tools/list"), False),
        (_door_503(None), False),  # no readable body: nothing proves the refused request preceded the effect
        (_door_503("tools/call", marked=False), False),  # the MCP session manager's own 503, or a pre-5.0.1 daemon's
    ],
    ids=["call", "initialize", "initialized", "post-call-list", "no-body", "unmarked"],
)
def test_only_a_marked_503_on_a_request_before_any_effect_counts_as_never_sent(
    error: RuntimeError, never_sent: bool
) -> None:
    assert _session.never_sent(error) is never_sent


async def test_a_write_whose_post_call_tools_list_is_refused_by_a_drain_is_not_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write ran; only the schema listing after it met the closed door. A replay would apply it twice."""
    calls: list[str] = []
    info = DaemonInfo(
        pid=os.getpid(),
        url="http://127.0.0.1:9/mcp",
        started_at=datetime.now(timezone.utc).isoformat(),
        version=_MINE,
        process_start=process_start(os.getpid()),
    )

    async def applied_then_refused(_self: object, _info: object, name: str, _arguments: object) -> object:
        calls.append(name)
        raise _door_503("tools/list")

    monkeypatch.setattr(DaemonClient, "_attach", lambda _self: info)
    monkeypatch.setattr(DaemonClient, "_call_once", applied_then_refused)
    monkeypatch.setattr(client_module, "withdrawn", lambda *_args: True)
    client = DaemonClient("token", config=MemoryConfig(memory_daemon_startup_timeout_seconds=0.1))

    with pytest.raises(client_module.DaemonUnreachableError, match="may have been applied"):
        await client.forget("M-applied00000001", _NAMESPACE)

    assert calls == ["memory_forget"], "the applied write was replayed"
