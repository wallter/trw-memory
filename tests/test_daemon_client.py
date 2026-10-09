"""PRD-CORE-253 FR08 — the client fails closed, and says how to fix it.

Unreachability is produced for real here, not simulated: a discovery record
naming a LIVE process (this one) and a port nothing is listening on is exactly
the state a killed daemon or a reused pid leaves behind, and it is the state in
which a fail-open client would quietly return an empty recall.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import structlog

from trw_memory.daemon import (
    DaemonInfo,
    DaemonPaths,
    DiscoveryAbsent,
    _direct,
    _session,
    mint_grant,
    read_live_discovery,
)
from trw_memory.daemon import client as client_module
from trw_memory.daemon._discovery import AGENT_MUST_NOT_STOP
from trw_memory.daemon.client import DAEMON_START_COMMAND, DaemonClient
from trw_memory.exceptions import DaemonAuthError, DaemonProtocolError, DaemonUnreachableError
from trw_memory.models.config import MemoryConfig
from trw_memory.testing.daemon_reaper import reap_daemons_under

from ._test_daemon_support import read_discovery

pytest.importorskip("fastmcp")

_START_DEADLINE_SECONDS = 60.0

#: The daemon's idle-shutdown window for these tests. Same reasoning as
#: ``test_cli_namespace``: the daemon loads its embedding model LAZILY, on the
#: first tool call rather than at boot, so in an environment that has
#: `sentence-transformers` installed -- which the mirror CI does and a typical
#: dev venv does not -- that first call pays the model load. Measured at 13.5s
#: against a warm cache; unbounded against a cold one, where it is a download.
#: At 15s the daemon shut itself down mid-call and the client reported a
#: ConnectError against a daemon that had been alive moments earlier, so the
#: window has to outlast the SLOWEST legitimate first call, not the typical one.
_TEST_IDLE_SECONDS = 120.0


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
        started_at="2026-09-03T00:00:00+00:00",
        version="test",
    )
    paths.discovery.write_text(info.model_dump_json(), encoding="utf-8")


def _spawn_daemon(paths: DaemonPaths, provisioned_embedding_cache: str) -> subprocess.Popen[bytes]:
    """Start a real daemon against *paths*; does not wait for it to publish discovery."""
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "trw_memory.server",
            "serve",
            "http",
            "--idle-shutdown-seconds",
            str(_TEST_IDLE_SECONDS),
        ],
        env={
            **os.environ,
            "TRW_USER_DIR": str(paths.user_memory_dir.parent),
            "HF_HUB_CACHE": provisioned_embedding_cache,
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _await_discovery(paths: DaemonPaths, proc: subprocess.Popen[bytes]) -> DaemonInfo:
    """Block until *proc* (already spawned against *paths*) publishes its discovery record."""
    deadline = time.monotonic() + _START_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail("daemon exited during startup")
        info = read_discovery(paths)
        if info is not None:
            return info
        time.sleep(0.05)
    pytest.fail("daemon never published a discovery file")


def _spawn_and_await_daemon(paths: DaemonPaths, provisioned_embedding_cache: str) -> Iterator[DaemonInfo]:
    """Shared body: spawn a real daemon against ``paths`` and wait for discovery."""
    proc = _spawn_daemon(paths, provisioned_embedding_cache)
    try:
        yield _await_discovery(paths, proc)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)


@pytest.fixture
def running_daemon(paths: DaemonPaths, provisioned_embedding_cache: str) -> Iterator[DaemonInfo]:
    """Function-scoped: for the one test that must observe an EMPTY grants file.

    ``test_a_rejected_token_fails_closed_without_minting`` asserts
    ``not paths.grants.exists()`` -- a guarantee only a daemon nothing else has
    minted a grant against can make. Every other daemon-backed test in this
    module shares ``shared_running_daemon`` instead (see below).
    """
    yield from _spawn_and_await_daemon(paths, provisioned_embedding_cache)


@pytest.fixture(scope="module")
def shared_daemon_home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One on-disk home for every test in this module that shares a daemon."""
    return tmp_path_factory.mktemp("shared-daemon-home")


@pytest.fixture(scope="module")
def shared_paths(shared_daemon_home: Path) -> Iterator[DaemonPaths]:
    """``DaemonPaths`` resolved against the module-shared home.

    ``monkeypatch`` is function-scoped, so a module-scoped fixture cannot use
    it; ``pytest.MonkeyPatch()`` is the same mechanism used directly, with an
    explicit ``undo()`` at module teardown.
    """
    mp = pytest.MonkeyPatch()
    mp.setenv("TRW_USER_DIR", str(shared_daemon_home))
    try:
        yield DaemonPaths.resolve()
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def _shared_daemon_box(
    shared_paths: DaemonPaths, provisioned_embedding_cache: str
) -> Iterator[list[tuple[subprocess.Popen[bytes], DaemonInfo]]]:
    """A one-element mutable box holding the live shared daemon's (process, discovery).

    Module-scoped so the box (and whatever it currently holds) survives across every test
    in this module; the box's *contents* are replaced in place by ``shared_running_daemon``
    below when a liveness check finds the daemon dead, rather than by re-running this
    fixture (module-scoped fixtures run exactly once per module).
    """
    box: list[tuple[subprocess.Popen[bytes], DaemonInfo]] = []
    try:
        yield box
    finally:
        if box:
            proc, _ = box[0]
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)


@pytest.fixture
def shared_running_daemon(
    shared_paths: DaemonPaths,
    provisioned_embedding_cache: str,
    _shared_daemon_box: list[tuple[subprocess.Popen[bytes], DaemonInfo]],
) -> DaemonInfo:
    """The module-shared daemon, checked for liveness and restarted if dead before every test.

    One real daemon is reused by every test below that:

    - only reads/writes through namespace-scoped calls (each test below uses a
      distinct namespace suffix -- e.g. ``roundtrip-``, ``lost-e``, ``held-1`` --
      so no two tests observe each other's rows), or
    - replaces ``fastmcp.Client`` (what a session call opens) entirely (the ``gated_sessions``
      tests) and only needs a discovery record naming a genuinely live pid.

    Excluded: the rejected-token test (needs a grants file no other test has
    touched) and the version-mismatch/attach tests (fabricate their own
    discovery record and never start a subprocess).

    ``memory_daemon_idle_shutdown_seconds`` (and the CLI override this module always
    passes, ``--idle-shutdown-seconds``) cannot be disabled --
    ``DaemonServeOptions.idle_shutdown_seconds`` is a bare ``Field(gt=0.0)``
    (``trw_memory/src/trw_memory/daemon/_serve.py``) with no "never shut down" value. A gap
    between two tests in this module longer than ``_TEST_IDLE_SECONDS`` can therefore let
    the daemon shut itself down between tests that each individually run well inside the
    window, leaving ``shared_paths.discovery`` naming a pid that is no longer listening --
    exactly the stale-discovery-record failure mode this fixture exists to prevent. A single
    spawn at module setup trusts that gap never happens for the whole module's test run;
    this checks and restarts instead, every time a dependent test asks for the daemon, using
    ``_shared_daemon_box`` (module-scoped) to hold the current (process, discovery) pair
    across the function-scoped liveness check below.
    """
    if _shared_daemon_box:
        proc, info = _shared_daemon_box[0]
        if proc.poll() is None:
            return info
        proc.wait(timeout=30)  # already exited; reap it before replacing
        _shared_daemon_box.clear()
    proc = _spawn_daemon(shared_paths, provisioned_embedding_cache)
    try:
        info = _await_discovery(shared_paths, proc)
    except BaseException:
        # Not yet tracked by teardown: stop it here, or it outlives the module.
        proc.kill()
        proc.wait(timeout=30)
        raise
    _shared_daemon_box.append((proc, info))
    return info


async def test_daemon_unreachable_fails_closed_with_actionable_error(paths: DaemonPaths, config: MemoryConfig) -> None:
    """FR08: BOTH a read and a write fail; no store is created; the remedy is named."""
    _record_an_unreachable_daemon(paths)
    client = DaemonClient("any-grant", config=config, paths=paths)

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
    client = DaemonClient("any-grant", config=config, paths=paths)

    with structlog.testing.capture_logs() as logs:
        with pytest.raises(DaemonUnreachableError):
            await client.recall("anything", "project:closed-aaaaaaaa")

    attempts = [entry["attempt"] for entry in logs if entry.get("event") == "daemon_call_failed"]
    assert attempts == [1, 2], f"expected exactly two attempts, saw {attempts}"


@pytest.mark.parametrize(("bound_to", "dialled"), [("2026-09-03T00:00:00+00:00", [1, 2]), ("another start", [])])
async def test_a_bound_client_dials_only_the_daemon_it_was_bound_to(
    bound_to: str, dialled: list[int], paths: DaemonPaths, config: MemoryConfig
) -> None:
    """PRD-CORE-298 FR07: a daemon other than the checked one is refused before any request, retries included."""
    _record_an_unreachable_daemon(paths)
    client = DaemonClient("any-grant", config=config, paths=paths, instance=(os.getpid(), bound_to))

    with structlog.testing.capture_logs() as logs:
        with pytest.raises(DaemonUnreachableError) as refused:
            await client.recall("anything", "project:closed-aaaaaaaa")

    assert [entry["attempt"] for entry in logs if entry.get("event") == "daemon_call_failed"] == dialled
    assert ("replaced" in str(refused.value)) is not dialled


async def test_a_rejected_token_fails_closed_without_minting(
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo
) -> None:
    """FR08 clause 3: rejection is a distinct, non-retried failure that mints nothing."""
    wrong = "a-token-this-daemon-never-granted"
    client = DaemonClient(wrong, config=config, paths=paths)

    with structlog.testing.capture_logs() as logs:
        with pytest.raises(DaemonAuthError, match="nothing was re-minted"):
            await client.recall("anything", "project:auth-bbbbbbbb")

    assert not paths.grants.exists(), "a rejection minted a grant"
    assert [entry for entry in logs if entry.get("event") == "daemon_call_failed"] == [], "a rejection was retried"
    assert wrong not in str(logs), "the token leaked into a log event"


async def test_a_live_daemon_serves_reads_and_writes_over_loopback(
    shared_paths: DaemonPaths, config: MemoryConfig, shared_running_daemon: DaemonInfo
) -> None:
    """The positive path: the same client that fails closed also works."""
    namespace = "project:roundtrip-dddddddd"
    client = DaemonClient(mint_grant(shared_paths, [namespace]), config=config, paths=shared_paths)

    stored = await client.store("a learning written through the daemon client", namespace)
    assert stored["status"] == "stored"

    recalled = await client.recall("learning written through", namespace)
    assert isinstance(recalled, dict)


async def test_memory_anchored_served_by_real_daemon(
    shared_paths: DaemonPaths, config: MemoryConfig, shared_running_daemon: DaemonInfo
) -> None:
    """PRD-CORE-332 FR04: a real daemon answers ``memory_anchored`` for a row stored with anchors."""
    namespace = "project:anchored-eeeeeeee"
    client = DaemonClient(mint_grant(shared_paths, [namespace]), config=config, paths=shared_paths)
    anchors = [{"file": "./pkg/mod.py", "symbol_name": "handler"}]
    stored = await client.store("a lesson whose text names no file", namespace, learning={"anchors": anchors})
    await client.store("an unanchored lesson", namespace)

    page = await client.anchored(namespace, "pkg/mod.py", 10, status="active")

    assert page["status"] == "ok"
    assert [row["id"] for row in page["memories"]] == [stored["memory_id"]]
    assert page["memories"][0]["anchors"][0]["file"] == "./pkg/mod.py"


async def test_memory_store_over_a_real_daemon_carries_anchors_and_distill_source(
    shared_paths: DaemonPaths, config: MemoryConfig, shared_running_daemon: DaemonInfo
) -> None:
    """The daemon's store tool takes the same ``anchors`` and the ``distill`` source the library client now does."""
    namespace = "project:distill-ffffffff"
    client = DaemonClient(mint_grant(shared_paths, [namespace]), config=config, paths=shared_paths)
    anchors = [{"file": "pkg/mod.py", "symbol_name": "handler"}]
    stored = await client.store("a mined lesson", namespace, learning={"anchors": anchors, "source": "distill"})

    page = await client.anchored(namespace, "pkg/mod.py", 10, status="active")

    assert [row["id"] for row in page["memories"]] == [stored["memory_id"]]
    assert page["memories"][0]["source"] == "distill"


async def test_a_record_still_refusing_past_the_deadline_names_its_holder(
    paths: DaemonPaths, config: MemoryConfig
) -> None:
    """PRD-CORE-310 FR02: a live process that holds the record and refuses every connection is named.

    This record carries no start (a 4.0 daemon's), so the pid is not proven to be the
    daemon: the remedy says to check it before killing it.
    """
    _record_an_unreachable_daemon(paths)
    client = DaemonClient("any-grant", config=config, paths=paths)

    with pytest.raises(DaemonUnreachableError, match=f"process {os.getpid()}") as refused:
        await client.recall("anything", "project:closed-aaaaaaaa")
    assert f"if pid {os.getpid()} is not trw_memory.server, the user can remove {paths.discovery}" in str(refused.value)
    assert "report this to the user" in str(refused.value)


async def test_a_client_pinned_to_one_daemon_does_not_wait_for_a_successor(paths: DaemonPaths) -> None:
    """It could not use one (it refuses any other daemon), so waiting would only delay its failure."""
    _record_an_unreachable_daemon(paths)
    patient = MemoryConfig(memory_daemon_startup_timeout_seconds=30.0)
    client = DaemonClient("any-grant", config=patient, paths=paths, instance=(os.getpid(), "2026-09-03T00:00:00+00:00"))

    started = time.monotonic()
    with pytest.raises(DaemonUnreachableError):
        await client.recall("anything", "project:closed-aaaaaaaa")

    assert time.monotonic() - started < 10.0


async def test_a_call_refused_while_a_daemon_drains_reaches_what_serves_next(
    paths: DaemonPaths, running_daemon: DaemonInfo
) -> None:
    """PRD-CORE-310 FR02: a draining daemon closes its socket before it withdraws its record.

    Pre-change the retry dialled the same closed port at once, and the call failed.
    """
    namespace = "project:drain-eeeeeeee"
    patient = MemoryConfig(memory_daemon_startup_timeout_seconds=30.0)
    client = DaemonClient(mint_grant(paths, [namespace]), config=patient, paths=paths)
    successor = paths.discovery.read_text(encoding="utf-8")
    _record_an_unreachable_daemon(paths)  # the predecessor: a live pid, its socket closed

    def _withdraw() -> None:
        time.sleep(0.5)
        paths.discovery.write_text(successor, encoding="utf-8")

    withdrawal = threading.Thread(target=_withdraw)
    withdrawal.start()
    try:
        counted = await client.status(namespace)
    finally:
        withdrawal.join()

    assert isinstance(counted, dict)


async def test_a_killed_daemon_is_replaced_by_the_next_call(
    paths: DaemonPaths, provisioned_embedding_cache: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD-CORE-310 DoD: kill -9 the daemon mid-session and the next call recovers on a fresh one."""
    monkeypatch.setenv("HF_HUB_CACHE", provisioned_embedding_cache)
    namespace = "project:killed-ffffffff"
    patient = MemoryConfig(memory_daemon_startup_timeout_seconds=60.0)
    client = DaemonClient(mint_grant(paths, [namespace]), config=patient, paths=paths)
    try:
        await client.status(namespace)  # auto-starts the first daemon
        first = read_discovery(paths)
        assert first is not None
        os.kill(first.pid, signal.SIGKILL)

        await client.status(namespace)

        second = read_discovery(paths)
        assert second is not None
        assert second.pid != first.pid
    finally:
        reap_daemons_under(paths.user_memory_dir, wait=True)


async def test_with_auto_start_off_a_call_fails_closed_and_spawns_nothing(
    paths: DaemonPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD-CORE-310 FR04: TRW's post-commit hook sets it, so a commit never leaves a daemon behind."""
    spawns: list[DaemonPaths] = []
    monkeypatch.setattr(client_module, "start_daemon_detached", spawns.append)
    monkeypatch.setenv("MEMORY_DAEMON_AUTOSTART", "false")

    with pytest.raises(DaemonUnreachableError, match="MEMORY_DAEMON_AUTOSTART=false"):
        await DaemonClient("any-grant", paths=paths).recall("anything", "project:closed-aaaaaaaa")

    assert spawns == []


def test_reading_the_record_never_starts_a_daemon(paths: DaemonPaths) -> None:
    """A reachability probe must not be the thing that starts one."""
    assert isinstance(read_live_discovery(paths), DiscoveryAbsent)
    assert not paths.discovery.exists()
    assert not paths.token.exists()


def test_reading_the_record_reports_a_running_daemon(
    shared_paths: DaemonPaths, shared_running_daemon: DaemonInfo
) -> None:
    """And it does report one that is genuinely there."""
    probed = read_live_discovery(shared_paths)

    assert isinstance(probed, DaemonInfo)
    assert probed.pid == shared_running_daemon.pid
    assert probed.url == shared_running_daemon.url


class _LoseFirstResponse:
    """A ``Client`` whose first call reaches the daemon and commits, then loses the response."""

    calls: list[str] = []

    def __init__(self, transport: object) -> None:
        from fastmcp.client import Client  # the real one: the tests below patch ``fastmcp.Client``

        self._inner = Client(transport)  # type: ignore[arg-type]

    async def __aenter__(self) -> _LoseFirstResponse:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._inner.__aexit__(*exc)  # type: ignore[arg-type]

    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        import httpx

        result = await self._inner.call_tool(name, arguments)
        _LoseFirstResponse.calls.append(name)
        if len(_LoseFirstResponse.calls) == 1:
            raise httpx.ReadError("the connection dropped after the daemon answered")
        return result


async def test_a_store_whose_response_is_lost_after_commit_writes_one_row(
    shared_paths: DaemonPaths, config: MemoryConfig, shared_running_daemon: DaemonInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = "project:lost-eeeeeeee"
    client = DaemonClient(mint_grant(shared_paths, [namespace]), config=config, paths=shared_paths)
    _LoseFirstResponse.calls = []
    monkeypatch.setattr("fastmcp.Client", _LoseFirstResponse)

    await client.store("committed once, answered twice", namespace)
    monkeypatch.undo()
    listed = await client.search(namespace, limit=10)

    assert _LoseFirstResponse.calls == ["memory_store", "memory_store"]
    assert [row["content"] for row in listed["entries"]] == ["committed once, answered twice"]


async def test_a_write_that_cannot_be_replayed_is_not_retried_once_sent(
    shared_paths: DaemonPaths, config: MemoryConfig, shared_running_daemon: DaemonInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = "project:lost-ffffffff"
    client = DaemonClient(mint_grant(shared_paths, [namespace]), config=config, paths=shared_paths)
    stored = await client.store("forget me once", namespace)
    _LoseFirstResponse.calls = []
    monkeypatch.setattr("fastmcp.Client", _LoseFirstResponse)

    with pytest.raises(DaemonUnreachableError, match="may have been applied"):
        await client.forget(stored["memory_id"], namespace)

    assert _LoseFirstResponse.calls == ["memory_forget"]


async def test_a_held_session_serves_many_calls_on_one_initialize(
    shared_paths: DaemonPaths, config: MemoryConfig, shared_running_daemon: DaemonInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W27: one session for many calls, and the stateless daemon never holds an SSE GET open."""
    import collections

    import httpx

    sent: collections.Counter[str] = collections.Counter()
    real_send = httpx.AsyncClient.send

    async def counting(self: httpx.AsyncClient, request: httpx.Request, **kwargs: object) -> httpx.Response:
        body = request.content.decode(errors="replace") if request.method == "POST" else ""
        marker = '"method":"'
        method = body.split(marker)[1].split('"')[0] if marker in body else ""
        sent[f"{request.method} {method}"] += 1
        return await real_send(self, request, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx.AsyncClient, "send", counting)
    namespace = "project:held-11111111"
    client = DaemonClient(mint_grant(shared_paths, [namespace]), config=config, paths=shared_paths, keep_session=True)

    stored = await client.store("held session row", namespace)
    for _ in range(3):
        assert (await client.get(stored["memory_id"], namespace))["entry"]["id"] == stored["memory_id"]
    await client._sessions.drop()

    assert sent["POST initialize"] == 1
    assert sent["POST tools/call"] == 4
    assert sent["POST tools/list"] <= 1
    assert not [key for key in sent if key.startswith(("GET", "DELETE"))], sent


async def test_a_held_session_is_replaced_after_a_lost_response(
    shared_paths: DaemonPaths, config: MemoryConfig, shared_running_daemon: DaemonInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W27: a transport failure drops the held session; the replayable retry opens a fresh one."""
    namespace = "project:held-22222222"
    client = DaemonClient(mint_grant(shared_paths, [namespace]), config=config, paths=shared_paths, keep_session=True)
    _LoseFirstResponse.calls = []
    monkeypatch.setattr("fastmcp.Client", _LoseFirstResponse)

    await client.store("committed once on a held session", namespace)
    held = client._sessions.held
    assert _LoseFirstResponse.calls == ["memory_store", "memory_store"]
    listed = await client.search(namespace, limit=10)

    assert held is not None and isinstance(held.client, _LoseFirstResponse), "the retry did not open a new session"
    assert [row["content"] for row in listed["entries"]] == ["committed once on a held session"]


@pytest.mark.parametrize("header", [None, "0.9.0"])
async def test_a_client_of_another_major_is_refused_by_name(
    shared_paths: DaemonPaths, config: MemoryConfig, shared_running_daemon: DaemonInfo, header: str | None
) -> None:
    """W45: a 3.1.0-shaped call (no version header, the old signature) gets the upgrade, not a contract error."""
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport
    from fastmcp.exceptions import ToolError

    from trw_memory.daemon._version_gate import VERSION_HEADER

    namespace = "project:oldclient-33333333"
    token = mint_grant(shared_paths, [namespace])
    transport = StreamableHttpTransport(
        url=shared_running_daemon.url, auth=token, headers={VERSION_HEADER: header} if header else None
    )
    async with Client(transport) as old_client:
        with pytest.raises(ToolError, match="daemon_version_mismatch") as refused:
            await old_client.call_tool("memory_recall", {"query": "anything", "namespace": namespace, "limit": 5})

    message = str(refused.value)
    assert f"pid {shared_running_daemon.pid}" in message
    assert "pip install -U trw-mcp trw-memory" in message
    assert ("3.x or older" in message) if header is None else (f"is trw-memory {header}" in message)
    # A current client on the same daemon is unaffected.
    current = DaemonClient(token, config=config, paths=shared_paths)
    assert (await current.store("the current client still writes", namespace))["status"] == "stored"


@pytest.mark.parametrize(("theirs", "refused"), [("3.1.0", True), ("4.2.1", False), ("test", False)])
def test_attach_refuses_a_daemon_of_another_major(
    paths: DaemonPaths, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch, theirs: str, refused: bool
) -> None:
    """PRD-CORE-302 C7: the package floor cannot cover a daemon already running from another install."""
    from trw_memory.daemon import client as client_module
    from trw_memory.exceptions import DaemonVersionMismatchError

    monkeypatch.setattr(client_module, "_package_version", lambda: "4.0.0")
    paths.user_memory_dir.mkdir(parents=True, exist_ok=True)
    info = DaemonInfo(
        pid=os.getpid(), url="http://127.0.0.1:1/mcp", started_at="2026-09-24T00:00:00+00:00", version=theirs
    )
    paths.discovery.write_text(info.model_dump_json(), encoding="utf-8")
    client = DaemonClient("any-grant", config=config, paths=paths)

    if refused:
        with pytest.raises(DaemonVersionMismatchError, match=r"daemon_version_mismatch.*serves 3\.1\.0.*is 4\.0\.0"):
            client._attach()
    else:
        assert client._attach().version == theirs


class _GatedSession:
    """A ``Client`` that records its sessions and holds ``call_tool`` until the gate opens."""

    opened: list[_GatedSession] = []
    gate: asyncio.Event
    entered: asyncio.Event
    open_gate: asyncio.Event

    def __init__(self, _transport: object) -> None:
        self.closed = 0
        _GatedSession.opened.append(self)

    async def __aenter__(self) -> _GatedSession:
        await _GatedSession.open_gate.wait()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed += 1

    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        if name == "memory_fail_in_transport":
            import httpx

            raise httpx.ReadError("the connection dropped")
        _GatedSession.entered.set()
        await _GatedSession.gate.wait()
        return types.SimpleNamespace(data={"status": "ok", "tool": name})


@pytest.fixture
def gated_sessions(monkeypatch: pytest.MonkeyPatch) -> type[_GatedSession]:
    _GatedSession.opened = []
    _GatedSession.gate = asyncio.Event()
    _GatedSession.entered = asyncio.Event()
    _GatedSession.open_gate = asyncio.Event()
    _GatedSession.open_gate.set()
    monkeypatch.setattr("fastmcp.Client", _GatedSession)
    return _GatedSession


async def test_retiring_a_client_mid_call_closes_its_session_only_after_the_call_returns(
    shared_paths: DaemonPaths,
    config: MemoryConfig,
    shared_running_daemon: DaemonInfo,
    gated_sessions: type[_GatedSession],
) -> None:
    """Release-verify RES-01: a replaced client's held session is closed, but never under an in-flight call."""
    client = DaemonClient("grant", config=config, paths=shared_paths, keep_session=True)
    in_flight = asyncio.create_task(client.call_tool("memory_get", {"memory_id": "M-x", "namespace": "project:x"}))
    await gated_sessions.entered.wait()

    await client.retire()
    held = gated_sessions.opened[0]
    assert held.closed == 0, "the in-flight call's session was closed under it"

    gated_sessions.gate.set()
    assert (await in_flight)["status"] == "ok"
    assert held.closed == 1

    await client.call_tool("memory_get", {"memory_id": "M-x", "namespace": "project:x"})
    assert len(gated_sessions.opened) == 2, "a retired client opens a session per call"
    assert gated_sessions.opened[1].closed == 1, "and closes it when the call returns"
    assert held.closed == 1


async def test_retiring_an_idle_client_closes_its_session_at_once(
    shared_paths: DaemonPaths,
    config: MemoryConfig,
    shared_running_daemon: DaemonInfo,
    gated_sessions: type[_GatedSession],
) -> None:
    gated_sessions.gate.set()
    client = DaemonClient("grant", config=config, paths=shared_paths, keep_session=True)
    await client.call_tool("memory_get", {"memory_id": "M-x", "namespace": "project:x"})
    assert gated_sessions.opened[0].closed == 0, "the session is held between calls"

    await client.retire()

    assert gated_sessions.opened[0].closed == 1


async def test_a_transport_failure_never_closes_the_session_under_another_in_flight_call(
    shared_paths: DaemonPaths,
    config: MemoryConfig,
    shared_running_daemon: DaemonInfo,
    gated_sessions: type[_GatedSession],
) -> None:
    """Sol round 2: the failing call releases the shared session; the call still using it closes it on return."""
    client = DaemonClient("grant", config=config, paths=shared_paths, keep_session=True)
    in_flight = asyncio.create_task(client.call_tool("memory_get", {"memory_id": "M-x", "namespace": "project:x"}))
    await gated_sessions.entered.wait()
    shared = gated_sessions.opened[0]

    with pytest.raises(DaemonUnreachableError):
        await client.call_tool("memory_fail_in_transport", {})
    assert shared.closed == 0, "the failing call closed a session another call was using"

    gated_sessions.gate.set()
    assert (await in_flight)["status"] == "ok"
    assert shared.closed == 1
    await client.call_tool("memory_get", {"memory_id": "M-x", "namespace": "project:x"})
    assert gated_sessions.opened[-1] is not shared, "the released session was handed out again"


async def test_a_session_still_opening_when_the_client_is_retired_is_closed_after_its_call(
    shared_paths: DaemonPaths,
    config: MemoryConfig,
    shared_running_daemon: DaemonInfo,
    gated_sessions: type[_GatedSession],
) -> None:
    """Sol round 3: retire() landing while a session opens must not leave that session held forever."""
    gated_sessions.open_gate.clear()
    gated_sessions.gate.set()
    client = DaemonClient("grant", config=config, paths=shared_paths, keep_session=True)
    opening_call = asyncio.create_task(client.call_tool("memory_get", {"memory_id": "M-x", "namespace": "project:x"}))
    while not gated_sessions.opened:  # noqa: ASYNC110 -- waits for the Client constructor, which has no event
        await asyncio.sleep(0)

    await client.retire()
    gated_sessions.open_gate.set()
    assert (await opening_call)["status"] == "ok"

    assert gated_sessions.opened[0].closed == 1
    assert client._sessions.held is None


def test_a_held_session_closes_when_its_loop_ends(
    shared_paths: DaemonPaths,
    config: MemoryConfig,
    shared_running_daemon: DaemonInfo,
    gated_sessions: type[_GatedSession],
) -> None:
    """PRD-CORE-331 FR09 (B71-09): a keep_session client with no explicit close() must not leak.

    A synchronous test on purpose: ``asyncio.run`` gives the held session a loop that
    genuinely ends (unlike a pytest-asyncio test body, whose loop keeps running after
    the test function returns). Pre-fix (79e84147d), ``gated_sessions.opened[0].closed``
    stayed ``0`` here -- the held session's transport was never closed once its loop
    was gone, because ``_close`` no-ops off the session's own loop and nothing else
    ever ran it back on that loop.
    """
    gated_sessions.gate.set()
    client = DaemonClient("grant", config=config, paths=shared_paths, keep_session=True)

    asyncio.run(client.call_tool("memory_get", {"memory_id": "M-x", "namespace": "project:x"}))

    assert gated_sessions.opened[0].closed == 1, "the held session outlived the loop that opened it"


# PRD-CORE-333 S3b: the reads a client sends as one stateless POST (``_direct.DIRECT_TOOLS``).


def _count_requests(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every HTTP request any client sends from here on, as ``"<verb> <json-rpc method>"``."""
    import json

    sent: list[str] = []
    real_send = httpx.AsyncClient.send

    async def counting(self: httpx.AsyncClient, request: httpx.Request, **kwargs: object) -> httpx.Response:
        method = json.loads(request.content).get("method", "") if request.method == "POST" else ""
        sent.append(f"{request.method} {method}")
        return await real_send(self, request, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx.AsyncClient, "send", counting)
    return sent


async def _session_answer(info: DaemonInfo, token: str, name: str, arguments: dict[str, object]) -> object:
    """What a full MCP session (initialize, ``tools/list``, ``tools/call``) returns for the same call."""
    from fastmcp.client import Client

    async with Client(_session.transport(info, token, client_module._package_version())) as session:
        return (await session.call_tool(name, arguments)).data


@pytest.mark.parametrize("keep_session", [False, True])
async def test_a_direct_read_is_one_post_and_opens_no_session(
    shared_paths: DaemonPaths,
    config: MemoryConfig,
    shared_running_daemon: DaemonInfo,
    monkeypatch: pytest.MonkeyPatch,
    keep_session: bool,
) -> None:
    """Held or not, ``memory_status`` and ``memory_list_page`` are each one ``tools/call``: no initialize, no list."""
    namespace = f"project:direct-{'held' if keep_session else 'call'}aaaa"
    token = mint_grant(shared_paths, [namespace])
    await DaemonClient(token, config=config, paths=shared_paths).store("a row the direct page lists", namespace)
    client = DaemonClient(token, config=config, paths=shared_paths, keep_session=keep_session)
    sent = _count_requests(monkeypatch)

    status = await client.call_tool("memory_status", {"namespace": namespace, "security_settings_only": True})
    page = await client.list_page(namespace, 10, None, status="active")

    assert sent == ["POST tools/call", "POST tools/call"]
    assert client._sessions.held is None, "a direct read opened a held session"
    assert status["daemon"] == [shared_running_daemon.pid, shared_running_daemon.started_at]
    assert [row["content"] for row in page["entries"]] == ["a row the direct page lists"]


async def test_direct_reads_answer_what_a_session_answers(
    shared_paths: DaemonPaths, config: MemoryConfig, shared_running_daemon: DaemonInfo
) -> None:
    """Randomised (seeded) property: for any page arguments, one POST's answer == a full session's ``.data``.

    Walks cursors, tag filters, status filters (an unknown status included) and the
    limit bound's refusal, so an unwrapping or typing difference in either path shows.
    """
    import random

    rnd = random.Random(20260926)
    namespace = "project:direct-equivbbbb"
    token = mint_grant(shared_paths, [namespace])
    client = DaemonClient(token, config=config, paths=shared_paths)
    vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon"]
    for index in range(40):
        stored = await client.store(f"row {index} {rnd.choice(vocabulary)}", namespace, tags=rnd.sample(vocabulary, 2))
        if rnd.random() < 0.3:
            await client.update(stored["memory_id"], namespace, {"status": "obsolete"})

    cases = 0
    for _ in range(25):
        arguments: dict[str, object] = {
            "namespace": namespace,
            "limit": rnd.choice([0, 1, 3, 7, 50, 1001]),
            "after": None,
            "status": rnd.choice([None, "active", "obsolete", "no-such-status"]),
            "tags": rnd.choice([None, [], rnd.sample(vocabulary, 1), rnd.sample(vocabulary, 2)]),
        }
        for _page in range(4):
            direct = await client.call_tool("memory_list_page", arguments)
            assert direct == await _session_answer(shared_running_daemon, token, "memory_list_page", arguments)
            cases += 1
            if not isinstance(direct, dict) or not direct.get("next"):
                break
            arguments = {**arguments, "after": direct["next"]}
    for only in (True, False):
        arguments = {"namespace": namespace, "security_settings_only": only}
        direct = await client.call_tool("memory_status", arguments)
        answered = await _session_answer(shared_running_daemon, token, "memory_status", arguments)
        assert direct == answered
    assert cases >= 25


async def test_the_edit_hooks_recall_and_anchored_reads_are_one_post_answering_what_a_session_answers(
    shared_paths: DaemonPaths, config: MemoryConfig, shared_running_daemon: DaemonInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``memory_recall`` and ``memory_anchored`` (the edit hook's T1 reads) go direct: one POST, no session.

    Opening a session for them imported all of fastmcp (~0.7 s of the hook's wall time). The
    answer must equal a full session's ``.data`` -- hits, misses, filters and a refusal alike.
    """
    from fastmcp.exceptions import ToolError

    namespace = "project:direct-recalljjjj"
    token = mint_grant(shared_paths, [namespace])
    client = DaemonClient(token, config=config, paths=shared_paths)
    anchor = {"file": "src/db.py", "symbol_name": "connect", "symbol_type": "function"}
    await client.store("pin the sqlite driver before connect", namespace, learning={"anchors": [anchor]})
    await client.store("the connect retry backs off twice", namespace, tags=["sqlite"])
    await client.store("an unrelated row about pagination", namespace)

    calls: list[tuple[str, dict[str, object]]] = [
        ("memory_recall", {"query": "sqlite connect", "namespace": namespace, "limit": 5, "record_access": False}),
        ("memory_recall", {"query": "sqlite", "namespace": namespace, "tags": ["sqlite"], "record_access": False}),
        ("memory_recall", {"query": "no-such-word-anywhere", "namespace": namespace, "record_access": False}),
        ("memory_anchored", {"namespace": namespace, "file": "src/db.py", "limit": 10, "status": "active"}),
        ("memory_anchored", {"namespace": namespace, "file": "src/none.py", "limit": 10}),
    ]
    for name, arguments in calls:
        sent = _count_requests(monkeypatch)
        direct = await client.call_tool(name, arguments)
        assert sent == ["POST tools/call"], name
        monkeypatch.undo()
        assert direct == await _session_answer(shared_running_daemon, token, name, arguments), (name, arguments)
    anchored = await client.call_tool("memory_anchored", calls[3][1])
    assert [row["content"] for row in anchored["memories"]] == ["pin the sqlite driver before connect"]

    with pytest.raises(ToolError, match="not granted namespace"):
        await client.call_tool("memory_recall", {"query": "x", "namespace": "project:direct-otherjjjj"})


async def test_a_direct_read_the_daemon_refuses_is_a_tool_error_and_not_retried(
    shared_paths: DaemonPaths, config: MemoryConfig, shared_running_daemon: DaemonInfo
) -> None:
    """A namespace outside the grant: the daemon answered, so the refusal surfaces as a session call's would."""
    from fastmcp.exceptions import ToolError

    client = DaemonClient(mint_grant(shared_paths, ["project:direct-mine-cccc"]), config=config, paths=shared_paths)

    with structlog.testing.capture_logs() as logs:
        with pytest.raises(ToolError, match="not granted namespace 'project:direct-other-cccc'"):
            await client.list_page("project:direct-other-cccc", 5, None)

    assert [entry for entry in logs if entry.get("event") == "daemon_call_failed"] == []


def _answer_with(monkeypatch: pytest.MonkeyPatch, respond: object) -> list[httpx.Request]:
    """Route every ``httpx.AsyncClient`` through *respond* (a MockTransport handler); returns the requests."""
    seen: list[httpx.Request] = []
    real = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return respond(request)  # type: ignore[operator,no-any-return]

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs))
    return seen


async def test_a_direct_read_presents_the_grant_and_this_clients_version(
    paths: DaemonPaths, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The POST carries the bearer grant and the W45 version header; the answer is its structuredContent."""
    from trw_memory.daemon._discovery import VERSION_HEADER

    _record_an_unreachable_daemon(paths)
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [], "structuredContent": {"status": "ok"}, "isError": False},
    }
    seen = _answer_with(monkeypatch, lambda _request: httpx.Response(200, json=body))

    answer = await DaemonClient("the-grant", config=config, paths=paths).status("project:direct-dddd")

    assert answer == {"status": "ok"}
    assert len(seen) == 1
    assert seen[0].headers["authorization"] == "Bearer the-grant"
    assert seen[0].headers[VERSION_HEADER] == client_module._package_version()


async def test_a_direct_read_to_a_non_loopback_url_carries_no_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bearer comes from build_platform_headers: never sent to a host that is neither loopback nor trusted.

    A daemon record is loopback-only (DaemonInfo refuses any other url), so this builds one past that check.
    """
    body = {"jsonrpc": "2.0", "id": 1, "result": {"content": [], "structuredContent": {"status": "ok"}}}
    seen = _answer_with(monkeypatch, lambda _request: httpx.Response(200, json=body))
    info = DaemonInfo.model_construct(
        pid=os.getpid(), url="http://daemon.example.net:1/mcp", started_at="x", version="t"
    )

    assert await _direct.post_tool(info, "the-grant", "4.0.0", "memory_status", {}) == {"status": "ok"}
    assert "authorization" not in seen[0].headers


async def test_a_direct_read_answered_with_a_protocol_error_is_a_tool_error(
    paths: DaemonPaths, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JSON-RPC ``error`` (no ``result``) is the daemon answering: a ``ToolError`` naming it, not retried."""
    from fastmcp.exceptions import ToolError

    _record_an_unreachable_daemon(paths)
    body = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "Invalid request parameters"}}
    seen = _answer_with(monkeypatch, lambda _request: httpx.Response(200, json=body))

    with pytest.raises(ToolError, match="Invalid request parameters"):
        await DaemonClient("the-grant", config=config, paths=paths).status("project:direct-eeee")
    assert len(seen) == 1


def _envelope(result: object) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": 1, "result": result}


#: Replies that are neither an answer nor a refusal: each must raise, never return (review r1 P2).
_MALFORMED_REPLIES = {
    "structuredContent is a string": _envelope({"content": [], "structuredContent": "invalid", "isError": False}),
    "structuredContent is missing": _envelope({"content": [{"type": "text", "text": "{}"}], "isError": False}),
    "structuredContent is a list": _envelope({"content": [], "structuredContent": [1, 2], "isError": False}),
    "result is a list": _envelope([{"structuredContent": {}}]),
    "isError is not a bool": _envelope({"content": [], "structuredContent": {}, "isError": "yes"}),
    "neither result nor error": {"jsonrpc": "2.0", "id": 1},
    "both result and error": {**_envelope({"structuredContent": {}}), "error": {"code": 1, "message": "x"}},
    "another request's id": {**_envelope({"structuredContent": {"status": "ok"}}), "id": 2},
    "no id": {"jsonrpc": "2.0", "result": {"structuredContent": {"status": "ok"}}},
    "not JSON-RPC 2.0": {**_envelope({"structuredContent": {"status": "ok"}}), "jsonrpc": "1.0"},
    "an error without a message": {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000}},
    "the envelope is a list": [_envelope({"structuredContent": {"status": "ok"}})],
    "the body is not JSON": "<html>502 Bad Gateway</html>",
}


@pytest.mark.parametrize("shape", sorted(_MALFORMED_REPLIES))
async def test_a_malformed_direct_reply_fails_closed_without_a_retry(
    paths: DaemonPaths, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """Checked at least as strictly as a session: a malformed reply is a DaemonProtocolError, an unreachable store."""
    _record_an_unreachable_daemon(paths)
    reply = _MALFORMED_REPLIES[shape]
    seen = _answer_with(
        monkeypatch,
        lambda _request: httpx.Response(200, text=reply) if isinstance(reply, str) else httpx.Response(200, json=reply),
    )

    with pytest.raises(DaemonProtocolError, match="nothing was read") as refused:
        await DaemonClient("the-grant", config=config, paths=paths).list_page("project:direct-hhhh", 5, None)
    assert isinstance(refused.value, DaemonUnreachableError), "every fail-closed handler must catch it"
    assert AGENT_MUST_NOT_STOP in str(refused.value), "HB-2: the restart advice is addressed to the user"
    assert len(seen) == 1, "a malformed answer was retried"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ({"text": "denied"}, "denied"),
        ([{"type": "text", "text": "denied"}], "denied"),
        ([{"type": "text", "text": "first"}, {"type": "image"}, {"type": "text", "text": "second"}], "first\nsecond"),
        ("denied", "denied"),
        (None, "memory_list_page refused"),
        ([], "memory_list_page refused"),
        ([{"type": "image"}, 7], "memory_list_page refused"),
        ({"text": 7}, "memory_list_page refused"),
    ],
)
async def test_a_refusal_is_a_tool_error_whatever_shape_its_content_has(
    paths: DaemonPaths, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch, content: object, message: str
) -> None:
    """``isError: true`` is the daemon answering: a ToolError with its text, never a KeyError or a retry."""
    from fastmcp.exceptions import ToolError

    _record_an_unreachable_daemon(paths)
    result: dict[str, object] = {"isError": True}
    if content is not None:
        result["content"] = content
    seen = _answer_with(monkeypatch, lambda _request: httpx.Response(200, json=_envelope(result)))

    with pytest.raises(ToolError) as refused:
        await DaemonClient("the-grant", config=config, paths=paths).list_page("project:direct-iiii", 5, None)
    assert str(refused.value) == message
    assert len(seen) == 1


async def test_a_direct_read_with_a_rejected_grant_fails_closed_without_retrying(
    paths: DaemonPaths, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 401 on the POST is the same ``DaemonAuthError`` a session's rejected initialize raises."""
    _record_an_unreachable_daemon(paths)
    seen = _answer_with(monkeypatch, lambda _request: httpx.Response(401, json={"error": "invalid_token"}))

    with pytest.raises(DaemonAuthError, match="nothing was re-minted"):
        await DaemonClient("a-stale-grant", config=config, paths=paths).list_page("project:direct-ffff", 5, None)
    assert len(seen) == 1


async def test_a_direct_read_against_a_dead_endpoint_is_tried_twice_then_fails_closed(
    paths: DaemonPaths, config: MemoryConfig
) -> None:
    """FR08 clause 1 holds on the direct path: a connect failure is retried once, then names the remedy."""
    _record_an_unreachable_daemon(paths)
    client = DaemonClient("any-grant", config=config, paths=paths)

    with structlog.testing.capture_logs() as logs:
        with pytest.raises(DaemonUnreachableError, match=r"daemon\.json"):
            await client.status("project:direct-gggg")

    assert [entry["attempt"] for entry in logs if entry.get("event") == "daemon_call_failed"] == [1, 2]


@pytest.mark.parametrize(("error", "answered"), [("tool", True), ("transport", False)])
def test_only_a_tool_error_counts_as_the_daemon_answering(error: str, answered: bool) -> None:
    from fastmcp.exceptions import ToolError

    exc = ToolError("refused") if error == "tool" else httpx.ReadError("dropped")
    assert _direct.answered(exc) is answered


def test_importing_the_client_does_not_import_fastmcp() -> None:
    """A process that only makes direct reads never pays for fastmcp (the hook's ~0.3 s)."""
    probe = "import sys, trw_memory.daemon.client; sys.exit('fastmcp' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0
