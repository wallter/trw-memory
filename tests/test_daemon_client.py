"""PRD-CORE-253 FR08 — the client fails closed, and says how to fix it.

Unreachability is produced for real here, not simulated: a discovery record
naming a LIVE process (this one) and a port nothing is listening on is exactly
the state a killed daemon or a reused pid leaves behind, and it is the state in
which a fail-open client would quietly return an empty recall.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from trw_memory.daemon import (
    DaemonInfo,
    DaemonPaths,
    DiscoveryAbsent,
    mint_grant,
    read_live_discovery,
)
from trw_memory.daemon.client import DAEMON_START_COMMAND, DaemonClient
from trw_memory.exceptions import DaemonAuthError, DaemonUnreachableError
from trw_memory.models.config import MemoryConfig

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


@pytest.fixture
def running_daemon(paths: DaemonPaths, provisioned_embedding_cache: str) -> Iterator[DaemonInfo]:
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
        env={
            **os.environ,
            "TRW_USER_DIR": str(paths.user_memory_dir.parent),
            "HF_HUB_CACHE": provisioned_embedding_cache,
        },
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
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo
) -> None:
    """The positive path: the same client that fails closed also works."""
    namespace = "project:roundtrip-dddddddd"
    client = DaemonClient(mint_grant(paths, [namespace]), config=config, paths=paths)

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


class _LoseFirstResponse:
    """A ``Client`` whose first call reaches the daemon and commits, then loses the response."""

    calls: list[str] = []

    def __init__(self, transport: object) -> None:
        from fastmcp import Client

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
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = "project:lost-eeeeeeee"
    client = DaemonClient(mint_grant(paths, [namespace]), config=config, paths=paths)
    _LoseFirstResponse.calls = []
    monkeypatch.setattr("trw_memory.daemon.client.Client", _LoseFirstResponse)

    await client.store("committed once, answered twice", namespace)
    monkeypatch.undo()
    listed = await client.search(namespace, limit=10)

    assert _LoseFirstResponse.calls == ["memory_store", "memory_store"]
    assert [row["content"] for row in listed["entries"]] == ["committed once, answered twice"]


async def test_a_write_that_cannot_be_replayed_is_not_retried_once_sent(
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = "project:lost-ffffffff"
    client = DaemonClient(mint_grant(paths, [namespace]), config=config, paths=paths)
    stored = await client.store("forget me once", namespace)
    _LoseFirstResponse.calls = []
    monkeypatch.setattr("trw_memory.daemon.client.Client", _LoseFirstResponse)

    with pytest.raises(DaemonUnreachableError, match="may have been applied"):
        await client.forget(stored["memory_id"], namespace)

    assert _LoseFirstResponse.calls == ["memory_forget"]


async def test_a_held_session_serves_many_calls_on_one_initialize(
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo, monkeypatch: pytest.MonkeyPatch
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
    client = DaemonClient(mint_grant(paths, [namespace]), config=config, paths=paths, keep_session=True)

    stored = await client.store("held session row", namespace)
    for _ in range(3):
        assert (await client.get(stored["memory_id"], namespace))["entry"]["id"] == stored["memory_id"]
    await client._sessions.drop()

    assert sent["POST initialize"] == 1
    assert sent["POST tools/call"] == 4
    assert sent["POST tools/list"] <= 1
    assert not [key for key in sent if key.startswith(("GET", "DELETE"))], sent


async def test_a_held_session_is_replaced_after_a_lost_response(
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W27: a transport failure drops the held session; the replayable retry opens a fresh one."""
    namespace = "project:held-22222222"
    client = DaemonClient(mint_grant(paths, [namespace]), config=config, paths=paths, keep_session=True)
    _LoseFirstResponse.calls = []
    monkeypatch.setattr("trw_memory.daemon.client.Client", _LoseFirstResponse)

    await client.store("committed once on a held session", namespace)
    held = client._sessions.held
    assert _LoseFirstResponse.calls == ["memory_store", "memory_store"]
    listed = await client.search(namespace, limit=10)

    assert held is not None and isinstance(held.client, _LoseFirstResponse), "the retry did not open a new session"
    assert [row["content"] for row in listed["entries"]] == ["committed once on a held session"]


@pytest.mark.parametrize("header", [None, "0.9.0"])
async def test_a_client_of_another_major_is_refused_by_name(
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo, header: str | None
) -> None:
    """W45: a 3.1.0-shaped call (no version header, the old signature) gets the upgrade, not a contract error."""
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport
    from fastmcp.exceptions import ToolError

    from trw_memory.daemon._version_gate import VERSION_HEADER

    namespace = "project:oldclient-33333333"
    token = mint_grant(paths, [namespace])
    transport = StreamableHttpTransport(
        url=running_daemon.url, auth=token, headers={VERSION_HEADER: header} if header else None
    )
    async with Client(transport) as old_client:
        with pytest.raises(ToolError, match="daemon_version_mismatch") as refused:
            await old_client.call_tool("memory_recall", {"query": "anything", "namespace": namespace, "limit": 5})

    message = str(refused.value)
    assert f"pid {running_daemon.pid}" in message
    assert "pip install -U trw-mcp trw-memory" in message
    assert ("3.x or older" in message) if header is None else (f"is trw-memory {header}" in message)
    # A current client on the same daemon is unaffected.
    current = DaemonClient(token, config=config, paths=paths)
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
    monkeypatch.setattr("trw_memory.daemon.client.Client", _GatedSession)
    return _GatedSession


async def test_retiring_a_client_mid_call_closes_its_session_only_after_the_call_returns(
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo, gated_sessions: type[_GatedSession]
) -> None:
    """Release-verify RES-01: a replaced client's held session is closed, but never under an in-flight call."""
    client = DaemonClient("grant", config=config, paths=paths, keep_session=True)
    in_flight = asyncio.create_task(client.call_tool("memory_status", {"namespace": "project:x"}))
    await gated_sessions.entered.wait()

    await client.retire()
    held = gated_sessions.opened[0]
    assert held.closed == 0, "the in-flight call's session was closed under it"

    gated_sessions.gate.set()
    assert (await in_flight)["status"] == "ok"
    assert held.closed == 1

    await client.call_tool("memory_status", {"namespace": "project:x"})
    assert len(gated_sessions.opened) == 2, "a retired client opens a session per call"
    assert gated_sessions.opened[1].closed == 1, "and closes it when the call returns"
    assert held.closed == 1


async def test_retiring_an_idle_client_closes_its_session_at_once(
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo, gated_sessions: type[_GatedSession]
) -> None:
    gated_sessions.gate.set()
    client = DaemonClient("grant", config=config, paths=paths, keep_session=True)
    await client.call_tool("memory_status", {"namespace": "project:x"})
    assert gated_sessions.opened[0].closed == 0, "the session is held between calls"

    await client.retire()

    assert gated_sessions.opened[0].closed == 1


async def test_a_transport_failure_never_closes_the_session_under_another_in_flight_call(
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo, gated_sessions: type[_GatedSession]
) -> None:
    """Sol round 2: the failing call releases the shared session; the call still using it closes it on return."""
    client = DaemonClient("grant", config=config, paths=paths, keep_session=True)
    in_flight = asyncio.create_task(client.call_tool("memory_status", {"namespace": "project:x"}))
    await gated_sessions.entered.wait()
    shared = gated_sessions.opened[0]

    with pytest.raises(DaemonUnreachableError):
        await client.call_tool("memory_fail_in_transport", {})
    assert shared.closed == 0, "the failing call closed a session another call was using"

    gated_sessions.gate.set()
    assert (await in_flight)["status"] == "ok"
    assert shared.closed == 1
    await client.call_tool("memory_status", {"namespace": "project:x"})
    assert gated_sessions.opened[-1] is not shared, "the released session was handed out again"


async def test_a_session_still_opening_when_the_client_is_retired_is_closed_after_its_call(
    paths: DaemonPaths, config: MemoryConfig, running_daemon: DaemonInfo, gated_sessions: type[_GatedSession]
) -> None:
    """Sol round 3: retire() landing while a session opens must not leave that session held forever."""
    gated_sessions.open_gate.clear()
    gated_sessions.gate.set()
    client = DaemonClient("grant", config=config, paths=paths, keep_session=True)
    opening_call = asyncio.create_task(client.call_tool("memory_status", {"namespace": "project:x"}))
    while not gated_sessions.opened:  # noqa: ASYNC110 -- waits for the Client constructor, which has no event
        await asyncio.sleep(0)

    await client.retire()
    gated_sessions.open_gate.set()
    assert (await opening_call)["status"] == "ok"

    assert gated_sessions.opened[0].closed == 1
    assert client._sessions.held is None
