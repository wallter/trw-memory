"""The platform contact switch: one resolution, read live at every network boundary (B71-106, B71-107).

rc11 F3 folded ``platform_contact_enabled`` into ``MemoryConfig.sync_enabled`` once, at construction:
a running client or daemon kept its startup answer, and the SSE reconnect loop never asked again.
trw-memory also let a project ``null`` override a machine ``false`` and raised on an invalid value,
where trw-mcp fell back. ``trw_memory.platform_contact`` is now the one resolver both packages call,
and every sender asks it (through ``platform_contact_blocked``) at the moment it would connect.
"""

from __future__ import annotations

import threading
from pathlib import Path

import httpx
import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.platform_contact import platform_contact_enabled
from trw_memory.sync import subscriber as subscriber_module
from trw_memory.sync._remote_fetch import fetch_shared_memories
from trw_memory.sync._remote_publish import drain_retry_queue, publish_memory_result, retire_remote_memory
from trw_memory.sync.retry_queue import RetryQueue
from trw_memory.sync.subscriber import SSESubscriber


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home, root = tmp_path / "home", tmp_path / "project"
    (home / ".trw").mkdir(parents=True)
    (root / ".trw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TRW_PROJECT_ROOT", str(root))
    monkeypatch.delenv("TRW_PLATFORM_CONTACT_ENABLED", raising=False)
    return root


def _write(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def _switch(project: Path, value: str | None) -> None:
    _write(project / ".trw" / "config.yaml", "" if value is None else f"platform_contact_enabled: {value}\n")


# --- resolution ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "project_value", "machine_value", "expected"),
    [
        (None, None, None, True),  # nothing says anything: on
        (None, None, "false", False),  # machine off
        (None, "null", "false", False),  # a project null says nothing; the machine off stands (B71-107)
        (None, "true", "false", True),  # the project beats the machine
        ("false", "true", "true", False),  # the environment beats both
        ("", "false", None, False),  # a blank env var says nothing
        (None, "no", None, False),  # YAML 1.2 loads `no` as a string; it still means off
        (None, "off", None, False),
    ],
)
def test_the_layers_resolve_env_then_project_then_machine(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    env: str | None,
    project_value: str | None,
    machine_value: str | None,
    expected: bool,
) -> None:
    if env is not None:
        monkeypatch.setenv("TRW_PLATFORM_CONTACT_ENABLED", env)
    _switch(project, project_value)
    if machine_value is not None:
        _write(Path.home() / ".trw" / "config.yaml", f"platform_contact_enabled: {machine_value}\n")

    assert platform_contact_enabled() is expected


@pytest.mark.parametrize(
    ("layer", "body"),
    [
        ("machine", "platform_contact_enabled: flase\n"),  # not a bool
        ("project", "platform_contact_enabled: 2\n"),
        ("project", "platform_contact_enabled: [unclosed\n"),  # the file does not parse
        ("project", "- platform_contact_enabled\n- false\n"),  # parses, but is not a mapping (sol P2)
        ("machine", "false\n"),  # a bare scalar document is not a config either
    ],
)
def test_an_invalid_value_or_unreadable_file_fails_closed_with_a_warning(project: Path, layer: str, body: str) -> None:
    """B71-107: trw-memory raised and trw-mcp fell back to contact-on; this switch only restricts."""
    from structlog.testing import capture_logs

    _write((Path.home() if layer == "machine" else project) / ".trw" / "config.yaml", body)
    with capture_logs() as logs:
        assert platform_contact_enabled() is False
    assert {entry["outcome"] for entry in logs} == {"contact_off"}
    assert "flase" not in str(logs)  # the value itself is never logged


def test_without_a_named_root_the_nearest_trw_folder_above_the_cwd_decides(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command run from a subdirectory still finds the project's switch (agy critique, B71-106)."""
    _switch(project, "false")
    (project / "src" / "pkg").mkdir(parents=True)
    monkeypatch.delenv("TRW_PROJECT_ROOT")
    monkeypatch.chdir(project / "src" / "pkg")

    assert platform_contact_enabled() is False


# --- every boundary reads it live -------------------------------------------------------------------


class _Refuse(httpx.BaseTransport):
    """Records every request a sender tries to make and refuses it."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        raise httpx.ConnectError("refused in tests", request=request)


def _config(project: Path) -> MemoryConfig:
    return MemoryConfig(
        sync_enabled=True,
        platform_url="https://api.trwframework.com",
        platform_api_key="k",
        project_root=str(project),
    )


def test_a_running_config_stops_every_sender_the_moment_the_switch_turns_off(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B71-106: one config built while contact was on, then the operator switches it off."""
    refuse = _Refuse()
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: real_client(*a, **{**k, "transport": refuse}))
    cfg = _config(project)
    entry = MemoryEntry(id="L-1", content="a shareable learning", namespace="project:default", importance=0.9)
    queue = RetryQueue(tmp_path / "retry.jsonl")
    queue.enqueue("L-0", {"source_learning_id": "L-0", "summary": "queued"})

    publish_memory_result(entry, cfg)
    assert refuse.urls, "control: with contact on, the sender does try to connect"

    _switch(project, "false")
    refuse.urls.clear()
    publish_memory_result(entry, cfg)
    drain = drain_retry_queue(queue, cfg)
    retire_remote_memory("R-1", cfg)
    fetched = fetch_shared_memories("anything", cfg, admit=lambda _entry: True)

    assert refuse.urls == []
    assert drain["skipped"] == 1 and queue.depth() == 1  # kept for when contact returns
    assert fetched.status == "disabled"


def test_the_in_process_veto_stops_contact_even_when_the_switch_is_on(project: Path) -> None:
    cfg = _config(project).model_copy(update={"platform_contact_enabled": False})

    assert publish_memory_result(MemoryEntry(id="L-1", content="x", importance=0.9), cfg)["success"] is False


class _Stream:
    """An SSE response that stays open, silent, until it is closed."""

    def __init__(self) -> None:
        self.closed = threading.Event()

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.closed.set()

    def iter_lines(self) -> object:
        self.closed.wait(timeout=10)
        return iter(())

    def close(self) -> None:
        self.closed.set()


class _Client:
    connects: list[_Stream] = []

    def __init__(self, *_a: object, **_k: object) -> None:
        pass

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def stream(self, *_a: object, **_k: object) -> _Stream:
        stream = _Stream()
        self.connects.append(stream)
        return stream

    def close(self) -> None:
        return None


def test_the_subscriber_closes_an_open_stream_when_contact_turns_off_and_resumes_when_it_returns(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B71-106: the stream is silent, so only the watcher can notice; the loop then waits, no reconnect."""
    monkeypatch.setattr(subscriber_module, "SWITCH_POLL", 0.05)
    monkeypatch.setattr(subscriber_module, "RECONNECT_DELAY", 0.05)
    monkeypatch.setattr(subscriber_module.httpx, "Client", _Client)
    _Client.connects = []
    subscriber = SSESubscriber(_config(project), on_event=lambda _data: None)
    subscriber.start()
    try:
        _wait_for(lambda: len(_Client.connects) == 1)

        _switch(project, "false")
        assert _Client.connects[0].closed.wait(timeout=2), "the open stream is closed within a poll"
        threading.Event().wait(0.3)  # several reconnect periods pass
        assert len(_Client.connects) == 1, "no reconnect while contact is off"

        _switch(project, "true")
        _wait_for(lambda: len(_Client.connects) == 2)
    finally:
        subscriber.stop()
    assert not subscriber._thread.is_alive() and not subscriber._watcher.is_alive()


def _wait_for(condition: object, timeout: float = 3.0) -> None:
    done = threading.Event()
    for _ in range(int(timeout / 0.02)):
        if condition():  # type: ignore[operator]
            return
        done.wait(0.02)
    raise AssertionError("condition not reached")


class _StalledHandshake:
    """A client whose connect never completes until the client is closed from another thread."""

    instances: list[_StalledHandshake] = []

    def __init__(self, *_a: object, **_k: object) -> None:
        self.closed = threading.Event()
        self.instances.append(self)

    def __enter__(self) -> _StalledHandshake:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def stream(self, *_a: object, **_k: object) -> object:
        self.closed.wait(timeout=10)
        raise httpx.ConnectError("closed during the handshake")

    def close(self) -> None:
        self.closed.set()


def test_stop_closes_a_stalled_handshake(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """sol P1: the client is registered before it connects, so stop() has something to close."""
    monkeypatch.setattr(subscriber_module.httpx, "Client", _StalledHandshake)
    _StalledHandshake.instances = []
    subscriber = SSESubscriber(_config(project), on_event=lambda _data: None)
    subscriber.start()
    _wait_for(lambda: len(_StalledHandshake.instances) == 1)

    subscriber.stop()

    assert _StalledHandshake.instances[0].closed.is_set()
    assert not subscriber._thread.is_alive()


def test_a_switch_turned_off_during_the_handshake_reads_nothing(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[object] = []

    class _FlipOnConnect(_Client):
        def stream(self, *_a: object, **_k: object) -> _Stream:
            _switch(project, "false")  # the operator switches contact off while the connection opens
            stream = _Stream()
            stream.iter_lines = lambda: iter(["event: learning_published", 'data: {"summary": "x"}', ""])  # type: ignore[method-assign]
            self.connects.append(stream)
            return stream

    monkeypatch.setattr(subscriber_module, "RECONNECT_DELAY", 0.05)
    monkeypatch.setattr(subscriber_module.httpx, "Client", _FlipOnConnect)
    _Client.connects = []
    subscriber = SSESubscriber(_config(project), on_event=received.append)
    subscriber.start()
    try:
        _wait_for(lambda: len(_Client.connects) == 1)
        threading.Event().wait(0.2)
    finally:
        subscriber.stop()
    assert received == []
    assert len(_Client.connects) == 1  # and no reconnect while it stays off


def test_a_fault_while_resolving_fails_closed_instead_of_raising(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A working directory removed between two retries made Path.cwd() raise out of a sender's retry loop."""
    monkeypatch.delenv("TRW_PROJECT_ROOT")

    def _vanished() -> Path:
        raise FileNotFoundError("the working directory was removed")

    monkeypatch.setattr(Path, "cwd", staticmethod(_vanished))

    assert platform_contact_enabled() is False


def test_a_sender_whose_check_faults_between_attempts_stops_quietly(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry loop asks the switch before every POST; a fault on the second ask ends the loop, no traceback."""
    refuse = _Refuse()
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: real_client(*a, **{**k, "transport": refuse}))
    cfg = _config(project)
    entry = MemoryEntry(id="L-1", content="a shareable learning", namespace="project:default", importance=0.9)
    publish_memory_result(entry, cfg)  # one POST with the switch readable
    assert len(refuse.urls) == 1

    monkeypatch.setattr("trw_memory.platform_contact._resolve", lambda *_a: 1 / 0)  # the next ask faults
    assert publish_memory_result(entry, cfg)["success"] is False
    assert len(refuse.urls) == 1


def test_even_a_failing_log_call_cannot_escape_the_fail_closed_path(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trw_memory.platform_contact as module

    monkeypatch.setattr(module, "_resolve", lambda *_a: 1 / 0)
    monkeypatch.setattr(module.logger, "warning", lambda *_a, **_k: 1 / 0)

    assert platform_contact_enabled() is False
