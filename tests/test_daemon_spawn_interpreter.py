"""An auto-started daemon runs from the interpreter its caller names (DAEMON-AUTOSTART-VERSION-RACE).

After a drain, the first client to call starts the replacement. Left to itself it starts it from
its OWN interpreter, so a stale client beside a newer shared server publishes an older daemon.
A client that knows which installation must serve passes a ``launcher``; the spawn itself takes
the interpreter and the child's whole environment.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from trw_memory.daemon import DaemonPaths
from trw_memory.daemon import _spawn as spawn_module
from trw_memory.daemon import client as client_module
from trw_memory.daemon.client import DaemonClient
from trw_memory.exceptions import DaemonUnreachableError
from trw_memory.models.config import MemoryConfig


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DaemonPaths:
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "userhome"))
    monkeypatch.setenv("MEMORY_DAEMON_AUTOSTART", "true")  # the impact gate and policy runs set it false
    return DaemonPaths.resolve()


class _Run:
    """Stands in for ``subprocess.run``: records the launcher's argv and environment, reports a pid."""

    def __init__(self) -> None:
        self.argv: list[str] = []
        self.env: dict[str, str] | None = {}

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.argv, self.env = list(argv), kwargs.get("env")
        return subprocess.CompletedProcess(argv, 0, stdout=b"4242\n")


def test_the_daemon_starts_from_the_named_interpreter_with_exactly_the_named_environment(
    paths: DaemonPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _Run()
    monkeypatch.setattr(spawn_module.subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", "/callers/worktree/src")

    spawn_module.start_daemon_detached(paths, python="/envs/stable/bin/python", environ={"HOME": "/h", "X": "1"})

    assert run.argv[0] == "/envs/stable/bin/python", "the launcher runs under the named interpreter"
    assert run.argv[run.argv.index("-c") + 2] == "/envs/stable/bin/python", "and starts the daemon under it too"
    assert run.argv[-4:] == list(spawn_module._DAEMON_ARGV)
    assert run.env == {"HOME": "/h", "X": "1"}, "the caller's PYTHONPATH must not reach another installation"


def test_without_a_named_interpreter_the_daemon_starts_from_this_one_and_inherits(
    paths: DaemonPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _Run()
    monkeypatch.setattr(spawn_module.subprocess, "run", run)

    spawn_module.start_daemon_detached(paths)

    assert run.argv[0] == sys.executable
    assert run.env is None, "the default start still inherits the caller's environment"


def test_the_named_interpreter_really_runs_the_daemon_with_the_named_environment(
    paths: DaemonPaths, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End to end: a wrapper stands in for another venv's python, so a daemon that skipped it leaves no mark."""
    wrapper = tmp_path / "other-venv-python"
    marks = tmp_path / "marks"
    wrapper.write_text(f'#!/bin/sh\necho "$0" >> {marks}\nexec {sys.executable} "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    seen = tmp_path / "seen"
    program = f"import os; open({str(seen)!r}, 'w').write(repr((os.environ.get('PYTHONPATH'), os.environ.get('LEAK'))))"
    monkeypatch.setattr(spawn_module, "_DAEMON_ARGV", ("-c", program))
    monkeypatch.setenv("PYTHONPATH", "/callers/worktree/src")
    monkeypatch.setenv("LEAK", "from-the-caller")

    spawned = spawn_module.start_daemon_detached(
        paths, python=str(wrapper), environ={"PATH": "/usr/bin:/bin", "PYTHONPATH": "/envs/stable/src"}
    )
    deadline = time.monotonic() + 30
    while not seen.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    spawned.stop()

    assert seen.read_text(encoding="utf-8") == repr(("/envs/stable/src", None))
    assert marks.read_text(encoding="utf-8").split() == [str(wrapper)] * 2, (
        "launcher and daemon both ran via the wrapper"
    )


def test_a_client_with_a_launcher_starts_the_daemon_through_it_and_never_through_its_own_interpreter(
    paths: DaemonPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    own: list[DaemonPaths] = []
    monkeypatch.setattr(client_module, "start_daemon_detached", own.append)
    launched: list[DaemonPaths] = []

    def launcher(daemon_paths: DaemonPaths) -> None:
        launched.append(daemon_paths)

    client = DaemonClient(
        "grant",
        config=MemoryConfig(memory_daemon_startup_timeout_seconds=0.2),
        paths=paths,
        launcher=launcher,  # type: ignore[arg-type]  # a stub that spawns nothing, as the stall tests do
    )
    with pytest.raises(DaemonUnreachableError, match="did not publish"):
        client._attach()

    assert launched == [paths]
    assert own == [], "the client's own interpreter must not start a daemon beside the named one"


def test_a_client_without_a_launcher_still_starts_the_daemon_itself(
    paths: DaemonPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    own: list[DaemonPaths] = []
    monkeypatch.setattr(client_module, "start_daemon_detached", own.append)

    with pytest.raises(DaemonUnreachableError):
        DaemonClient("grant", config=MemoryConfig(memory_daemon_startup_timeout_seconds=0.2), paths=paths)._attach()

    assert own == [paths]


def test_a_launcher_is_not_asked_when_auto_start_is_off(paths: DaemonPaths) -> None:
    launched: list[DaemonPaths] = []
    client = DaemonClient(
        "grant",
        config=MemoryConfig(memory_daemon_autostart=False),
        paths=paths,
        launcher=launched.append,  # type: ignore[arg-type]
    )
    with pytest.raises(DaemonUnreachableError, match="auto-start is off"):
        client._attach()

    assert launched == []
