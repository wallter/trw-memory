"""An auto-started daemon that never publishes is stopped, and says why in a log.

2026-09-24: one ``init-project`` under a loaded test run spawned five daemons
about four seconds apart. Each one stalled before creating its memory directory;
the client gave up on each and the next call spawned another. A daemon stuck
before ``serve()`` never reaches its idle timer, so all five lived until someone
killed them, and with output sent to DEVNULL nothing said why they stalled.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from trw_memory.daemon import DaemonPaths
from trw_memory.daemon import client as client_module
from trw_memory.daemon.client import DaemonClient
from trw_memory.exceptions import DaemonUnreachableError
from trw_memory.models.config import MemoryConfig

#: Stands in for ``-m trw_memory.server serve http``: says something, then hangs before publishing.
_STALLS = ("-c", "import sys, time; print('stalled before publishing', file=sys.stderr, flush=True); time.sleep(60)")


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DaemonPaths:
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "userhome"))
    return DaemonPaths.resolve()


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Every process the real ``start_daemon_detached`` starts, with the stalling argv."""
    processes: list[object] = []
    real = client_module.start_daemon_detached

    def record(paths: DaemonPaths) -> object:
        process = real(paths)
        processes.append(process)
        return process

    monkeypatch.setattr(client_module, "_DAEMON_ARGV", _STALLS)
    monkeypatch.setattr(client_module, "start_daemon_detached", record)
    yield processes
    for process in processes:
        process.kill()  # type: ignore[attr-defined]
        process.wait()  # type: ignore[attr-defined]


def _client(paths: DaemonPaths) -> DaemonClient:
    return DaemonClient("grant", config=MemoryConfig(memory_daemon_startup_timeout_seconds=0.5), paths=paths)


def test_a_start_that_never_publishes_is_stopped_before_the_client_gives_up(
    paths: DaemonPaths, spawned: list[object]
) -> None:
    with pytest.raises(DaemonUnreachableError, match="stopped the daemon it started"):
        _client(paths)._attach()

    [process] = spawned
    assert process.poll() is not None, "the stalled daemon outlived the client that started it"  # type: ignore[attr-defined]


def test_repeated_attempts_do_not_accumulate_stalled_daemons(paths: DaemonPaths, spawned: list[object]) -> None:
    client = _client(paths)
    for _ in range(3):
        with pytest.raises(DaemonUnreachableError):
            client._attach()

    assert len(spawned) == 3
    assert [process.poll() is not None for process in spawned] == [True, True, True]  # type: ignore[attr-defined]


def test_the_stalled_start_says_why_in_a_private_log(paths: DaemonPaths, spawned: list[object]) -> None:
    with pytest.raises(DaemonUnreachableError) as raised:
        _client(paths)._attach()

    log = paths.user_memory_dir / "daemon-start.log"
    assert "stalled before publishing" in log.read_text(encoding="utf-8")
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert str(log) in str(raised.value), "the error should point at the log"


def test_each_start_truncates_the_previous_log(paths: DaemonPaths, spawned: list[object]) -> None:
    client = _client(paths)
    for _ in range(2):
        with pytest.raises(DaemonUnreachableError):
            client._attach()

    assert (paths.user_memory_dir / "daemon-start.log").read_text(encoding="utf-8").count("stalled") == 1


def test_a_stub_that_spawned_nothing_still_fails_closed(paths: DaemonPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers stub the spawn with ``lambda _paths: None``; the timeout path must still raise, not crash."""
    monkeypatch.setattr(client_module, "start_daemon_detached", lambda _paths: None)

    with pytest.raises(DaemonUnreachableError, match="did not publish"):
        _client(paths)._attach()


def test_the_daemon_argv_is_the_module_entry_point() -> None:
    assert (sys.executable, *client_module._DAEMON_ARGV) == (sys.executable, "-m", "trw_memory.server", "serve", "http")
