"""PRD-INFRA-200 FR02 / NFR02: an upgrade stops the daemon only when it serves another version.

Every case plants a real ``daemon.json`` naming a real live process, so the
identity check (pid plus OS start) is the one production runs, never a stub.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.daemon import DaemonPaths, stop_outdated_daemon
from trw_memory.daemon._discovery import DaemonInfo
from trw_memory.daemon._paths import write_secret_file
from trw_memory.storage._pid_liveness import process_start

_INSTALLED = "9.9.1"


@pytest.fixture
def paths(tmp_path: Path) -> DaemonPaths:
    return DaemonPaths(user_memory_dir=tmp_path / "memory")


@pytest.fixture
def sleeper() -> Iterator[subprocess.Popen[bytes]]:
    """A live process standing in for the serving daemon."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield process
    process.kill()
    process.wait()


def _plant(paths: DaemonPaths, pid: int, *, version: str, start: str | None) -> None:
    paths.user_memory_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = DaemonInfo(
        pid=pid,
        url="http://127.0.0.1:9/mcp",
        started_at=datetime.now(timezone.utc).isoformat(),
        version=version,
        process_start=start,
    )
    write_secret_file(paths.discovery, info.model_dump_json())


@pytest.mark.parametrize("served", ["9.9.0", "9.8.7", "8.0.0"])
def test_a_daemon_serving_another_version_is_stopped(
    paths: DaemonPaths, sleeper: subprocess.Popen[bytes], served: str
) -> None:
    _plant(paths, sleeper.pid, version=served, start=process_start(sleeper.pid))

    result = stop_outdated_daemon(paths, _INSTALLED)

    assert result.outcome == "stopped"
    assert str(sleeper.pid) in result.detail
    sleeper.wait(timeout=10)  # raises TimeoutExpired: the outdated daemon is still running


def test_a_daemon_already_serving_the_installed_version_is_left_running(
    paths: DaemonPaths, sleeper: subprocess.Popen[bytes]
) -> None:
    _plant(paths, sleeper.pid, version=_INSTALLED, start=process_start(sleeper.pid))

    assert stop_outdated_daemon(paths, _INSTALLED).outcome == "current"
    assert sleeper.poll() is None


@pytest.mark.parametrize(
    ("start", "outcome"),
    [
        (None, "unproven"),  # a 4.0 record: the pid may name any process now
        ("not-the-sleepers-start", "absent"),  # a reused pid: the record's process is gone
    ],
)
def test_a_record_whose_identity_is_not_proven_is_never_signalled(
    paths: DaemonPaths, sleeper: subprocess.Popen[bytes], start: str | None, outcome: str
) -> None:
    _plant(paths, sleeper.pid, version="1.0.0", start=start)

    result = stop_outdated_daemon(paths, _INSTALLED)

    assert result.outcome == outcome
    assert sleeper.poll() is None, "a process not proven to be the daemon was signalled"


def test_an_unproven_record_carries_the_manual_remedy(paths: DaemonPaths, sleeper: subprocess.Popen[bytes]) -> None:
    _plant(paths, sleeper.pid, version="1.0.0", start=None)

    assert f"process {sleeper.pid}" in stop_outdated_daemon(paths, _INSTALLED).detail


def test_no_record_and_an_untrusted_record_are_reported_not_acted_on(paths: DaemonPaths) -> None:
    assert stop_outdated_daemon(paths, _INSTALLED).outcome == "absent"

    paths.user_memory_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_secret_file(paths.discovery, "{not json")
    result = stop_outdated_daemon(paths, _INSTALLED)
    assert result.outcome == "invalid"
    assert str(paths.discovery) in result.detail


def test_a_start_that_cannot_be_read_now_proves_nothing(
    paths: DaemonPaths, sleeper: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Liveness keeps an unreadable start live (no second daemon); a signal must read it as unproven.

    Codex sol round 1 on infra-200-e: ``is_process_live`` answers live when the current start reading is
    ``None``, so a stop gated on it alone signalled a pid whose identity nothing had proven.
    """
    from trw_memory.daemon import _spawn

    _plant(paths, sleeper.pid, version="1.0.0", start=process_start(sleeper.pid))
    monkeypatch.setattr(_spawn, "process_start", lambda _pid: None)

    assert stop_outdated_daemon(paths, _INSTALLED).outcome == "unproven"
    assert _spawn.SpawnedDaemon(sleeper.pid, "any-recorded-start", paths.lock).stop() is False
    assert sleeper.poll() is None, "a process whose start could not be read was signalled"


def test_a_stop_refused_at_signal_time_is_not_reported_as_stopped(
    paths: DaemonPaths, sleeper: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex sol round 2 on infra-200-e: the stop's own refusal must reach the outcome."""
    from trw_memory.daemon import _spawn

    _plant(paths, sleeper.pid, version="1.0.0", start=process_start(sleeper.pid))
    monkeypatch.setattr(_spawn.SpawnedDaemon, "stop", lambda _self: False)

    result = stop_outdated_daemon(paths, _INSTALLED)

    assert result.outcome == "unproven"
    assert "not signalled" in result.detail


def _planted(paths: DaemonPaths) -> DaemonInfo:
    from trw_memory.daemon._discovery import read_discovery_result

    found = read_discovery_result(paths)
    assert isinstance(found, DaemonInfo)
    return found


@pytest.mark.parametrize(
    "served",
    [
        _INSTALLED,  # equal
        "9.9.2",  # newer patch
        "10.0.0",  # newer major, and a numeric (not lexical) comparison: "10" < "9" as text
        "9.10.0",  # newer minor, numerically
        "not-a-version",  # unparseable: never read as older
        "",
    ],
)
def test_older_only_never_stops_an_equal_newer_or_unparseable_daemon(
    paths: DaemonPaths, sleeper: subprocess.Popen[bytes], served: str
) -> None:
    """DAEMON-AUTO-RESTART-ON-UPGRADE: a newer daemon is never stopped; an older client must not downgrade it."""
    _plant(paths, sleeper.pid, version=served, start=process_start(sleeper.pid))

    result = stop_outdated_daemon(paths, _INSTALLED, expect=_planted(paths), older_only=True)

    assert result.outcome in {"current", "not_older"}
    assert sleeper.poll() is None, f"a daemon serving {served!r} was signalled by an older_only stop"


@pytest.mark.parametrize("served", ["9.9.0", "9.8.7", "8.0.0", "9.9"])
def test_older_only_stops_a_proven_older_daemon(
    paths: DaemonPaths, sleeper: subprocess.Popen[bytes], served: str
) -> None:
    _plant(paths, sleeper.pid, version=served, start=process_start(sleeper.pid))

    result = stop_outdated_daemon(paths, _INSTALLED, expect=_planted(paths), older_only=True)

    assert result.outcome == "stopped"
    sleeper.wait(timeout=10)


def test_a_record_swapped_since_the_caller_observed_it_is_changed_and_not_signalled(
    paths: DaemonPaths, sleeper: subprocess.Popen[bytes]
) -> None:
    """The caller names the instance it saw; a different instance at signal time is reported, never signalled."""
    _plant(paths, sleeper.pid, version="1.0.0", start=process_start(sleeper.pid))
    observed = _planted(paths).model_copy(update={"process_start": "an-earlier-instance"})

    result = stop_outdated_daemon(paths, _INSTALLED, expect=observed)

    assert result.outcome == "changed"
    assert str(sleeper.pid) in result.detail
    assert sleeper.poll() is None, "an instance the caller never observed was signalled"


def test_init_is_never_signalled_even_when_its_start_reads_back(
    paths: DaemonPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stubbed launcher made a test's spawn record pid 1 with launchd's real start; its reaper sent SIGKILL.

    Found by the dev16 canary (2026-09-28): ``patch("subprocess.run")`` returns a MagicMock, and
    ``int(MagicMock())`` is 1, so start_daemon_detached recorded init. Only EPERM stopped the signal.
    """
    from trw_memory.daemon import _spawn

    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(_spawn.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    # Liveness as the canary saw it (live, start proven), so only the pid guard can refuse the signal.
    monkeypatch.setattr(_spawn.SpawnedDaemon, "running", lambda _self: True)
    monkeypatch.setattr(_spawn, "_STOP_GRACE_SECONDS", 0.0)
    _plant(paths, 1, version="1.0.0", start=process_start(1))

    assert _spawn.SpawnedDaemon(1, process_start(1), paths.lock).stop() is False
    assert stop_outdated_daemon(paths, _INSTALLED).outcome != "stopped"
    assert [s for s in sent if s[1] != 0] == [], "init was sent a real signal (0 is only a liveness probe)"


def test_a_launcher_reporting_init_is_a_failed_start(paths: DaemonPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    """A launcher whose output parses to pid 1 or less started no daemon; it must not be recorded as one."""
    from unittest.mock import MagicMock

    from trw_memory.daemon import _spawn
    from trw_memory.exceptions import DaemonUnreachableError

    paths.user_memory_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    monkeypatch.setattr(_spawn.subprocess, "run", lambda *_a, **_k: MagicMock())

    with pytest.raises(DaemonUnreachableError, match="launcher failed"):
        _spawn.start_daemon_detached(paths)
