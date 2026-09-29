"""The shared daemon reaper both suites' conftests use (``trw_memory.testing.daemon_reaper``).

Moved from trw-mcp's ``tests/test_daemon_reaper.py`` (placement, bystander, COLUMNS,
spawn-handle arms) and ``tests/test_daemon_owner_reap.py`` (reap-by-owner arm) when
the two ``tests/_daemon_reaper.py`` copies became this one module.

2026-09-24: five daemons from one test stalled before creating their memory
directory. No discovery file named them, so a sweep by discovery file could not
find them; the auto-started daemon's environment still places it under the run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.testing import daemon_reaper
from trw_memory.testing.daemon_reaper import (
    IDLE_VARIABLE,
    OWNER_VARIABLE,
    SessionSweep,
    daemon_env_passthrough,
    daemon_pids_owned_by,
    daemon_pids_under,
    reap_daemons_under,
    stop_spawned,
    sweep_session_daemons,
)

pytestmark = pytest.mark.integration

_SLEEP = "import time; time.sleep(60)"


@pytest.fixture
def spawned() -> Iterator[list[subprocess.Popen[bytes]]]:
    processes: list[subprocess.Popen[bytes]] = []
    yield processes
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait()


def _unpublished(cwd: Path, env: dict[str, str], *, detached: bool = False) -> subprocess.Popen[bytes]:
    """A process whose command line is the daemon's and that never wrote ``daemon.json``."""
    return subprocess.Popen(
        [sys.executable, "-c", _SLEEP, "trw_memory.server", "serve", "http"],
        env={**os.environ, **env},
        cwd=cwd,
        start_new_session=detached,
    )


def _owner() -> str:
    """A per-run owner token, so a concurrent run's daemons can never match."""
    return f"test-{uuid.uuid4().hex[:12]}"


# ── placement and discovery ──────────────────────────────────────────────


@pytest.mark.parametrize("variable", ["TRW_USER_DIR", "HOME"])
def test_an_unpublished_daemon_placed_under_the_run_is_stopped(
    tmp_path: Path, spawned: list[subprocess.Popen[bytes]], variable: str
) -> None:
    root, elsewhere = tmp_path / "run", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    daemon = _unpublished(elsewhere, {variable: str(root / "home" / ".trw")})
    spawned.append(daemon)

    assert reap_daemons_under(root, wait=True, by_process=True) == [daemon.pid]
    assert daemon.wait(timeout=10) is not None


def test_an_unpublished_daemon_running_in_the_run_is_stopped(
    tmp_path: Path, spawned: list[subprocess.Popen[bytes]]
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    daemon = _unpublished(root, {"TRW_USER_DIR": str(tmp_path / "outside"), "HOME": str(tmp_path / "outside")})
    spawned.append(daemon)

    assert reap_daemons_under(root, wait=True, by_process=True) == [daemon.pid]


def test_a_published_daemon_is_found_by_its_discovery_file_without_a_process_scan(
    tmp_path: Path, spawned: list[subprocess.Popen[bytes]]
) -> None:
    root, elsewhere = tmp_path / "run", tmp_path / "elsewhere"
    (root / "user" / "memory").mkdir(parents=True)
    elsewhere.mkdir()
    daemon = _unpublished(elsewhere, {"TRW_USER_DIR": str(elsewhere), "HOME": str(elsewhere)})
    spawned.append(daemon)
    (root / "user" / "memory" / "daemon.json").write_text(json.dumps({"pid": daemon.pid}), encoding="utf-8")
    (root / "partial" / "memory").mkdir(parents=True)
    (root / "partial" / "memory" / "daemon.json").write_text("{", encoding="utf-8")

    assert daemon_pids_under(root) == [daemon.pid]
    assert reap_daemons_under(root, wait=True) == [daemon.pid]


def test_a_daemon_placed_and_running_elsewhere_is_left_alone(
    tmp_path: Path, spawned: list[subprocess.Popen[bytes]]
) -> None:
    root, elsewhere = tmp_path / "run", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    other = _unpublished(elsewhere, {"TRW_USER_DIR": str(elsewhere), "HOME": str(elsewhere)})
    spawned.append(other)

    assert reap_daemons_under(root, wait=True, by_process=True) == []
    assert other.poll() is None


def test_a_process_placed_under_the_run_that_is_not_a_daemon_is_never_signalled(
    tmp_path: Path, spawned: list[subprocess.Popen[bytes]]
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    bystander = subprocess.Popen([sys.executable, "-c", _SLEEP], env={**os.environ, "HOME": str(root)}, cwd=root)
    spawned.append(bystander)

    assert reap_daemons_under(root, wait=True, by_process=True) == []
    assert bystander.poll() is None


def test_a_narrow_columns_setting_does_not_hide_a_daemon(
    tmp_path: Path, spawned: list[subprocess.Popen[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linux ps cuts piped output to COLUMNS, and pytest sets COLUMNS: the command-line mark was lost (6.1.0 Linux leg)."""
    monkeypatch.setenv("COLUMNS", "20")
    root = tmp_path / "run"
    root.mkdir()
    daemon = _unpublished(root, {"TRW_USER_DIR": str(root / "home" / ".trw")})
    spawned.append(daemon)

    assert reap_daemons_under(root, wait=True, by_process=True) == [daemon.pid]


# ── owner ────────────────────────────────────────────────────────────────


def test_the_sweep_stops_a_daemon_by_owner_even_when_no_path_lies_under_the_run(
    tmp_path: Path, spawned: list[subprocess.Popen[bytes]]
) -> None:
    """Only the owner tag places this daemon; a different owner's daemon is left alone."""
    run, elsewhere = tmp_path / "run", tmp_path / "elsewhere"
    run.mkdir()
    elsewhere.mkdir()
    outside = {"HOME": str(elsewhere), "TRW_USER_DIR": str(elsewhere)}
    owner_a, owner_b = _owner(), _owner()
    mine = _unpublished(elsewhere, {**outside, OWNER_VARIABLE: owner_a}, detached=True)
    theirs = _unpublished(elsewhere, {**outside, OWNER_VARIABLE: owner_b}, detached=True)
    spawned += [mine, theirs]

    assert reap_daemons_under(run, wait=True, by_process=True, owner=owner_a) == [mine.pid]
    assert mine.wait(timeout=10) is not None
    assert theirs.poll() is None, "another session's daemon must survive this session's sweep"
    assert daemon_pids_owned_by(owner_b) == [theirs.pid]


def test_the_session_sweep_reports_what_it_stopped_and_finds_no_survivor(
    tmp_path: Path, spawned: list[subprocess.Popen[bytes]]
) -> None:
    basetemp, elsewhere = tmp_path / "bt", tmp_path / "elsewhere"
    basetemp.mkdir()
    elsewhere.mkdir()
    owner = _owner()
    placed = _unpublished(basetemp, {"HOME": str(elsewhere), "TRW_USER_DIR": str(elsewhere)})
    owned = _unpublished(
        elsewhere, {"HOME": str(elsewhere), "TRW_USER_DIR": str(elsewhere), OWNER_VARIABLE: owner}, detached=True
    )
    spawned += [placed, owned]

    sweep = sweep_session_daemons(basetemp, owner)

    assert sweep.leaked == sorted([placed.pid, owned.pid])
    assert sweep.survivors == []
    assert f"{OWNER_VARIABLE}={owner}" in sweep.placements[owned.pid]
    assert sweep.report()[0].startswith("FAIL: 2 leaked memory daemon(s) stopped at session end under ")


def test_the_session_sweep_censuses_survivors_after_reaping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The survivor guard: a daemon still found after the reap is reported, even if the reap counted none."""
    monkeypatch.setattr(daemon_reaper, "reap_daemons_under", lambda *_a, **_k: [])
    monkeypatch.setattr(daemon_reaper, "daemon_pids_placed_under", lambda _root: [7])
    monkeypatch.setattr(daemon_reaper, "daemon_pids_owned_by", lambda owner: [9] if owner == "o" else [])

    assert sweep_session_daemons(tmp_path, "o").survivors == [7, 9]
    assert sweep_session_daemons(tmp_path, None).survivors == [7]


@pytest.mark.parametrize(
    ("sweep", "handed_over", "expected"),
    [
        (SessionSweep(Path("/bt"), [], []), [], []),
        (
            SessionSweep(Path("/bt"), [1], [], {1: "cwd=/bt/x"}),
            [2],
            [
                "FAIL: 2 leaked memory daemon(s) stopped at session end under /bt: pids [1, 2]",
                "  leaked daemon 1: cwd=/bt/x",
                "  leaked daemon 2: ?",
            ],
        ),
        (SessionSweep(Path("/bt"), [], [5]), [], ["FAIL: memory daemon(s) survived the session-end sweep: pids [5]"]),
    ],
    ids=["clean", "own-and-worker-leaks", "survivor"],
)
def test_the_session_report_names_every_leak_and_survivor(
    sweep: SessionSweep, handed_over: list[int], expected: list[str]
) -> None:
    assert sweep.report(handed_over) == expected


# ── the child env a sanitized builder must pass through ──────────────────


def test_the_passthrough_carries_exactly_the_owner_and_the_idle_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OWNER_VARIABLE, "owner-x")
    monkeypatch.setenv(IDLE_VARIABLE, "60")
    monkeypatch.setenv("TRW_SECRET_THING", "never")

    assert daemon_env_passthrough() == {OWNER_VARIABLE: "owner-x", IDLE_VARIABLE: "60"}


def test_the_passthrough_omits_a_variable_that_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OWNER_VARIABLE, raising=False)
    monkeypatch.delenv(IDLE_VARIABLE, raising=False)

    assert daemon_env_passthrough() == {}


def test_tagging_gives_this_process_a_token_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(IDLE_VARIABLE, raising=False)
    monkeypatch.setenv(OWNER_VARIABLE, "stale")

    first = daemon_reaper.tag_daemon_ownership()
    second = daemon_reaper.tag_daemon_ownership()

    assert first.startswith(f"{os.getpid()}-") and first != second
    assert os.environ[OWNER_VARIABLE] == second
    assert os.environ[IDLE_VARIABLE] == "60"


# ── the spawn handle (C1 2026-09-25: a daemon a test auto-started is stopped before it publishes) ──


def test_stop_spawned_stops_a_running_daemon_and_skips_an_exited_one(
    tmp_path: Path, spawned: list[subprocess.Popen[bytes]]
) -> None:
    from trw_memory.daemon._spawn import SpawnedDaemon
    from trw_memory.storage._pid_liveness import process_start

    running = _unpublished(tmp_path, {})
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait()
    spawned.append(running)
    handles = [SpawnedDaemon(p.pid, process_start(p.pid), tmp_path / "lock") for p in (running, exited)]

    assert stop_spawned(handles) == [running.pid]
    assert running.wait(timeout=10) is not None


def test_the_module_imports_only_the_stdlib() -> None:
    """It ships in the wheel, where pytest is not installed: every import must be stdlib (or ``__future__``)."""
    import ast

    for module in (daemon_reaper, sys.modules["trw_memory.testing"]):
        tree = ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8"))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        outside = {name for name in imported if name.split(".")[0] not in (*sys.stdlib_module_names, "__future__")}
        assert outside == set(), f"{module.__name__} imports {sorted(outside)}"


def test_the_controller_report_fails_on_survivors_a_worker_handed_over() -> None:
    """Under xdist a worker's exit status never reaches the controller, so its survivors must be handed up."""
    report = SessionSweep(Path("/bt"), [], []).report(handed_over_survivors=[42])

    assert report == ["FAIL: memory daemon(s) survived the session-end sweep: pids [42]"]
