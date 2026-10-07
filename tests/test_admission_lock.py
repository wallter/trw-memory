"""The machine-wide admission lock for wide pytest runs (tests/_admission_lock.py).

Every test points the lock at its own ``tmp_path`` through ``LOCK_ENV`` in an explicit
env dict, so none of them touches the real ``~/.cache/trw-pytest/heavy.lock`` or the
running session's environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import _admission_lock as al


def _env(lock: Path, **extra: str) -> dict[str, str]:
    return {al.LOCK_ENV: str(lock), **extra}


@pytest.mark.parametrize(
    ("numprocesses", "expected"),
    [(None, False), (0, False), (1, False), (2, False), (3, True), (4, True), ("auto", True), ("logical", True)],
)
def test_only_a_controller_wider_than_two_workers_needs_admission(numprocesses: object, expected: bool) -> None:
    assert al.needs_admission(numprocesses, {}) is expected


@pytest.mark.parametrize(
    "marker",
    [{"PYTEST_XDIST_WORKER": "gw0"}, {al.HELD_ENV: "1234"}, {al.SKIP_ENV: "1"}],
    ids=["xdist-worker", "nested-run", "operator-skip"],
)
def test_workers_nested_runs_and_the_skip_override_never_need_admission(marker: dict[str, str]) -> None:
    assert al.needs_admission(4, marker) is False


def test_acquire_records_the_holder_and_release_lets_the_next_run_in(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"

    fd = al.acquire(lock, wait=0, cmdline="pytest tests -n 4")
    assert al.read_holder(lock) == f"pid {os.getpid()}: pytest tests -n 4"
    with pytest.raises(al.AdmissionRefused):
        al.acquire(lock, wait=0, cmdline="second")
    al.release(fd)

    al.release(al.acquire(lock, wait=0, cmdline="second"))


def test_a_second_controller_waits_then_fails_naming_the_holder(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    held = al.acquire(lock, wait=0, cmdline="python -m pytest tests -n 4 (first agent)")
    try:
        started = time.monotonic()
        with pytest.raises(al.AdmissionRefused) as refused:
            al.acquire(lock, wait=0.3, cmdline="second agent", poll=0.05)
        waited = time.monotonic() - started
    finally:
        al.release(held)

    message = str(refused.value)
    assert f"pid {os.getpid()}" in message
    assert "python -m pytest tests -n 4 (first agent)" in message
    assert al.SKIP_ENV in message
    # It waited out its bound before refusing (a lower bound only: never an upper one).
    assert waited >= 0.3


def test_admit_marks_the_env_so_workers_and_nested_runs_skip_the_held_lock(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    env = _env(lock, **{al.WAIT_ENV: "0"})

    fd = al.admit(4, env)
    assert fd is not None
    try:
        assert env[al.HELD_ENV] == str(os.getpid())
        # A nested pytest inherits the marker: it returns at once instead of waiting on its parent.
        assert al.admit(4, dict(env)) is None
        # An xdist worker skips it even without the marker.
        worker_env = _env(lock, PYTEST_XDIST_WORKER="gw1", **{al.WAIT_ENV: "0"})
        assert al.admit(4, worker_env) is None
        # Anyone else is refused while it is held.
        with pytest.raises(al.AdmissionRefused):
            al.admit(4, _env(lock, **{al.WAIT_ENV: "0"}))
    finally:
        al.dismiss(fd, env)
    assert al.HELD_ENV not in env
    # Dismissed means released: the next controller is admitted at once.
    after = _env(lock, **{al.WAIT_ENV: "0"})
    again = al.admit(4, after)
    assert again is not None
    al.dismiss(again, after)


def test_a_narrow_run_never_creates_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "sub" / "heavy.lock"
    assert al.admit(2, _env(lock)) is None
    assert not lock.parent.exists()


def test_the_lock_is_released_when_its_holder_dies(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    child = (
        "import importlib.util, sys, time\n"
        "from pathlib import Path\n"
        "spec = importlib.util.spec_from_file_location('al', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "m.acquire(Path(sys.argv[2]), wait=0, cmdline='doomed holder')\n"
        "print('held', flush=True)\n"
        "time.sleep(120)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child, al.__file__, str(lock)],
        stdout=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)},
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "held"
        with pytest.raises(al.AdmissionRefused, match=f"pid {proc.pid}: doomed holder"):
            al.acquire(lock, wait=0, cmdline="waiter")
    finally:
        proc.kill()
        proc.wait(timeout=30)

    al.release(al.acquire(lock, wait=5, cmdline="after the holder died", poll=0.05))


def test_the_conftest_refuses_a_wide_run_while_another_holds_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wiring: the package conftest's ``pytest_configure`` really consults the lock."""
    from tests import conftest

    lock = tmp_path / "heavy.lock"
    monkeypatch.setenv(al.LOCK_ENV, str(lock))
    monkeypatch.setenv(al.WAIT_ENV, "0")
    for name in ("PYTEST_XDIST_WORKER", al.HELD_ENV, al.SKIP_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(conftest, "tag_daemon_ownership", lambda: "owner")
    monkeypatch.setattr(conftest, "_refuse_on_low_disk", lambda _config: None)
    monkeypatch.setattr(conftest, "snapshot_package_store", lambda _config: None)
    monkeypatch.delenv("TRW_REQUIRE_SQLITE_VEC", raising=False)
    config = SimpleNamespace(option=SimpleNamespace(numprocesses=4), stash=pytest.Stash())

    held = al.acquire(lock, wait=0, cmdline="the other agent's suite")
    try:
        with pytest.raises(pytest.exit.Exception, match="the other agent's suite"):
            conftest.pytest_configure(config)  # type: ignore[arg-type]
    finally:
        al.release(held)

    try:
        conftest.pytest_configure(config)  # type: ignore[arg-type]
        assert os.environ[al.HELD_ENV] == str(os.getpid())
    finally:
        conftest.pytest_unconfigure(config)  # type: ignore[arg-type]
    assert al.HELD_ENV not in os.environ
    al.release(al.acquire(lock, wait=0, cmdline="free again"))
