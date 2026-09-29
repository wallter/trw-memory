"""PRD-QUAL-146 FR09: a test run leaves nothing in the real TMPDIR."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

_LEAKY_TEST = """
import tempfile
from pathlib import Path


def test_leaks_a_mkdtemp():
    made = Path(tempfile.mkdtemp(prefix="leaky-"))
    print("MADE", made)
    assert made.is_dir()
"""


def _run(tmp_path: Path, *plugin: str) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    real_tmp = tmp_path / "real-tmpdir"
    real_tmp.mkdir()
    basetemp = tmp_path / "basetemp"
    (tmp_path / "t").mkdir()
    test_file = tmp_path / "t" / "test_leaky.py"
    test_file.write_text(_LEAKY_TEST, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "COV_"))}
    env["TMPDIR"] = str(real_tmp)
    env["PYTHONPATH"] = os.pathsep.join([str(_PACKAGE_ROOT), env.get("PYTHONPATH", "")])
    cmd = [sys.executable, "-m", "pytest", str(test_file), "-q", "-s", "-p", "no:cacheprovider", "-p", "no:randomly"]
    cmd += ["--noconftest", f"--basetemp={basetemp}", "-o", "addopts=", *plugin]
    done = subprocess.run(cmd, cwd=tmp_path / "t", env=env, capture_output=True, text=True, timeout=120)
    return done, real_tmp, basetemp


def test_the_redirect_keeps_mkdtemp_out_of_the_real_tmpdir_and_reports_it(tmp_path: Path) -> None:
    done, real_tmp, basetemp = _run(tmp_path, "-p", "tests._session_tmpdir")
    assert done.returncode == 0, done.stdout + done.stderr
    assert list(real_tmp.iterdir()) == [], "a mkdtemp outside tmp_path reached the real TMPDIR"
    made = Path(next(line.split(" ", 1)[1] for line in done.stdout.splitlines() if line.startswith("MADE ")))
    assert made.parent == basetemp / "session-tmp"
    assert "session TMPDIR redirect: 1 temp entries left outside tmp_path" in done.stdout


def test_without_the_redirect_the_same_run_leaks_into_the_real_tmpdir(tmp_path: Path) -> None:
    """Control arm: proves the leaky test really leaks, so the arm above is not vacuous."""
    done, real_tmp, _ = _run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [p.name.startswith("leaky-") for p in real_tmp.iterdir()] == [True]


def test_the_package_conftest_loads_the_redirect_plugin() -> None:
    from tests import conftest

    assert "tests._session_tmpdir" in conftest.pytest_plugins


def test_a_test_that_sets_its_own_tmpdir_keeps_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tempfile

    own = tmp_path / "own"
    own.mkdir()
    monkeypatch.setenv("TMPDIR", str(own))
    monkeypatch.setattr(tempfile, "tempdir", None)
    assert Path(tempfile.mkdtemp()).parent == own


def _dead_pid() -> int:
    done = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True, check=True
    )
    return int(done.stdout)


def _basetemp(root: Path, n: int, *, pid: int | None, age_s: float = 7200.0) -> Path:
    path = root / f"pytest-{n}"
    (path / "test_x0").mkdir(parents=True)
    if pid is not None:
        (path / ".lock").write_text(str(pid), encoding="utf-8")
    old = time.time() - age_s
    os.utime(path, (old, old))
    return path


def test_the_sweep_removes_only_old_basetemps_whose_lock_owner_is_dead(tmp_path: Path) -> None:
    from tests._session_tmpdir import sweep_dead_basetemps

    dead = _dead_pid()
    killed_run = _basetemp(tmp_path, 1, pid=dead)
    running = _basetemp(tmp_path, 2, pid=os.getpid())
    finished = _basetemp(tmp_path, 3, pid=None)
    just_killed = _basetemp(tmp_path, 4, pid=dead, age_s=0)
    not_numbered = _basetemp(tmp_path, 5, pid=dead).rename(tmp_path / "pytest-current-ish")

    assert sweep_dead_basetemps(tmp_path, min_age_s=600) == [killed_run]
    assert not killed_run.exists()
    assert running.is_dir() and finished.is_dir() and just_killed.is_dir() and not_numbered.is_dir()


def test_the_package_keeps_every_tmp_path_until_the_run_ends() -> None:
    """``tmp_path_retention_policy = "failed"`` deleted a passing test's tmp_path while its daemon or writer
    threads still used it: 8 failures and 21 errors across unrelated trw-memory files (2026-09-28). A normal
    exit already keeps only 3 runs; the killed-run sweep above covers the rest.
    """
    import tomllib

    options = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"][
        "ini_options"
    ]
    assert options.get("tmp_path_retention_policy", "all") == "all"


def test_windows_never_probes_a_lock_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """os.kill(pid, 0) terminates the process on Windows, so there every basetemp counts as live."""
    from tests import _session_tmpdir

    def refuse(_pid: int, _sig: int) -> None:
        raise AssertionError("os.kill was called on Windows")

    monkeypatch.setattr(_session_tmpdir.sys, "platform", "win32")
    monkeypatch.setattr(_session_tmpdir.os, "kill", refuse)
    assert _session_tmpdir._pid_alive(4242) is True
