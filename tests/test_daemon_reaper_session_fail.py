"""PRD-INFRA-196-FR07: trw-memory's suite gains the same session-end daemon sweep
trw-mcp and the root scripts suite already had (this package lacked it entirely).

Both suites now run one sweep, ``trw_memory.testing.daemon_reaper.sweep_session_daemons``,
which also fails the session when a daemon survives it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from trw_memory.testing.daemon_reaper import SessionSweep

pytestmark = pytest.mark.integration


def test_pytest_sessionfinish_fails_the_session_when_the_sweep_reaped_a_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.conftest as conftest_mod

    class _FakeSession:
        exitstatus = 0

        class config:
            _tmp_path_factory = type("F", (), {"getbasetemp": staticmethod(lambda: Path("/tmp/x"))})()
            stash = pytest.Stash()

    monkeypatch.setattr(
        conftest_mod, "sweep_session_daemons", lambda basetemp, _owner: SessionSweep(basetemp, [12345], [])
    )
    monkeypatch.setattr(conftest_mod, "_timing_sessionfinish", lambda *_a, **_k: None)
    session = _FakeSession()

    conftest_mod.pytest_sessionfinish(session, 0)  # type: ignore[arg-type]

    assert session.exitstatus == 1


def test_pytest_sessionfinish_leaves_a_clean_session_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.conftest as conftest_mod

    class _FakeSession:
        exitstatus = 0

        class config:
            _tmp_path_factory = type("F", (), {"getbasetemp": staticmethod(lambda: Path("/tmp/x"))})()
            stash = pytest.Stash()

    monkeypatch.setattr(conftest_mod, "sweep_session_daemons", lambda basetemp, _owner: SessionSweep(basetemp, [], []))
    monkeypatch.setattr(conftest_mod, "_timing_sessionfinish", lambda *_a, **_k: None)
    session = _FakeSession()

    conftest_mod.pytest_sessionfinish(session, 0)  # type: ignore[arg-type]

    assert session.exitstatus == 0


def test_pytest_sessionfinish_fails_the_session_when_a_daemon_survives_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The survivor guard: nothing leaked by this sweep's count, but a daemon is still alive after it."""
    import tests.conftest as conftest_mod

    class _FakeSession:
        exitstatus = 0

        class config:
            _tmp_path_factory = type("F", (), {"getbasetemp": staticmethod(lambda: Path("/tmp/x"))})()
            stash = pytest.Stash()

    monkeypatch.setattr(
        conftest_mod, "sweep_session_daemons", lambda basetemp, _owner: SessionSweep(basetemp, [], [4242])
    )
    monkeypatch.setattr(conftest_mod, "_timing_sessionfinish", lambda *_a, **_k: None)
    session = _FakeSession()

    conftest_mod.pytest_sessionfinish(session, 0)  # type: ignore[arg-type]

    assert session.exitstatus == 1


_PLANTED_LEAK_TEST = """
import subprocess, sys, time

def test_leaks_a_daemon():
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "trw_memory.server", "serve"])
    time.sleep(0.3)
"""


def test_a_planted_leaking_test_fails_the_trw_memory_suite_at_session_end(tmp_path: Path) -> None:
    """One real end-to-end demonstration, run from inside the package so the real conftest applies."""
    package_root = Path(__file__).resolve().parents[1]
    probe = Path(__file__).resolve().parent / f"_leak_probe_{os.getpid()}.py"
    probe.write_text(_PLANTED_LEAK_TEST, encoding="utf-8")
    basetemp = tmp_path / "bt"
    home = basetemp / "home"
    env = {**os.environ, "HOME": str(home), "TRW_USER_DIR": str(home / ".trw"), "COLUMNS": "80"}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--basetemp", str(basetemp), str(probe)],
            cwd=str(package_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        probe.unlink(missing_ok=True)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "leaked memory daemon" in output, output
