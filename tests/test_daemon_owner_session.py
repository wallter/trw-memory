"""trw-memory's test session stops every memory daemon it started, however it was placed.

The same owner tag, 60 s idle cap and survivor guard trw-mcp's suite has, through
``trw_memory.testing.daemon_reaper`` (DAEMON-ORPHAN-SPAWN, 2026-09-26: 15 orphaned
``trw_memory.server serve http`` daemons on the host at load >20).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import pytest

from trw_memory.testing.daemon_reaper import IDLE_VARIABLE, OWNER_VARIABLE

pytestmark = pytest.mark.integration

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_the_session_tags_an_owner_and_a_child_inherits_it_with_the_idle_cap(tmp_path: Path) -> None:
    """A child this session starts (as a client auto-start would) sees this session's owner and a 60 s idle cap."""
    owner = os.environ.get(OWNER_VARIABLE, "")
    assert owner.startswith(f"{os.getpid()}-"), f"this pytest process exported no owner token of its own: {owner!r}"
    probe = (
        "import os; from trw_memory.models.config import MemoryConfig; "
        f"print(os.environ.get('{OWNER_VARIABLE}', '<unset>'), MemoryConfig().memory_daemon_idle_shutdown_seconds)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=tmp_path)
    child_owner, idle = out.stdout.split()
    assert child_owner == owner
    assert idle == "60"


def test_the_idle_cap_does_not_override_a_value_the_caller_set(tmp_path: Path) -> None:
    """Boundary: the cap is a default, so a test (or operator) that sets its own value keeps it."""
    probe = (
        "from trw_memory.testing.daemon_reaper import tag_daemon_ownership; import os; "
        f"tag_daemon_ownership(); print(os.environ['{IDLE_VARIABLE}'])"
    )
    env = {**os.environ, IDLE_VARIABLE: "900"}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == "900"


def test_a_session_that_returns_with_a_detached_daemon_running_leaves_none_alive(tmp_path: Path) -> None:
    """End to end: an inner session starts a detached daemon outside its basetemp and returns; none survives."""
    run = uuid.uuid4().hex[:12]
    pid_file = tmp_path / f"daemon-{run}.pid"
    elsewhere = tmp_path / f"elsewhere-{run}"
    elsewhere.mkdir()
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "test_leaks.py").write_text(
        textwrap.dedent(
            f"""
            import os, subprocess, sys
            from pathlib import Path

            def test_starts_a_daemon_and_returns():
                env = {{**os.environ, "HOME": {str(elsewhere)!r}, "TRW_USER_DIR": {str(elsewhere)!r}}}
                daemon = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(120)", "trw_memory.server", "serve", "http"],
                    env=env, cwd={str(elsewhere)!r}, start_new_session=True,
                )
                Path({str(pid_file)!r}).write_text(str(daemon.pid))
            """
        ),
        encoding="utf-8",
    )
    env = {k: v for k, v in os.environ.items() if k not in (OWNER_VARIABLE, "PYTEST_XDIST_WORKER", "PYTEST_ADDOPTS")}
    result = subprocess.run(
        [
            *[sys.executable, "-m", "pytest", str(inner), "-p", "tests.conftest", "-q"],
            *["-p", "no:cacheprovider", "-p", "no:randomly", f"--rootdir={inner}"],
            f"--basetemp={tmp_path / 'inner-basetemp'}",
        ],
        capture_output=True,
        text=True,
        cwd=_PACKAGE_ROOT,
        env=env,
        check=False,
        timeout=180,
    )
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.1)
    try:
        assert not _alive(pid), f"the inner session left daemon {pid} running:\n{result.stdout}\n{result.stderr}"
        assert "leaked memory daemon" in result.stderr, "the leak still fails the session that caused it"
        assert result.returncode == 1, result.stdout + result.stderr
    finally:
        if _alive(pid):
            os.kill(pid, 9)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:  # trw-fail-silent-allow: no such process means not alive
        return False
    return True
