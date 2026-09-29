"""Session TMPDIR redirect (PRD-QUAL-146 FR09, backlog B80-62, L-d6WS).

Loaded as a pytest plugin from the package conftest's ``pytest_plugins``. For the
whole session every ``tempfile.mkdtemp`` / ``TemporaryDirectory`` / ``NamedTemporaryFile``
made OUTSIDE ``tmp_path`` -- by a test, by the code under test, or by a child
process that inherits ``TMPDIR`` -- lands in a directory inside pytest's basetemp,
so pytest's own retention prunes it instead of the real ``TMPDIR`` accumulating
thousands of dirs per swarm day. Each xdist worker has its own basetemp, so each
worker redirects its own process. The terminal summary reports the entries left
behind, so the leakers stay visible.

pytest keeps a numbered ``pytest-N`` basetemp that still holds its ``.lock`` for
three days (``LOCK_TIMEOUT``), and only an ``atexit`` handler removes that lock --
so every run killed by a harness timeout or a signal left its whole basetemp on
disk for three days (52 such dirs, 31 GB, on 2026-09-28). At session start the
controller removes sibling basetemps whose lock names a dead process.

A test that sets ``TMPDIR`` itself (``monkeypatch.setenv``) still wins: this
fixture runs first, and monkeypatch restores to the redirected value afterwards.

Duplicated verbatim in trw-mcp/tests and trw-memory/tests (separately published
mirrors; neither test tree may import the other's).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

REDIRECT_NAME = "session-tmp"
#: Tool caches that default to ``<TMPDIR>/<name>_<user>`` on first use (torch inductor, once any test imports
#: torch) and so would show up as leaked entries; point them at a sibling of the redirect dir instead.
CACHE_DIR_ENV = {"TORCHINDUCTOR_CACHE_DIR": "torchinductor-cache", "TRITON_CACHE_DIR": "triton-cache"}
#: A dead-lock basetemp younger than this is left alone (its orphaned workers may still be exiting).
DEAD_BASETEMP_MIN_AGE_S = 3600.0


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":  # os.kill(pid, 0) TERMINATES the process on Windows; keep every basetemp there
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:  # trw-fail-silent-allow: no such process IS the answer (the lock owner is dead)
        return False
    except PermissionError:
        return True
    return True


def sweep_dead_basetemps(root: Path, *, min_age_s: float = DEAD_BASETEMP_MIN_AGE_S) -> list[Path]:
    """Remove ``pytest-N`` dirs under *root* whose ``.lock`` names a process that no longer exists.

    A lock naming a live pid (a run in progress, or a reused pid) keeps its dir, as does an
    unlocked dir (pytest's own keep-3 rotation owns those) and any dir touched within *min_age_s*.
    """
    removed: list[Path] = []
    now = time.time()
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        lock = path / ".lock"
        if not re.fullmatch(r"pytest-\d+", path.name) or path.is_symlink() or not lock.is_file():
            continue
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
            age = now - path.stat().st_mtime
        except (OSError, ValueError):  # trw-fail-silent-allow: an unreadable lock keeps its dir (never delete on doubt)
            continue
        if pid <= 0 or age < min_age_s or _pid_alive(pid):
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def pytest_sessionstart(session: pytest.Session) -> None:
    config = session.config
    factory = getattr(config, "_tmp_path_factory", None)
    if factory is None or hasattr(config, "workerinput") or config.option.basetemp:
        return
    sweep_dead_basetemps(factory.getbasetemp().parent)


@pytest.fixture(scope="session", autouse=True)
def _session_tmpdir_redirect(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    target = tmp_path_factory.mktemp(REDIRECT_NAME, numbered=False)
    saved_env, saved_cached = os.environ.get("TMPDIR"), tempfile.tempdir
    os.environ["TMPDIR"] = str(target)
    tempfile.tempdir = str(target)
    saved_caches = {name: os.environ.get(name) for name in CACHE_DIR_ENV}
    for name, sub in CACHE_DIR_ENV.items():
        if saved_caches[name] is None:  # an operator-set cache dir still wins
            os.environ[name] = str(target.parent / sub)
    try:
        yield target
    finally:
        for name, value in saved_caches.items():
            if value is None:
                os.environ.pop(name, None)
        tempfile.tempdir = saved_cached
        if saved_env is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = saved_env


def leftover_entries(basetemp: Path) -> int:
    """Entries left in every redirect dir under *basetemp* (its own, and each xdist worker's)."""
    dirs = [basetemp / REDIRECT_NAME, *basetemp.glob(f"popen-gw*/{REDIRECT_NAME}")]
    return sum(len(list(d.iterdir())) for d in dirs if d.is_dir())


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, config: pytest.Config) -> None:
    factory = getattr(config, "_tmp_path_factory", None)
    if factory is None:
        return
    count = leftover_entries(factory.getbasetemp())
    terminalreporter.write_line(
        f"session TMPDIR redirect: {count} temp entries left outside tmp_path (PRD-QUAL-146 FR09)"
    )
