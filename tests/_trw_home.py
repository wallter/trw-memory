"""Shared HOME/XDG/TRW_USER_DIR test-isolation floor (2026-09-05 testing-strategy review).

CANONICAL COPY: ``trw-memory/tests/_trw_home.py``. ``trw-mcp``, ``trw-eval``,
``trw-autoresearch``, ``trw-distill`` and the repo-root suites (``scripts/tests/``,
which also serves ``tests/``) each carry a byte-identical copy of this exact file --
``scripts/tests/test_conftest_trw_home_parity.py`` asserts all six stay
identical. There is no shared cross-package test dependency between these
independently-distributed packages (the same reasoning the xdist fan-out
guard duplication in these same four ``conftest.py`` files already documents
under "xdist fan-out cap"): edit one copy, edit all six, then rerun that
parity test.

Every package ALSO keeps its own, more specific isolation fixture(s) --
trw-mcp's ``TRW_USER_DIR``/``HOME`` pair plus the ``_path_isolation`` sweep for
``resolve_trw_dir()``, trw-memory's ``TRW_DIR`` SEC-001 anchor, trw-eval's
``EVAL_*``/ledger path redirection (plus its own ``chdir``), and
trw-autoresearch's ``AUTORESEARCH_STATE_ROOT``. This fixture does not replace
or reorder ANY of those -- it is an ADDITIVE floor underneath them: whatever a
package's own resolution chain does, if any of it falls through to
``Path.home()`` or an unset ``TRW_USER_DIR``, this still lands in an isolated
tmp directory instead of the operator's real ``~``/``~/.trw``.

Deliberately does NOT ``chdir`` or pre-create a ``.trw`` directory, even
though that was the original ask -- two concrete, already-observed
collisions rule both out:

1. trw-mcp's own ``tmp_project`` fixture does
   ``(tmp_path / ".trw").mkdir()`` with no ``exist_ok`` on the very
   ``tmp_path`` a shared fixture would target. Pre-creating ``.trw`` here
   would turn every ``tmp_project``-using test into a collection-time
   ``FileExistsError``.
2. trw-autoresearch's ``tests/test_calibration.py`` resolves a MODULE-LEVEL
   relative path (a repository-relative documentation path)
   against whatever the cwd happens to be when the test body runs. An
   autouse ``chdir`` away from the package root would silently break that
   read (and any other cwd-relative read this sweep did not find).

Redirecting HOME/XDG_DATA_HOME/TRW_USER_DIR carries no equivalent risk: no
test in any of the six suites asserts the literal redirected path, and
none of the six currently depends on the real operator HOME being reachable
(if they did, they would already be polluting it on every run).
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

#: The real home, captured at import: before any fixture redirects HOME, whatever the fixture order.
_REAL_HOME = Path.home()

#: The one home-scoped file TRW's init/update/uninstall writes (Antigravity CLI's global MCP config). A test run
#: touching it is a hard failure. Watched by (mtime_ns, size) only; the tripwire never opens it.
_WATCHED_HARD = (".gemini/config/mcp_config.json",)
#: Rewritten by every live Claude Code session, and never by TRW: a change here proves nothing about the run, so
#: it is reported once as a warning and never fails a test.
_WATCHED_SOFT = (".claude.json",)

_Stat = tuple[int, int] | None
_baseline: dict[str, _Stat] = {}
_warned_soft: set[str] = set()


def snapshot_real_config(home: Path, rels: tuple[str, ...]) -> dict[str, _Stat]:
    """``(mtime_ns, size)`` of each file in *rels* under *home*, ``None`` where it does not exist."""
    stats: dict[str, _Stat] = {}
    for rel in rels:
        try:
            st = (home / rel).stat()
        except FileNotFoundError:
            stats[rel] = None
        else:
            stats[rel] = (st.st_mtime_ns, st.st_size)
    return stats


def changed_real_config(home: Path, before: dict[str, _Stat]) -> list[str]:
    """The files in *before* whose stat now differs (created, deleted, rewritten or resized)."""
    now = snapshot_real_config(home, tuple(before))
    return [rel for rel, was in before.items() if now[rel] != was]


@functools.cache
def _real_uv_cache() -> str | None:
    """The real uv cache dir, resolved once before any HOME redirect; ``None`` when uv is unavailable."""
    try:
        done = subprocess.run(
            ["uv", "cache", "dir"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    # trw-fail-silent-allow: no uv means nothing to pin
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


@pytest.fixture(autouse=True)
def real_config_tripwire(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail the test that was running when the operator's real global MCP config changed.

    ``isolated_trw_home`` below is the floor; this is the alarm for anything that slips past it (a
    subprocess that resolves HOME another way, a test that pins the real HOME back). Checked at every
    test's teardown (two stats), so a hit names the test. Under xdist another worker's test may be the
    writer: rerun with ``-n 0`` to bisect. The baseline moves to the new stat after a hit, so one
    write fails one test rather than every later one.
    """
    if not _baseline:
        _baseline.update(snapshot_real_config(_REAL_HOME, _WATCHED_HARD + _WATCHED_SOFT))
    yield
    hard = {rel: _baseline[rel] for rel in _WATCHED_HARD}
    if changed := changed_real_config(_REAL_HOME, hard):
        _baseline.update(snapshot_real_config(_REAL_HOME, tuple(changed)))
        pytest.fail(
            f"{request.node.nodeid} ran while the operator's real config changed: {', '.join(changed)} under "
            f"{_REAL_HOME}. Some test or CLI it spawned inherited the real HOME (a subprocess env built without "
            "HOME falls back to the passwd home); give it an isolated HOME. Under xdist, rerun with -n 0.",
            pytrace=False,
        )
    soft = {rel: _baseline[rel] for rel in _WATCHED_SOFT}
    for rel in changed_real_config(_REAL_HOME, soft):
        _baseline.update(snapshot_real_config(_REAL_HOME, (rel,)))
        if rel not in _warned_soft:
            _warned_soft.add(rel)
            # Warn-only means warn-only: a developer's ``-W error`` must not turn it into an error.
            with warnings.catch_warnings():
                warnings.simplefilter("default")
                warnings.warn(
                    f"{rel} under the real home changed during this run (live Claude Code sessions do this)",
                    stacklevel=1,
                )


def _redirect_home(mp: pytest.MonkeyPatch, home_dir: Path) -> None:
    """Point HOME, XDG_CONFIG_HOME, XDG_DATA_HOME and TRW_USER_DIR at *home_dir* (via *mp*, so undone by it).

    The HF model cache and the uv cache must survive the redirect: the embedding fixture is provisioned
    into the real ~/.cache/huggingface, and an offline CI replay (HF_HUB_OFFLINE=1) cannot re-download it
    into the temp home; a test that runs ``uv`` under the redirected HOME would otherwise rebuild every
    wheel into an empty cache. Both are pinned to the real locations (``_REAL_HOME``) unless already set, and the
    HF cache is read-only in effect: HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE default to 1, so a test that lacks
    the weights fails (or skips) instead of downloading. XDG_CACHE_HOME goes to *home_dir*; SENTENCE_TRANSFORMERS_HOME is deliberately not redirected (it would hide the provisioned weights).
    """
    if not os.environ.get("HF_HOME"):
        mp.setenv("HF_HOME", str(_REAL_HOME / ".cache" / "huggingface"))
    # Read the provisioned cache, never write to it: offline switches (an explicit value, even "0", is kept) so
    # no test can download into the operator's real model cache. SENTENCE_TRANSFORMERS_HOME is left alone on
    # purpose: when set it REPLACES the HF cache as the model folder, so pointing it at the temp home hides the
    # provisioned weights (test_prd_frontier_001_gar failed that way, 2026-09-30).
    for offline in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if not os.environ.get(offline):
            mp.setenv(offline, "1")
    mp.setenv("XDG_CACHE_HOME", str(home_dir / ".cache"))
    if not os.environ.get("UV_CACHE_DIR") and (uv_cache := _real_uv_cache()):
        mp.setenv("UV_CACHE_DIR", uv_cache)
    home_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        # Every macOS home has ~/.Trash; uninstall moves TRW's own unchanged captures there.
        (home_dir / ".Trash").mkdir(exist_ok=True)
    mp.setenv("HOME", str(home_dir))
    mp.setenv("XDG_CONFIG_HOME", str(home_dir / ".config"))
    mp.setenv("XDG_DATA_HOME", str(home_dir / ".local" / "share"))
    mp.setenv("TRW_USER_DIR", str(home_dir / ".trw-user"))


@pytest.fixture(scope="session", autouse=True)
def session_trw_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """The floor under every module-, class- and session-scoped fixture.

    pytest sets up higher-scoped fixtures first, so a module-scoped fixture (one that runs init or update
    once for a whole file) runs BEFORE the function-scoped :func:`isolated_trw_home` below: without this it
    sees the process HOME, which is the operator's real home in a normal run. That is how the real
    ``~/.gemini/config/mcp_config.json`` got written (2026-09-29). Session-scoped and autouse, it redirects
    HOME before anything else in the session runs; the per-test floor still layers a fresh home on top.
    """
    mp = pytest.MonkeyPatch()
    try:
        _redirect_home(mp, tmp_path_factory.mktemp("trw-session-home"))
        yield
    finally:
        mp.undo()


@pytest.fixture(autouse=True)
def isolated_trw_home(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Redirect HOME, XDG_CONFIG_HOME, XDG_DATA_HOME, and TRW_USER_DIR to isolated tmp dirs.

    ``monkeypatch.setenv`` restores the prior value (or absence) automatically
    at fixture teardown -- no manual save/restore is needed here. Runs
    alongside each package's own isolation fixtures, not instead of them; see
    the module docstring for why a forced ``chdir``/``.trw`` pre-create is not
    part of this shared floor. Layered on :func:`session_trw_home`.
    """
    # Outside ``tmp_path``: tests that enumerate their own tmp tree (the FIX-128
    # traversal checks list ``tmp_path`` and expect only what they created)
    # must not see the fixture's home directory beside their project.
    # pytest owns the directory (no rmtree call of our own at teardown: installer
    # tests monkeypatch shutil.rmtree with a one-argument stand-in), and it sits
    # beside — never inside — the test's own tmp_path.
    _redirect_home(monkeypatch, tmp_path_factory.mktemp("trw-home"))
    yield
