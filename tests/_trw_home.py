"""Shared HOME/XDG/TRW_USER_DIR test-isolation floor (2026-09-05 testing-strategy review).

CANONICAL COPY: ``trw-memory/tests/_trw_home.py``. ``trw-mcp``, ``trw-eval``,
and ``trw-autoresearch`` each carry a byte-identical copy of this exact file --
``scripts/tests/test_conftest_trw_home_parity.py`` asserts all four stay
identical. There is no shared cross-package test dependency between these
independently-distributed packages (the same reasoning the xdist fan-out
guard duplication in these same four ``conftest.py`` files already documents
under "xdist fan-out cap"): edit one copy, edit all four, then rerun that
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
test in any of the four packages asserts the literal redirected path, and
none of the four currently depends on the real operator HOME being reachable
(if they did, they would already be polluting it on every run).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def isolated_trw_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Redirect HOME, XDG_DATA_HOME, and TRW_USER_DIR to isolated tmp dirs.

    ``monkeypatch.setenv`` restores the prior value (or absence) automatically
    at fixture teardown -- no manual save/restore is needed here. Runs
    alongside each package's own isolation fixtures, not instead of them; see
    the module docstring for why a forced ``chdir``/``.trw`` pre-create is not
    part of this shared floor.
    """
    # Outside ``tmp_path``: tests that enumerate their own tmp tree (the FIX-128
    # traversal checks list ``tmp_path`` and expect only what they created)
    # must not see the fixture's home directory beside their project.
    # pytest owns the directory (no rmtree call of our own at teardown: installer
    # tests monkeypatch shutil.rmtree with a one-argument stand-in), and it sits
    # beside — never inside — the test's own tmp_path.
    home_dir = tmp_path_factory.mktemp("trw-home")
    home_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("XDG_DATA_HOME", str(home_dir / ".local" / "share"))
    monkeypatch.setenv("TRW_USER_DIR", str(home_dir / ".trw-user"))
    yield
