"""No test writes a store under the package checkout's ``.memory/``.

``MemoryConfig.storage_path`` defaults to ``.memory`` beside the ``.trw`` its config loaded from --
the cwd's own (before that anchor existed, the cwd itself) -- and pytest runs from the package
root, so any test that built a store without pinning a path left
``<package>/.memory/<namespace>/memory.db`` behind. The cross-project graph enumerates
``namespace_store_locations(config or MemoryConfig())``, so every later test (and every later
build of a different schema version) opened those leftovers -- one of them failed another build
with ``SchemaDowngradeError``.

The class fix is one autouse fixture: each test runs with its cwd in a fresh project directory of
its own (holding an empty ``.trw``), so the default store roots there and dies with the test. The guard is a session check: any file
created or rewritten under the package ``.memory/`` during the session fails the session, naming it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_STORE = PACKAGE_ROOT / ".memory"


def store_files(root: Path = PACKAGE_STORE) -> dict[Path, tuple[int, int]]:
    """Every file under *root* with its (mtime_ns, size) -- empty when *root* does not exist."""
    if not root.is_dir():
        return {}
    return {p: (st.st_mtime_ns, st.st_size) for p in root.rglob("*") if p.is_file() and (st := p.stat())}


def leaked_store_files(before: dict[Path, tuple[int, int]], root: Path = PACKAGE_STORE) -> list[Path]:
    """Files under *root* created or rewritten since *before* was taken, sorted.

    A rewrite counts: a test that opens a leftover store from an older run writes into it
    without creating a new path.
    """
    return sorted(p for p, signature in store_files(root).items() if before.get(p) != signature)


@pytest.fixture(autouse=True)
def _cwd_outside_the_package(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test from its own project directory (an empty ``.trw``), never the package root.

    A sibling of ``tmp_path`` (not inside it), so a default ``.memory`` never shows up in a
    test's own ``tmp_path.rglob`` assertions.
    """
    cwd = tmp_path_factory.mktemp("cwd")
    (cwd / ".trw").mkdir()
    monkeypatch.chdir(cwd)


_SNAPSHOT = pytest.StashKey[dict[Path, tuple[int, int]]]()


def snapshot_package_store(config: pytest.Config, root: Path = PACKAGE_STORE) -> None:
    """Record what the package ``.memory/`` held before any test ran (called from ``pytest_configure``)."""
    config.stash[_SNAPSHOT] = store_files(root)


def fail_session_on_leaked_store(session: pytest.Session, root: Path = PACKAGE_STORE) -> None:
    """Fail a passing session when a test left files under the package ``.memory/`` (``pytest_sessionfinish``)."""
    stash = getattr(session.config, "stash", None)  # a hand-built fake session has none: nothing to diff
    before = None if stash is None else stash.get(_SNAPSHOT, None)
    if before is None:
        return
    leaked = leaked_store_files(before, root)
    if leaked:
        shown = ", ".join(str(p.relative_to(root.parent)) for p in leaked[:5])
        print(
            f"\nFAIL: {len(leaked)} file(s) written under the package .memory/ during this session: {shown}",
            file=sys.stderr,
        )
        if session.exitstatus == 0:
            session.exitstatus = 1
