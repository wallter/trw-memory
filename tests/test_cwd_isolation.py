"""No test leaves a store under the package's ``.memory/``.

``MemoryConfig.storage_path`` defaults to ``.memory`` beside the cwd's ``.trw``; pytest ran from the
package root, so unpinned tests left ``<package>/.memory/<ns>/memory.db`` behind for the cross-project graph
of every later test (and later build) to open. ``tests/_cwd_isolation.py`` moves each test's cwd out of
the package and fails the session when a file appears under the package ``.memory/``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests._cwd_isolation import (
    _SNAPSHOT,
    PACKAGE_ROOT,
    PACKAGE_STORE,
    fail_session_on_leaked_store,
    leaked_store_files,
    snapshot_package_store,
    store_files,
)
from trw_memory.client import MemoryClient


def test_each_test_runs_from_its_own_project_directory_outside_the_package(tmp_path: Path) -> None:
    cwd = Path.cwd().resolve()
    assert not cwd.is_relative_to(PACKAGE_ROOT)
    assert not cwd.is_relative_to(tmp_path.resolve())  # a sibling, so tmp_path.rglob never sees it
    assert list(cwd.iterdir()) == [cwd / ".trw"]  # an empty project anchor, nothing else
    assert list((cwd / ".trw").iterdir()) == []


async def test_a_default_store_lands_in_the_test_cwd_not_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MEMORY_STORAGE_PATH", "MEMORY_SINGLE_STORE_PATH"):
        monkeypatch.delenv(name, raising=False)
    before = store_files()

    client = MemoryClient(namespace="project:cwd-default", mode="local")
    try:
        stored = await client.store("a row written with the default storage path")
    finally:
        await client.close()

    assert stored["status"] == "stored"
    assert list((Path.cwd() / ".memory").rglob("memory.db")) != []
    assert leaked_store_files(before) == []


def _session(root: Path) -> Any:
    config = SimpleNamespace(stash=pytest.Stash())
    snapshot_package_store(config, root)  # type: ignore[arg-type]
    return SimpleNamespace(config=config, exitstatus=0)


def _create_new_store(root: Path) -> None:
    (root / "project_x").mkdir()
    (root / "project_x" / "memory.db").write_bytes(b"")


def _rewrite_leftover_store(root: Path) -> None:
    leftover = root / "default" / "memory.db"
    before = leftover.stat().st_mtime_ns
    leftover.write_bytes(b"an older run's store, reopened and written by this session")
    os.utime(leftover, ns=(before + 10**9, before + 10**9))  # a coarse-mtime filesystem still differs


@pytest.mark.parametrize(
    ("session_write", "expected_status", "named"),
    [(None, 0, None), (_create_new_store, 1, "project_x"), (_rewrite_leftover_store, 1, "default")],
    ids=["preexisting-files-only", "new-store-created", "leftover-store-rewritten"],
)
def test_the_session_guard_fails_only_on_a_file_written_during_the_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    session_write: Callable[[Path], None] | None,
    expected_status: int,
    named: str | None,
) -> None:
    root = tmp_path / ".memory"
    (root / "default").mkdir(parents=True)
    (root / "default" / "memory.db").write_bytes(b"left by an older run")  # not this session's leak by itself
    session = _session(root)
    if session_write is not None:
        session_write(root)

    fail_session_on_leaked_store(session, root)

    assert session.exitstatus == expected_status
    err = capsys.readouterr().err
    assert ("FAIL:" in err) is (named is not None)
    if named is not None:
        assert f".memory/{named}/memory.db" in err


def test_the_guard_never_turns_a_failed_session_green(tmp_path: Path) -> None:
    root = tmp_path / ".memory"
    session = _session(root)
    session.exitstatus = 2
    (root / "ns").mkdir(parents=True)
    (root / "ns" / "memory.db").write_bytes(b"")

    fail_session_on_leaked_store(session, root)

    assert session.exitstatus == 2


def test_this_session_snapshotted_the_real_package_store(request: pytest.FixtureRequest) -> None:
    """The conftest wires the guard: pytest_configure took the snapshot the session check diffs against."""
    assert PACKAGE_STORE == PACKAGE_ROOT / ".memory"
    assert request.config.stash[_SNAPSHOT].keys() <= store_files().keys()
