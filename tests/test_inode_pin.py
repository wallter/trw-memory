"""PRD-SEC-016 on Linux: a pinned identity survives the replacement getting the freed inode back.

Linux filesystems hand a freed inode number straight to the next file created,
so an unpinned before/after ``(st_dev, st_ino)`` comparison passes a file that
was unlinked and recreated in between. These tests fail on Linux without the pin.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

from tests.test_permission_lock_safety import _other_process_can_write
from trw_memory._inode_pin import current_identity, pinned_identity

_POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX inode semantics")


@_POSIX_ONLY
def test_a_file_recreated_inside_the_pin_never_shares_its_identity(tmp_path: Path) -> None:
    target = tmp_path / "store.db"
    for attempt in range(20):  # without the pin, Linux reuses the number on the first try
        target.write_bytes(b"original")
        with pinned_identity(target) as before:
            target.unlink()
            target.write_bytes(b"swapped in")
            assert before is not None
            assert current_identity(target) != before, f"attempt {attempt}: the replacement reused the inode"
        target.unlink()


def test_an_absent_entry_a_directory_and_a_symlink_have_no_identity(tmp_path: Path) -> None:
    real = tmp_path / "real.db"
    real.write_bytes(b"x")
    link = tmp_path / "link.db"
    link.symlink_to(real)
    for path in (tmp_path / "missing.db", tmp_path, link):
        assert current_identity(path) is None
        with pinned_identity(path) as pinned:
            assert pinned is None
    with pinned_identity(real) as pinned:
        assert pinned == current_identity(real) is not None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the O_PATH pin descriptor is Linux-only")
def test_releasing_a_pin_on_a_locked_store_keeps_the_lock(tmp_path: Path) -> None:
    """The pin is an O_PATH descriptor: closing it must not drop this process's SQLite locks (C15)."""
    db = tmp_path / "memory.db"
    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("CREATE TABLE t(x)")
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pinned_identity(db) as pinned:
            assert pinned is not None
        assert _other_process_can_write(db) is False
    finally:
        holder.execute("ROLLBACK")
        holder.close()
