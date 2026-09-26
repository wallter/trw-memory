"""A file's identity, held so a swap of the name cannot reuse it (PRD-SEC-016).

Identity checks compare ``(st_dev, st_ino)`` before and after an open that
must go by path (SQLite's own ``connect``). That comparison is only sound if
the "before" inode cannot be freed and its number handed to a replacement in
between -- and Linux filesystems (ext4, xfs, tmpfs, overlayfs) do exactly
that: a file unlinked and recreated at once usually gets the SAME inode
number back, so a plain before/after ``stat`` passes a swapped file.

:func:`pinned_identity` closes that on Linux by holding an ``O_PATH``
descriptor on the entry for the duration of the check: an inode with an open
descriptor stays allocated, so no replacement can receive its number. Closing
an ``O_PATH`` descriptor releases no POSIX locks (the kernel skips
``locks_remove_posix`` for ``FMODE_PATH`` files), so pinning a store another
connection holds cannot drop that connection's locks (C15, see
:mod:`trw_memory._live_stores`).

Elsewhere no descriptor is taken -- on macOS closing ANY descriptor on a file
drops the process's locks on it -- and the identity is a plain ``lstat``.
APFS and HFS+ allocate object ids from an increasing counter, so the plain
check is sound there; a platform that reuses inode numbers without
``O_PATH`` (e.g. FreeBSD UFS) keeps that residual.
"""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

Identity = tuple[int, int]

#: ``O_PATH`` where its close is known to leave POSIX locks alone (Linux only).
_O_PATH: int | None = getattr(os, "O_PATH", None) if sys.platform.startswith("linux") else None
_PIN_FLAGS = (_O_PATH or 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


def _regular_identity(st: os.stat_result) -> Identity | None:
    return (st.st_dev, st.st_ino) if stat.S_ISREG(st.st_mode) else None


def current_identity(path: Path | str, *, dir_fd: int | None = None) -> Identity | None:
    """The regular file at *path* right now (not following a symlink), or ``None``."""
    try:
        return _regular_identity(os.stat(path, dir_fd=dir_fd, follow_symlinks=False))
    except OSError:  # trw-fail-silent-allow: no identity; every caller treats None as "unknown", never as a match
        return None


@contextmanager
def pinned_identity(path: Path | str, *, dir_fd: int | None = None) -> Iterator[Identity | None]:
    """Yield *path*'s identity, held for the ``with`` block so its inode number cannot be reused.

    ``None`` when *path* is absent or not a regular file (a symlink included).
    Compare it with :func:`current_identity` after the path-based open.
    """
    if _O_PATH is None:
        yield current_identity(path, dir_fd=dir_fd)
        return
    try:
        fd = os.open(path, _PIN_FLAGS, dir_fd=dir_fd)
    except OSError:  # trw-fail-silent-allow: an absent or unopenable entry has no identity to pin; None never matches
        fd = -1
    try:
        yield _regular_identity(os.fstat(fd)) if fd != -1 else None
    finally:
        if fd != -1:
            os.close(fd)
