"""Logged tree removal for cleanup paths (PRD-FIX-156 FR01; B71-10, B71-14).

Responsibility: remove a scratch file or directory tree on a cleanup path, where
raising would mask the error being cleaned up after, without ever hiding a
failure. ``shutil.rmtree(..., ignore_errors=True)`` did the first and not the
second, and on a symlink it removed nothing at all while saying nothing.

Interface: :func:`remove_tree`, and :func:`remove_tree_beneath` for a path built under a root the caller does not fully control. It is the one place in ``trw_memory`` and in
``trw_mcp`` (which imports it) that may pass an error handler to
``shutil.rmtree``; each package's ``tests/test_tree_removal.py`` holds its census.

Invariants: a missing path is a no-op; a symlink or other non-directory is
unlinked, never its target; a directory goes through ``shutil.rmtree``, whose
descriptor-based walk does not follow a symlink inside the tree; every failure
is collected, the rest of the tree still goes, and ONE ``tree_removal_failed``
warning names the path, the caller's purpose, the failure count and the first
error. It never raises for a removal failure.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path, PurePath

import structlog

from trw_memory import safe_fs
from trw_memory.exceptions import UnsafeWriteError

logger = structlog.get_logger(__name__)


def _rmtree(path: str | Path, failures: list[str], *, dir_fd: int | None = None) -> None:
    """``shutil.rmtree`` collecting failures into *failures* (a missing entry is the goal state, not a failure)."""

    def collect(failed: object, exc: BaseException) -> None:
        if not isinstance(exc, FileNotFoundError):
            failures.append(f"{failed}: {exc}")

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=lambda _func, failed, exc: collect(failed, exc), dir_fd=dir_fd)
    else:
        shutil.rmtree(path, onerror=lambda _func, failed, info: collect(failed, info[1]), dir_fd=dir_fd)


def _report(path: Path, purpose: str, failures: list[str]) -> None:
    if failures:
        logger.warning(
            "tree_removal_failed", path=str(path), purpose=purpose, failures=len(failures), first_error=failures[0]
        )


def remove_tree(path: Path, *, purpose: str) -> None:
    """Remove *path* for *purpose*, logging (never raising) what could not be removed."""
    failures: list[str] = []
    try:
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            _rmtree(path, failures)
    except FileNotFoundError:  # trw-fail-silent-allow: already gone is the goal state
        pass
    except OSError as exc:
        failures.append(f"{path}: {exc}")
    _report(path, purpose, failures)


def remove_tree_beneath(root: Path, rel_path: str | PurePath, *, purpose: str) -> None:
    """Remove ``root/rel_path`` like :func:`remove_tree`, but never through a symlinked ancestor (REMOVE-TREE-ANCESTOR-SYMLINK).

    ``remove_tree`` resolves its whole path, so a directory above the leaf swapped for a link after the caller
    built the path would redirect the removal to the link's target. Here each ancestor below *root* is opened
    no-follow and the leaf is removed relative to its parent's descriptor, so a later swap cannot redirect it.
    An ancestor that is a link is a refusal, logged like any other failure; an absent one is a no-op. Where
    descriptors are not available (Windows) the ancestors are ``lstat``-checked instead -- best effort.
    """
    shown = root / rel_path
    failures: list[str] = []
    parent_fd: int | None = None
    try:
        if safe_fs.anchored_removal_supported() and shutil.rmtree.avoids_symlink_attacks:
            parent_fd, leaf = safe_fs.open_parent_beneath(root, rel_path)
            try:
                kind = stat.S_IFMT(os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False).st_mode)
            except FileNotFoundError:  # trw-fail-silent-allow: already gone is the goal state
                return
            if kind == stat.S_IFDIR:
                _rmtree(leaf, failures, dir_fd=parent_fd)
            else:
                os.unlink(leaf, dir_fd=parent_fd)
        else:
            parent, leaf = safe_fs.check_parent_beneath(root, rel_path)
            remove_tree(parent / leaf, purpose=purpose)
    except FileNotFoundError:  # trw-fail-silent-allow: an absent ancestor means nothing is there to remove
        pass
    except (OSError, UnsafeWriteError, ValueError) as exc:  # ValueError: an embedded NUL in the path
        failures.append(f"{shown}: {exc}")
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
    _report(shown, purpose, failures)
