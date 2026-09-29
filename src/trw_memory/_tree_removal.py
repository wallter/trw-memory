"""Logged tree removal for cleanup paths (PRD-FIX-156 FR01; B71-10, B71-14).

Responsibility: remove a scratch file or directory tree on a cleanup path, where
raising would mask the error being cleaned up after, without ever hiding a
failure. ``shutil.rmtree(..., ignore_errors=True)`` did the first and not the
second, and on a symlink it removed nothing at all while saying nothing.

Interface: :func:`remove_tree`. It is the one place in ``trw_memory`` and in
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

import shutil
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def remove_tree(path: Path, *, purpose: str) -> None:
    """Remove *path* for *purpose*, logging (never raising) what could not be removed."""
    failures: list[str] = []

    def collect(failed: object, exc: BaseException) -> None:
        if not isinstance(exc, FileNotFoundError):  # already gone is the goal state
            failures.append(f"{failed}: {exc}")

    try:
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        elif sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=lambda _func, failed, exc: collect(failed, exc))
        else:
            shutil.rmtree(path, onerror=lambda _func, failed, info: collect(failed, info[1]))
    except OSError as exc:
        collect(path, exc)
    if failures:
        logger.warning(
            "tree_removal_failed", path=str(path), purpose=purpose, failures=len(failures), first_error=failures[0]
        )
