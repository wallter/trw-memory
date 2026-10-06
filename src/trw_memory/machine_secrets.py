"""Owner-only machine secret files under ``~/.trw/``: the one reader and the one writer.

Every machine-level secret TRW keeps (``~/.trw/jev.env`` for the trw_assess key,
``~/.trw/credentials.yaml`` for the platform key) goes through this module, so the safety rule is
written once and both packages import it (trw-memory is the layer trw-mcp may import).

**Read** (:func:`read_private_file`): only a regular file, not a symlink, owned by the current
user, with no group or other permission bits (0600 or tighter). The checks run on the OPEN
descriptor (``O_NOFOLLOW | O_NONBLOCK`` then ``fstat``), so a swap between check and read cannot
slip a different file in, and the read is bounded. Anything else is refused: the result carries
``problem`` (a short phrase that names the fix, for ``doctor`` and status views), a warning is
logged, and the parser never sees the bytes. A missing file is not a problem.

**Write** (:func:`write_private_file`): atomic replace, created at 0600 from the start (never
write-then-chmod), refusing a symlinked ``~/.trw`` or leaf (:func:`trw_memory.safe_fs.write_beneath`).
A missing ``~/.trw`` is created 0700.

On Windows the POSIX owner and mode bits mean nothing, so only the regular-file and symlink checks
apply there; the user-profile ACL is the guard.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

import structlog

__all__ = ["PrivateRead", "machine_secret_path", "read_private_file", "write_private_file"]

logger = structlog.get_logger(__name__)

T = TypeVar("T")

#: A machine secret file is a few short lines; anything bigger is not one.
_MAX_BYTES = 64 * 1024
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_SYMLINK_ERRNOS = frozenset({errno.ELOOP, getattr(errno, "EMLINK", errno.ELOOP)})


def machine_secret_path(name: str) -> Path:
    """``~/.trw/<name>``, resolved at call time so a changed ``HOME`` (or a test) is honoured."""
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"not a plain file name: {name!r}")
    return Path.home() / ".trw" / name


@dataclass(frozen=True)
class PrivateRead(Generic[T]):
    """``value`` is the parser's result (``None`` when absent or refused); ``problem`` says why it was refused."""

    value: T | None = field(repr=False)
    present: bool
    problem: str = ""


def _refusal(st: os.stat_result, shown: str) -> str:
    if not stat.S_ISREG(st.st_mode):
        return "is not a regular file"
    if os.name == "nt":
        return ""
    if st.st_uid != os.geteuid():
        return "is not owned by the current user"
    if st.st_mode & 0o077:
        return f"has permissions {stat.S_IMODE(st.st_mode):04o}, looser than 0600 (fix: chmod 600 {shown})"
    return ""


def _refused(shown: str, problem: str) -> PrivateRead[T]:
    logger.warning("machine_secret_refused", file=shown, reason=problem)
    return PrivateRead(None, present=True, problem=problem)


def read_private_file(path: Path, parser: Callable[[str], T], *, shown: str | None = None) -> PrivateRead[T]:
    """Parse ``path`` with ``parser`` only if it is a private regular file the current user owns.

    ``shown`` is how the file is named in ``problem`` and the log (default: the path itself; callers
    pass a ``~/.trw/...`` label so no expanded home path reaches a report). Never raises for a file
    problem; a ``parser`` exception propagates (that is a caller bug, not a file condition).
    """
    label = str(path) if shown is None else shown
    try:
        fd = os.open(os.fspath(path), os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR):
            return PrivateRead(None, present=False)
        if exc.errno in _SYMLINK_ERRNOS or path.is_symlink():
            return _refused(label, "is a symlink")
        return _refused(label, f"could not be opened ({type(exc).__name__})")
    try:
        problem = _refusal(os.fstat(fd), label)
        if problem:
            return _refused(label, problem)
        raw = os.read(fd, _MAX_BYTES + 1)
    except OSError as exc:
        return _refused(label, f"could not be read ({type(exc).__name__})")
    finally:
        os.close(fd)
    if len(raw) > _MAX_BYTES:
        return _refused(label, "is too large to be a secret file")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _refused(label, "is not UTF-8")
    return PrivateRead(parser(text), present=True)


def write_private_file(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text``, created 0600 (exactly); ``path.parent`` is made 0700 if absent.

    Raises :class:`trw_memory.exceptions.UnsafeWriteError` when the parent or the leaf is a symlink
    or the parent is not a directory the current user can trust.
    """
    from trw_memory._dir_trust import make_private_dirs
    from trw_memory.safe_fs import write_beneath

    if not path.parent.exists():
        make_private_dirs(path.parent, 0o700)
    write_beneath(path.parent, path.name, text.encode("utf-8"), mode=0o600, exact_mode=True)
