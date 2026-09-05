"""Daemon file locations and the hardened writes that create them.

PRD-CORE-253 FR03/NFR03. Four files live beside the user-space store:

``memory.db``      the store the daemon serves
``daemon.json``    the discovery file: pid, url, token, start time, version
``daemon-token``   the bearer token every request must carry
``daemon.lock``    the advisory single-instance lock (``lock_for_rmw``)

The token and the discovery file both carry the bearer token, so both are
secrets. They are created with ``O_CREAT|O_EXCL`` (plus ``O_NOFOLLOW`` where
the platform has it) at mode 0600 into a private temporary name, then moved
into place with an atomic ``os.replace``. Two properties follow that a plain
``write_text`` does not give:

* a local attacker cannot pre-plant a symlink at the destination and have the
  daemon write a secret through it -- the write never touches the destination
  name, and ``os.replace`` replaces a symlink rather than following it; and
* a reader never observes a half-written file, so a client cannot parse a
  discovery record that names a port the daemon has not bound yet.

Reads of those files use ``O_NOFOLLOW`` for the same reason, and a read that is
refused RAISES rather than answering ``None``. ``None`` is reserved for "the
file does not exist", because that is the answer every caller responds to by
creating the file -- and creating a token over one that merely could not be
read rotates the secret a live daemon is still authenticating against.
"""

from __future__ import annotations

import errno as _errno
import os
from dataclasses import dataclass
from pathlib import Path

import structlog

from trw_memory.exceptions import DaemonSecretUnreadableError
from trw_memory.user_paths import resolve_user_memory_dir

__all__ = [
    "SECRET_DIR_MODE",
    "SECRET_FILE_MODE",
    "DaemonPaths",
    "read_failure_reason",
    "read_secret_file",
    "write_secret_file",
]

logger = structlog.get_logger(__name__)

#: Mode for the directory holding the token and discovery file.
SECRET_DIR_MODE = 0o700
#: Mode for the token and discovery files themselves.
SECRET_FILE_MODE = 0o600

#: ``O_NOFOLLOW`` is POSIX-only; on Windows the flag does not exist and the
#: symlink-planting attack it blocks needs privileges the threat model already
#: excludes. Resolved once so the open paths below stay branch-free.
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

#: ``errno`` values the refusal message translates into operator language.
#: ``ELOOP`` is what ``O_NOFOLLOW`` returns for a symlink -- the attack this
#: module exists to block -- so naming it is the difference between "it did not
#: work" and "someone planted a link at your token path".
_ELOOP = _errno.ELOOP
_EACCES = _errno.EACCES
_EPERM = _errno.EPERM
_EISDIR = _errno.EISDIR

_STORE_FILE_NAME = "memory.db"
_DISCOVERY_FILE_NAME = "daemon.json"
_TOKEN_FILE_NAME = "daemon-token"  # noqa: S105 - a filename, not a credential
#: ``lock_for_rmw(path)`` locks ``<path>.lock``, so the anchor is the stem.
_LOCK_ANCHOR_NAME = "daemon"


@dataclass(frozen=True)
class DaemonPaths:
    """Every path the loopback daemon owns, derived from one directory."""

    user_memory_dir: Path

    @classmethod
    def resolve(cls, *, create: bool = True) -> DaemonPaths:
        """Resolve from the machine-local user memory directory (FR01)."""
        return cls(user_memory_dir=resolve_user_memory_dir(create=create))

    @property
    def store(self) -> Path:
        """The single user-space SQLite store the daemon serves."""
        return self.user_memory_dir / _STORE_FILE_NAME

    @property
    def discovery(self) -> Path:
        """The 0600 discovery file clients read to find the daemon."""
        return self.user_memory_dir / _DISCOVERY_FILE_NAME

    @property
    def token(self) -> Path:
        """The 0600 bearer-token file."""
        return self.user_memory_dir / _TOKEN_FILE_NAME

    @property
    def lock_anchor(self) -> Path:
        """The path handed to ``lock_for_rmw``; it locks ``<anchor>.lock``."""
        return self.user_memory_dir / _LOCK_ANCHOR_NAME

    @property
    def lock(self) -> Path:
        """The advisory single-instance lock file itself."""
        return self.user_memory_dir / f"{_LOCK_ANCHOR_NAME}.lock"


def _harden_dir(directory: Path) -> None:
    """Create *directory* if needed and set it to 0700, best-effort."""
    directory.mkdir(parents=True, exist_ok=True)
    try:
        directory.chmod(SECRET_DIR_MODE)
    except OSError as exc:  # pragma: no cover - platform-dependent
        logger.warning("daemon_dir_chmod_failed", path=str(directory), error=type(exc).__name__)


def write_secret_file(path: Path, content: str) -> None:
    """Atomically write *content* to *path* at mode 0600.

    Writes to an exclusive temporary sibling (``O_CREAT|O_EXCL|O_NOFOLLOW``)
    and ``os.replace``s it into position, so the destination is never opened
    for write and a concurrent reader sees either the old file or the new one.

    Args:
        path: Destination file. Its parent is created and hardened to 0700.
        content: Text to write, encoded UTF-8.
    """
    _harden_dir(path.parent)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    # A crashed predecessor can leave the temporary name behind; O_EXCL would
    # then fail forever. Removing it is safe because the name embeds our pid.
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, SECRET_FILE_MODE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_failure_reason(exc: BaseException) -> str:
    """Explain a refused secret read in the operator's terms."""
    if isinstance(exc, UnicodeDecodeError):
        return "the file is not valid UTF-8"
    errno = getattr(exc, "errno", None)
    if errno == _ELOOP:
        return "the path is a symlink, and a secret is never read through one"
    if errno in (_EACCES, _EPERM):
        return "permission denied"
    if errno == _EISDIR:
        return "the path is a directory, not a file"
    return f"{type(exc).__name__}: {exc}"


def read_secret_file(path: Path) -> str | None:
    """Read *path* without following a symlink; ``None`` only when it is ABSENT.

    ``None`` means one thing and one thing only: the file does not exist. Every
    other outcome raises, because callers answer "absent" by CREATING the file
    -- generating a token, starting a daemon -- and a read that merely failed
    is not evidence the file is not there. Collapsing the two is how a planted
    symlink or a permission change turns into a rotated token that locks a live
    daemon's clients out.

    Args:
        path: The 0600 secret to read.

    Returns:
        The file's UTF-8 text, or ``None`` when the file does not exist.

    Raises:
        DaemonSecretUnreadableError: The file exists but could not be read --
            a symlink at the path, a permission failure, or non-UTF-8 bytes.
    """
    try:
        fd = os.open(path, os.O_RDONLY | _NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _unreadable(path, exc) from exc
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise _unreadable(path, exc) from exc


def _unreadable(path: Path, exc: BaseException) -> DaemonSecretUnreadableError:
    """Build the refusal, logging it once at the point it is decided."""
    reason = read_failure_reason(exc)
    logger.warning("daemon_secret_read_refused", path=str(path), error=type(exc).__name__)
    return DaemonSecretUnreadableError(f"{path} exists but could not be read: {reason}")
