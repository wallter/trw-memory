"""Daemon file locations and the hardened writes that create them.

PRD-CORE-253 FR03/NFR03, PRD-CORE-298 FR02. Files beside the user-space store:

``memory.db``            the store the daemon serves
``daemon.json``          the discovery file: pid, url, start time, version
``daemon-grants.json``   token digest -> granted namespaces
``daemon.lock``          the advisory single-instance lock (``lock_for_rmw``)

The grants file and each checkout's token are secrets. They are created with ``O_CREAT|O_EXCL`` (plus ``O_NOFOLLOW`` where
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
read revokes grants a live daemon is still authenticating against.
"""

from __future__ import annotations

import errno as _errno
import os
from dataclasses import dataclass
from pathlib import Path

import structlog

from trw_memory._dir_trust import open_or_create_at, open_verified_dir_fd, verify_and_harden_dir_fd
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
_START_LOG_NAME = "daemon-start.log"
#: Holds one ``<pid>-<token>`` directory per in-flight ``memory_import_checkout`` private
#: copy; a starting daemon removes those whose owning process is no longer live.
IMPORT_TMP_SUBDIR = "import-tmp"
_TOKEN_FILE_NAME = "daemon-token"  # noqa: S105 - a filename, not a credential
_GRANTS_FILE_NAME = "daemon-grants.json"
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
        """The retired Slice A all-namespace bearer; its presence refuses startup (PRD-CORE-298 FR02)."""
        return self.user_memory_dir / _TOKEN_FILE_NAME

    @property
    def grants(self) -> Path:
        """The 0600 map of token digest to granted namespaces (PRD-CORE-298 FR02)."""
        return self.user_memory_dir / _GRANTS_FILE_NAME

    @property
    def lock_anchor(self) -> Path:
        """The path handed to ``lock_for_rmw``; it locks ``<anchor>.lock``."""
        return self.user_memory_dir / _LOCK_ANCHOR_NAME

    @property
    def start_log(self) -> Path:
        """The 0600 stderr of the last auto-started daemon, rewritten on each start."""
        return self.user_memory_dir / _START_LOG_NAME

    @property
    def lock(self) -> Path:
        """The advisory single-instance lock file itself."""
        return self.user_memory_dir / f"{_LOCK_ANCHOR_NAME}.lock"


def _harden_dir(directory: Path) -> int:
    """Create *directory* if needed, set it to 0700, refuse a symlink or an untrusted owner.

    Opens the directory with ``O_NOFOLLOW`` (PRD-SEC-016) and hardens the
    permissions of that SAME open descriptor via ``fchmod`` -- unlike the
    ``mkdir`` + ``chmod(path)`` pair this replaces, nothing here re-resolves
    the path between the existence check and the permission change, so a
    symlink planted in that window cannot redirect either step.

    Returns the OPEN directory descriptor so callers can anchor their own
    ``dir_fd``-relative opens to it -- the same object just verified, not a
    fresh path lookup. The caller must ``os.close()`` it.
    """
    fd = open_verified_dir_fd(directory, create=True, mode=SECRET_DIR_MODE)
    try:
        # force=True: this directory holds ONLY secrets (token, grants,
        # discovery record) and is never a shared/sticky location, so its
        # mode is always pinned to SECRET_DIR_MODE, not merely tightened
        # when it happens to already be unsafe.
        verify_and_harden_dir_fd(fd, directory, target_mode=SECRET_DIR_MODE, force=True)
    except BaseException:
        os.close(fd)
        raise
    return fd


def open_private_log(path: Path) -> int:
    """Open *path* for writing at mode 0600, emptied, refusing a symlink; returns the descriptor.

    Its parent is created and hardened to 0700 first, and the file is opened
    ``dir_fd``-relative to that same verified parent -- no path re-resolution
    happens between the two.
    """
    dir_fd = _harden_dir(path.parent)
    try:
        # Two clients auto-starting at once create this log together; see open_or_create_at.
        fd = open_or_create_at(dir_fd, path.name, os.O_WRONLY | os.O_TRUNC | _NOFOLLOW, SECRET_FILE_MODE)
    finally:
        os.close(dir_fd)
    os.fchmod(fd, SECRET_FILE_MODE)
    return fd


def write_secret_file(path: Path, content: str) -> None:
    """Atomically write *content* to *path* at mode 0600.

    Writes to an exclusive temporary sibling (``O_CREAT|O_EXCL|O_NOFOLLOW``),
    both opened and renamed ``dir_fd``-relative to the same verified parent
    descriptor, and ``os.replace``s it into position -- the destination is
    never opened for write and a concurrent reader sees either the old file
    or the new one.

    Args:
        path: Destination file. Its parent is created and hardened to 0700.
        content: Text to write, encoded UTF-8.
    """
    dir_fd = _harden_dir(path.parent)
    try:
        tmp_name = f".{path.name}.{os.getpid()}.tmp"
        # A crashed predecessor can leave the temporary name behind; O_EXCL
        # would then fail forever. Removing it is safe because the name
        # embeds our pid.
        try:
            os.unlink(tmp_name, dir_fd=dir_fd)
        except FileNotFoundError:  # trw-fail-silent-allow: this IS the success case -- no stale temp file to remove; the O_EXCL open below still fails closed on a genuine race
            pass
        fd = os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, SECRET_FILE_MODE, dir_fd=dir_fd)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if os.rename in os.supports_dir_fd:
                # ``rename(2)`` on POSIX already atomically replaces an
                # existing destination -- the same guarantee ``os.replace``
                # adds over ``os.rename`` only for Windows, which does not
                # support ``dir_fd`` here in the first place.
                os.rename(tmp_name, path.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            else:
                os.replace(path.parent / tmp_name, path)  # pragma: no cover - platform fallback
        except BaseException:
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except FileNotFoundError:  # trw-fail-silent-allow: best-effort cleanup; already gone is fine, and the original exception is re-raised unconditionally below regardless
                pass
            raise
    finally:
        os.close(dir_fd)


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
