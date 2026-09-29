"""The drain key: the daemon-management secret ``memory_drain`` requires (review r1 P1).

A checkout's namespace grant authenticates a tool call, and the version header is
the caller's own claim, so neither may authorise stopping a daemon that serves
every checkout. At start the daemon writes a random key to ``<dir>/drain.key``
(0600, atomically, never through a symlink) and keeps a copy in memory; a drain
must present it. Reading the file takes the same OS user whose processes may
signal the daemon anyway, so the key adds no power a same-user process lacks,
and gives none to a process that only holds a checkout grant.

The record advertises ``drain`` only when the key was written, and the key is
removed on drain and on exit, before the record is withdrawn.
"""

from __future__ import annotations

import contextlib
import hmac
import os
import secrets
import stat

import structlog

from trw_memory.daemon._paths import _NOFOLLOW, DaemonPaths, read_secret_file, write_secret_file
from trw_memory.exceptions import DaemonSecretUnreadableError

__all__ = ["create_drain_key", "keys_match", "read_drain_key", "remove_drain_key"]

logger = structlog.get_logger(__name__)

#: Any group or other permission bit makes a key file untrustworthy to the client.
_SHARED_BITS = 0o077


def keys_match(presented: str, expected: str) -> bool:
    """Constant-time comparison of a presented key with the daemon's."""
    return hmac.compare_digest(presented.encode(), expected.encode())


def create_drain_key(paths: DaemonPaths) -> str:
    """Write a fresh key to :attr:`DaemonPaths.drain_key` and return it.

    ``write_secret_file`` creates an exclusive (``O_EXCL|O_NOFOLLOW``) 0600 temporary and renames it into
    place, so a reader sees the whole key or none, and a symlink planted at the path is replaced, never followed.
    """
    key = secrets.token_hex(32)
    write_secret_file(paths.drain_key, key)
    return key


def remove_drain_key(paths: DaemonPaths, key: str) -> None:
    """Remove the key file if it still holds *key*; any other content (or an unreadable file) is left alone."""
    try:
        current = read_secret_file(paths.drain_key)
    except DaemonSecretUnreadableError:  # trw-fail-silent-allow: not ours to delete on no evidence; logged where read
        return
    if current is not None and keys_match(current.strip(), key):
        paths.drain_key.unlink(missing_ok=True)
        logger.info("daemon_drain_key_removed", path=str(paths.drain_key))


def read_drain_key(paths: DaemonPaths) -> str | None:
    """The daemon's key, or ``None`` when the file is absent, a symlink, not a regular file, not this user's,
    or readable by group or others: a key another principal could have read or planted is not the daemon's.
    """
    try:
        fd = os.open(paths.drain_key, os.O_RDONLY | _NOFOLLOW)
    except OSError as exc:  # trw-fail-silent-allow: None IS the refusal; the caller names the file to the user
        logger.info("daemon_drain_key_unreadable", path=str(paths.drain_key), error=type(exc).__name__)
        return None
    with contextlib.closing(os.fdopen(fd, "rb")) as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & _SHARED_BITS:
            logger.warning("daemon_drain_key_untrusted", path=str(paths.drain_key), mode=oct(info.st_mode))
            return None
        try:
            return handle.read().decode("ascii").strip() or None
        except (
            OSError,
            UnicodeDecodeError,
        ) as exc:  # trw-fail-silent-allow: a corrupt key is no key; the caller refuses
            logger.warning("daemon_drain_key_corrupt", path=str(paths.drain_key), error=type(exc).__name__)
            return None
