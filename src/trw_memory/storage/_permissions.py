"""Secret-bearing SQLite database and WAL sidecar permissions.

TOCTOU hardening (PRD-SEC-016) lives in :mod:`trw_memory._dir_trust`; this
module is a consumer, not a second copy -- see that module's docstring for
the shared trust model and its documented residual window (SQLite's own
``sqlite3.connect`` opens the DB file by path, not fd).
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import structlog

from trw_memory._dir_trust import (
    DIR_FD_SUPPORTED,
    open_verified_dir_fd,
    verify_and_harden_dir_fd,
)
from trw_memory._live_stores import FD_LOCK
from trw_memory.exceptions import StorageError, UntrustedDirectoryError

logger = structlog.get_logger(__name__)

DB_FILE_MODE = 0o600
DB_PARENT_MODE = 0o700

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
#: Exclusive-create rounds before a store that keeps appearing and vanishing is refused.
_CREATE_ROUNDS = 8


def _file_backed_path(db_path: Path | str) -> Path | None:
    path = Path(db_path)
    name = str(path)
    return None if name == ":memory:" or name.startswith("file::memory:") else path


#: ``fchmodat(AT_SYMLINK_NOFOLLOW)``: chmod the entry itself, never a symlink's target.
_CHMOD_NOFOLLOW = os.chmod in os.supports_follow_symlinks
#: Linux: an ``O_PATH`` descriptor names the entry without opening it for I/O, and
#: closing one does not release the process's POSIX locks on the file.
_O_PATH: int | None = getattr(os, "O_PATH", None) if sys.platform.startswith("linux") else None


def _chmod_via_o_path(name: str, dir_fd: int, expected: tuple[int, int], o_path: int) -> None:
    """Linux chmod of exactly the ``lstat``-ed entry: an ``O_PATH|O_NOFOLLOW`` fd, then ``/proc/self/fd``."""
    fd = os.open(name, o_path | _NOFOLLOW | _CLOEXEC, dir_fd=dir_fd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) != expected:
            raise OSError(f"{name} was replaced before the chmod")
        os.chmod(f"/proc/self/fd/{fd}", DB_FILE_MODE)
    finally:
        os.close(fd)


def _chmod_entry(*, name: str, dir_fd: int | None, path: Path, strict: bool) -> bool:
    """Tighten one existing store entry to ``DB_FILE_MODE`` WITHOUT opening it; False when absent.

    Never a descriptor on the entry (C15): ``close()`` of any descriptor on a
    store this process has connected releases the connection's SQLite locks
    (see :mod:`trw_memory._live_stores`). The entry is ``lstat``-ed
    through the verified parent descriptor, then chmod-ed without following a
    symlink: on Linux through an ``O_PATH`` descriptor (whose close releases
    no locks), elsewhere by ``fchmodat(AT_SYMLINK_NOFOLLOW)``; a platform
    with neither is refused rather than chmod-ed by a following path. *strict*
    raises on a refusal or failure (the store itself); otherwise it is logged
    (sidecars).
    """

    def fail(message: str, exc: BaseException | None = None) -> bool:
        if strict:
            raise StorageError(message, path=str(path)) from exc
        logger.warning("db_chmod_failed", path=str(path), mode=oct(DB_FILE_MODE), error=message)
        return True

    try:
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False) if dir_fd is not None else os.lstat(path)
    except FileNotFoundError:  # trw-fail-silent-allow: a wal/shm sidecar (or a not-yet-created store) is absent -- the ordinary "nothing to harden yet" case, reported to the caller as False
        return False
    except OSError as exc:
        return fail(f"cannot stat {type(exc).__name__}", exc)
    if not stat.S_ISREG(st.st_mode):
        if stat.S_ISLNK(st.st_mode):
            logger.error("db_symlink_rejected", path=str(path))
        return fail("not a regular file")
    if stat.S_IMODE(st.st_mode) == DB_FILE_MODE:
        return True
    try:
        if dir_fd is None:
            os.chmod(path, DB_FILE_MODE)  # no dir_fd support: PRD-SEC-016's recorded residual
        elif _O_PATH is not None:
            _chmod_via_o_path(name, dir_fd, (st.st_dev, st.st_ino), _O_PATH)
        elif _CHMOD_NOFOLLOW:
            os.chmod(name, DB_FILE_MODE, dir_fd=dir_fd, follow_symlinks=False)
        else:
            return fail("this platform cannot chmod without following a symlink")
        after = os.stat(name, dir_fd=dir_fd, follow_symlinks=False) if dir_fd is not None else os.lstat(path)
    except (OSError, NotImplementedError) as exc:
        return fail(f"chmod {type(exc).__name__}", exc)
    if (after.st_dev, after.st_ino) != (st.st_dev, st.st_ino):
        return fail("the entry was replaced during the chmod")
    return True


def _create_or_tighten(path: Path, dir_fd: int | None) -> None:
    """Tighten an existing store by name, or create a missing one with ``O_EXCL`` -- never both.

    An existing store is chmod-ed, never opened: this process may already hold a
    connection on it, and closing a descriptor would release that connection's
    locks (C15). A missing store is created with ``O_EXCL`` only -- never "open it
    if someone else created it first", since that entry could be a hard link to
    a live store -- and a lost race goes back to the descriptor-free path. The
    one descriptor closed here is a brand-new inode created under ``FD_LOCK``,
    which every connect holds too, so no connection can hold it.
    """
    target: str | Path = path.name if dir_fd is not None else path
    with FD_LOCK:
        for _ in range(_CREATE_ROUNDS):
            if _chmod_entry(name=path.name, dir_fd=dir_fd, path=path, strict=True):
                return
            try:
                fd = os.open(
                    target, os.O_RDWR | os.O_CREAT | os.O_EXCL | _CLOEXEC | _NOFOLLOW, DB_FILE_MODE, dir_fd=dir_fd
                )
            except (
                FileExistsError
            ):  # trw-fail-silent-allow: another creator won the race; the next round chmods its file without opening it
                continue
            except OSError as exc:
                raise StorageError(f"Secure SQLite open failed: {exc}", path=str(path)) from exc
            try:
                os.fchmod(fd, DB_FILE_MODE)
            except OSError as exc:
                raise StorageError(f"Secure SQLite chmod failed: {exc}", path=str(path)) from exc
            finally:
                os.close(fd)
            return
    raise StorageError("Secure SQLite create lost every race against concurrent removal", path=str(path))


def harden_db_file_mode(db_path: Path | str) -> None:
    """Tighten an existing DB and live WAL/SHM sidecars, best-effort, opening none of them."""
    path = _file_backed_path(db_path)
    if path is None:
        return
    dir_fd: int | None = None
    if DIR_FD_SUPPORTED:
        try:
            dir_fd = open_verified_dir_fd(path.parent, create=False)
        except (OSError, UntrustedDirectoryError):
            dir_fd = None  # Best-effort: fall back to path-based opens below.
    try:
        for candidate_name, candidate_path in (
            (path.name, path),
            (f"{path.name}-wal", Path(f"{path}-wal")),
            (f"{path.name}-shm", Path(f"{path}-shm")),
        ):
            _chmod_entry(name=candidate_name, dir_fd=dir_fd, path=candidate_path, strict=False)
    finally:
        if dir_fd is not None:
            os.close(dir_fd)


def prepare_db_file_mode(db_path: Path | str) -> None:
    """Securely create or tighten a DB before SQLite enables WAL mode.

    The parent directory is opened once (refusing a symlink and an untrusted
    owner/permission set), and the DB file itself is then opened
    ``dir_fd``-relative to that SAME open parent descriptor -- there is no
    point between verifying the parent and opening the file where a path
    re-resolution could be redirected by a swap, because no path
    re-resolution happens.
    """
    path = _file_backed_path(db_path)
    if path is None:
        return
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.is_symlink():
            logger.error("db_symlink_rejected", path=str(candidate))
            raise StorageError("Refusing symlink SQLite database or sidecar path", path=str(candidate))

    if not DIR_FD_SUPPORTED:
        # Platform without openat(2) dir_fd support (rare / legacy Windows
        # Python): fall back to a path-based check, best-effort, and accept
        # the wider TOCTOU window -- recorded as residual risk in PRD-SEC-016.
        try:
            parent_mode = path.parent.stat().st_mode
        except OSError as exc:
            raise StorageError(f"Cannot verify SQLite parent permissions: {exc}", path=str(path.parent)) from exc
        if parent_mode & 0o022:
            raise StorageError("SQLite parent directory must not be group/world writable", path=str(path.parent))
        _create_or_tighten(path, None)
        harden_db_file_mode(path)
        return

    try:
        parent_fd = open_verified_dir_fd(path.parent, create=False)
    except OSError as exc:
        raise StorageError(f"Cannot securely open SQLite parent: {exc}", path=str(path.parent)) from exc
    try:
        try:
            verify_and_harden_dir_fd(parent_fd, path.parent, target_mode=DB_PARENT_MODE)
        except UntrustedDirectoryError as exc:
            raise StorageError(str(exc), path=str(path.parent)) from exc

        _create_or_tighten(path, parent_fd)
    finally:
        os.close(parent_fd)
    harden_db_file_mode(path)
