"""Secret-bearing SQLite database and WAL sidecar permissions."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import structlog

from trw_memory.exceptions import StorageError

logger = structlog.get_logger(__name__)

DB_FILE_MODE = 0o600
DB_PARENT_MODE = 0o700


def _file_backed_path(db_path: Path | str) -> Path | None:
    path = Path(db_path)
    name = str(path)
    return None if name == ":memory:" or name.startswith("file::memory:") else path


def harden_db_file_mode(db_path: Path | str) -> None:
    """Tighten an existing DB and live WAL/SHM sidecars, best-effort."""
    path = _file_backed_path(db_path)
    if path is None:
        return
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if not candidate.exists() or candidate.is_symlink():
            continue
        try:
            os.chmod(candidate, DB_FILE_MODE)
        except OSError as exc:
            logger.warning(
                "db_chmod_failed",
                path=str(candidate),
                mode=oct(DB_FILE_MODE),
                error=type(exc).__name__,
            )


def prepare_db_file_mode(db_path: Path | str) -> None:
    """Securely create or tighten a DB before SQLite enables WAL mode."""
    path = _file_backed_path(db_path)
    if path is None:
        return
    candidates = (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
    for candidate in candidates:
        if candidate.is_symlink():
            logger.error("db_symlink_rejected", path=str(candidate))
            raise StorageError("Refusing symlink SQLite database or sidecar path", path=str(candidate))

    geteuid = getattr(os, "geteuid", None)
    if callable(geteuid):
        parent_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            parent_fd = os.open(path.parent, parent_flags)
        except OSError as exc:
            raise StorageError(f"Cannot securely open SQLite parent: {exc}", path=str(path.parent)) from exc
        try:
            parent_stat = os.fstat(parent_fd)
            parent_mode = parent_stat.st_mode
            if parent_mode & 0o022 and not parent_mode & stat.S_ISVTX:
                if parent_stat.st_uid != geteuid():
                    raise StorageError(
                        "SQLite parent directory must not be group/world writable", path=str(path.parent)
                    )
                try:
                    os.fchmod(parent_fd, DB_PARENT_MODE)
                except OSError as exc:
                    raise StorageError(
                        f"Cannot harden SQLite parent permissions: {exc}", path=str(path.parent)
                    ) from exc
                hardened_mode = os.fstat(parent_fd).st_mode
                if hardened_mode & 0o022:
                    raise StorageError("SQLite parent permission hardening did not take effect", path=str(path.parent))
                logger.info(
                    "db_parent_permissions_hardened",
                    path=str(path.parent),
                    old_mode=oct(stat.S_IMODE(parent_mode)),
                    new_mode=oct(stat.S_IMODE(hardened_mode)),
                )
        finally:
            os.close(parent_fd)
    else:
        try:
            parent_mode = path.parent.stat().st_mode
        except OSError as exc:
            raise StorageError(f"Cannot verify SQLite parent permissions: {exc}", path=str(path.parent)) from exc
        if parent_mode & 0o022 and not parent_mode & stat.S_ISVTX:
            raise StorageError("SQLite parent directory must not be group/world writable", path=str(path.parent))

    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, DB_FILE_MODE)
    except OSError as exc:
        raise StorageError(f"Secure SQLite open failed: {exc}", path=str(path)) from exc
    else:
        try:
            os.fchmod(fd, DB_FILE_MODE)
        except OSError as exc:
            raise StorageError(f"Secure SQLite chmod failed: {exc}", path=str(path)) from exc
        finally:
            os.close(fd)
    harden_db_file_mode(path)
