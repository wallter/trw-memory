"""PRD-QUAL-110-FR02: SQLiteBackend hardens its DB and WAL sidecars to 0600.

A file-backed ``memory.db`` is secret-bearing (learning content, provenance).
The backend chmods it to 0600 on creation, mirroring the trw-mcp pins.json
0600 hardening. In-memory (``:memory:``) backends are unaffected, and a chmod
failure on a non-POSIX platform degrades gracefully (no raise).
"""

from __future__ import annotations

import errno
import os
import sqlite3
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.storage.sqlite_backend import SQLiteBackend

_POSIX_ONLY = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode bits")


@_POSIX_ONLY
def test_file_backed_db_is_0600(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    try:
        assert db_path.exists()
        assert stat.S_IMODE(os.stat(db_path).st_mode) == 0o600
    finally:
        backend.close()


@_POSIX_ONLY
@pytest.mark.parametrize("process_umask", [0o000, 0o002])
def test_new_database_parent_is_private_under_permissive_umask(tmp_path: Path, process_umask: int) -> None:
    db_path = tmp_path / "new-parent" / "memory.db"
    previous_umask = os.umask(process_umask)
    try:
        backend = SQLiteBackend(db_path)
        try:
            assert stat.S_IMODE(os.stat(db_path.parent).st_mode) == 0o700
            assert stat.S_IMODE(os.stat(db_path).st_mode) == 0o600
        finally:
            backend.close()
    finally:
        os.umask(previous_umask)


@_POSIX_ONLY
def test_owned_legacy_parent_is_migrated_to_private_mode(tmp_path: Path) -> None:
    parent = tmp_path / "legacy-parent"
    parent.mkdir(mode=0o775)
    os.chmod(parent, 0o775)

    backend = SQLiteBackend(parent / "memory.db")
    try:
        assert stat.S_IMODE(parent.stat().st_mode) == 0o700
    finally:
        backend.close()


@_POSIX_ONLY
def test_unowned_unsafe_parent_remains_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trw_memory.storage._permissions as permissions

    parent = tmp_path / "foreign-parent"
    parent.mkdir(mode=0o775)
    os.chmod(parent, 0o775)
    monkeypatch.setattr(permissions.os, "geteuid", lambda: parent.stat().st_uid + 1)

    with pytest.raises(StorageError, match="must not be group/world writable"):
        SQLiteBackend(parent / "memory.db")
    assert stat.S_IMODE(parent.stat().st_mode) == 0o775


@_POSIX_ONLY
def test_parent_fchmod_failure_remains_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trw_memory.storage._permissions as permissions

    parent = tmp_path / "chmod-failure"
    parent.mkdir(mode=0o775)
    os.chmod(parent, 0o775)

    def fail_fchmod(_fd: int, _mode: int) -> None:
        raise OSError("denied")

    monkeypatch.setattr(permissions.os, "fchmod", fail_fchmod)

    with pytest.raises(StorageError, match="Cannot harden SQLite parent permissions"):
        SQLiteBackend(parent / "memory.db")


@_POSIX_ONLY
def test_parent_hardening_postcondition_is_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trw_memory.storage._permissions as permissions

    parent = tmp_path / "chmod-noop"
    parent.mkdir(mode=0o775)
    os.chmod(parent, 0o775)
    monkeypatch.setattr(permissions.os, "fchmod", lambda _fd, _mode: None)

    with pytest.raises(StorageError, match="did not take effect"):
        SQLiteBackend(parent / "memory.db")


@_POSIX_ONLY
def test_sticky_shared_parent_is_not_hardened(tmp_path: Path) -> None:
    parent = tmp_path / "sticky-parent"
    parent.mkdir()
    os.chmod(parent, 0o1777)

    backend = SQLiteBackend(parent / "memory.db")
    try:
        assert stat.S_IMODE(parent.stat().st_mode) == 0o1777
    finally:
        backend.close()


@_POSIX_ONLY
def test_parent_symlink_is_rejected_without_chmodding_target(tmp_path: Path) -> None:
    target = tmp_path / "parent-target"
    target.mkdir(mode=0o775)
    os.chmod(target, 0o775)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(target, target_is_directory=True)

    with pytest.raises(StorageError, match="Cannot securely open SQLite parent"):
        SQLiteBackend(linked_parent / "memory.db")
    assert stat.S_IMODE(target.stat().st_mode) == 0o775


@_POSIX_ONLY
def test_live_wal_sidecars_are_0600_under_permissive_umask(tmp_path: Path) -> None:
    db_path = tmp_path / "secret.db"
    previous_umask = os.umask(0o022)
    try:
        backend = SQLiteBackend(db_path)
        try:
            sidecars = [db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]
            assert all(path.exists() for path in sidecars)
            assert {stat.S_IMODE(os.stat(path).st_mode) for path in sidecars} == {0o600}
        finally:
            backend.close()
    finally:
        os.umask(previous_umask)


def test_in_memory_db_does_not_raise() -> None:
    """The :memory: path has no on-disk file to chmod and must not raise."""
    backend = SQLiteBackend(Path(":memory:"))
    backend.close()


@_POSIX_ONLY
def test_chmod_failure_is_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A chmod OSError (non-POSIX) does not block backend construction."""
    import trw_memory.storage._permissions as mod

    real_chmod = os.chmod

    def _selective(path: object, mode: int, *a: object, **k: object) -> None:
        if str(path).endswith("memory.db"):
            raise OSError("not supported")
        real_chmod(path, mode, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(mod.os, "chmod", _selective)
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)  # must not raise
    backend.close()


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
@_POSIX_ONLY
def test_symlink_database_or_sidecar_is_rejected_without_chmodding_target(tmp_path: Path, suffix: str) -> None:
    target = tmp_path / "target.db"
    target.write_bytes(b"")
    os.chmod(target, 0o644)
    db_path = tmp_path / "memory.db"
    link = Path(f"{db_path}{suffix}")
    link.symlink_to(target)

    with pytest.raises(StorageError, match="symlink"):
        SQLiteBackend(db_path)

    assert stat.S_IMODE(os.stat(target).st_mode) == 0o644


@_POSIX_ONLY
def test_nofollow_race_failure_blocks_connection_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trw_memory.storage._permissions as permissions

    db_path = tmp_path / "memory.db"
    real_open = os.open

    def fail_open(path: object, flags: int, mode: int = 0o777) -> int:
        if Path(str(path)) == db_path:
            raise OSError(errno.ELOOP, "loop")
        return real_open(path, flags, mode)  # type: ignore[arg-type]

    monkeypatch.setattr(permissions.os, "open", fail_open)
    with (
        patch("trw_memory.storage.sqlite_backend._init_open_connection_with_recovery") as open_connection,
        pytest.raises(StorageError, match="Secure SQLite open failed"),
    ):
        SQLiteBackend(db_path)

    open_connection.assert_not_called()


@_POSIX_ONLY
def test_recovery_open_secures_db_and_sidecars(tmp_path: Path) -> None:
    from trw_memory.storage._recovery import _open_recovered_conn

    db_path = tmp_path / "recovered.db"
    previous_umask = os.umask(0o022)
    try:
        connection = _open_recovered_conn(db_path, dbapi=sqlite3, sqlcipher_key_hex=None)
        try:
            paths = [db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]
            assert all(path.exists() for path in paths)
            assert {stat.S_IMODE(os.stat(path).st_mode) for path in paths} == {0o600}
        finally:
            connection.close()
    finally:
        os.umask(previous_umask)


@_POSIX_ONLY
def test_reconnect_resecures_replaced_file_and_sidecars(tmp_path: Path) -> None:
    from trw_memory.storage._stale_handle import reconnect

    db_path = tmp_path / "reconnect.db"
    backend = SQLiteBackend(db_path)
    try:
        paths = [db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]
        for path in paths:
            os.chmod(path, 0o644)

        reconnect(backend)

        assert {stat.S_IMODE(os.stat(path).st_mode) for path in paths} == {0o600}
    finally:
        backend.close()
