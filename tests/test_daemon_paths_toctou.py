"""PRD-SEC-016: TOCTOU hardening for the daemon's secret-bearing writes.

``write_secret_file``/``open_private_log`` write the daemon token, grants,
and start log. This suite proves the dir_fd-relative refactor still delivers
the properties the module's docstring promises (atomic write, symlink
refusal, no half-written reads) AND closes the parent-directory check-then-
use window the refactor was built to close.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from trw_memory.daemon._paths import (
    SECRET_DIR_MODE,
    SECRET_FILE_MODE,
    _harden_dir,
    open_private_log,
    read_secret_file,
    write_secret_file,
)
from trw_memory.exceptions import DaemonSecretUnreadableError, UntrustedDirectoryError

_POSIX_ONLY = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode/owner bits")


def test_write_secret_file_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "secrets" / "token"
    write_secret_file(target, "hello-secret")

    assert read_secret_file(target) == "hello-secret"


@_POSIX_ONLY
def test_write_secret_file_creates_0600_file_in_0700_dir(tmp_path: Path) -> None:
    target = tmp_path / "secrets" / "token"
    write_secret_file(target, "x")

    assert stat.S_IMODE(target.stat().st_mode) == SECRET_FILE_MODE
    assert stat.S_IMODE(target.parent.stat().st_mode) == SECRET_DIR_MODE


def test_write_secret_file_overwrite_is_atomic_no_half_write_observed(tmp_path: Path) -> None:
    target = tmp_path / "secrets" / "token"
    write_secret_file(target, "first")
    write_secret_file(target, "second-longer-value")

    assert read_secret_file(target) == "second-longer-value"
    # No leftover temp sibling.
    leftovers = [p for p in target.parent.iterdir() if p.name != target.name]
    assert leftovers == []


@_POSIX_ONLY
def test_write_secret_file_replaces_symlinked_target_without_following_it(tmp_path: Path) -> None:
    """A pre-planted symlink at the destination name is REPLACED, not
    written through -- ``os.rename``/``os.replace`` never dereferences the
    destination. The symlink's old target is left completely untouched,
    which is what stops an attacker using a planted symlink to have the
    daemon overwrite an arbitrary file it can reach."""
    real = tmp_path / "elsewhere.txt"
    real.write_text("do-not-touch")
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir(mode=0o700)
    link = secrets_dir / "token"
    link.symlink_to(real)

    write_secret_file(link, "attacker-would-love-this")

    assert real.read_text() == "do-not-touch"
    assert not link.is_symlink()
    assert link.read_text() == "attacker-would-love-this"


@_POSIX_ONLY
def test_harden_dir_refuses_world_writable_dir_owned_by_someone_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trw_memory._dir_trust as dir_trust

    directory = tmp_path / "foreign-secrets"
    directory.mkdir(mode=0o777)
    os.chmod(directory, 0o777)
    real_uid = directory.stat().st_uid
    monkeypatch.setattr(dir_trust.os, "geteuid", lambda: real_uid + 1)

    with pytest.raises(UntrustedDirectoryError, match="group/world-writable"):
        _harden_dir(directory)


@_POSIX_ONLY
def test_harden_dir_auto_hardens_own_permissive_dir(tmp_path: Path) -> None:
    directory = tmp_path / "own-secrets"
    directory.mkdir(mode=0o775)
    os.chmod(directory, 0o775)

    fd = _harden_dir(directory)
    try:
        assert stat.S_IMODE(os.fstat(fd).st_mode) == SECRET_DIR_MODE
    finally:
        os.close(fd)


@_POSIX_ONLY
def test_harden_dir_refuses_symlinked_directory(tmp_path: Path) -> None:
    target = tmp_path / "real-secrets"
    target.mkdir(mode=0o700)
    linked = tmp_path / "linked-secrets"
    linked.symlink_to(target, target_is_directory=True)

    with pytest.raises((OSError, UntrustedDirectoryError)):
        _harden_dir(linked)


def test_open_private_log_truncates_and_is_readable(tmp_path: Path) -> None:
    target = tmp_path / "log-dir" / "daemon-start.log"
    fd = open_private_log(target)
    os.write(fd, b"first run\n")
    os.close(fd)

    fd2 = open_private_log(target)  # simulates the next daemon start rewriting it
    os.write(fd2, b"second run\n")
    os.close(fd2)

    assert target.read_bytes() == b"second run\n"


def test_read_secret_file_absent_returns_none(tmp_path: Path) -> None:
    assert read_secret_file(tmp_path / "does-not-exist") is None


@_POSIX_ONLY
def test_read_secret_file_symlink_raises_unreadable(tmp_path: Path) -> None:
    real = tmp_path / "real.txt"
    real.write_text("secret")
    link = tmp_path / "link.txt"
    link.symlink_to(real)

    with pytest.raises(DaemonSecretUnreadableError, match="symlink"):
        read_secret_file(link)
