"""PRD-SEC-016: ``resolve_user_memory_dir`` refuses an untrusted resolved directory.

``TRW_USER_DIR`` (or ``XDG_DATA_HOME``, or the home fallback) names a base
directory that is then trusted to hold secrets (the daemon token, grants,
the SQLite store). This suite proves that trust is verified, not assumed.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from trw_memory.exceptions import UntrustedDirectoryError
from trw_memory.user_paths import resolve_user_memory_dir

_POSIX_ONLY = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode/owner bits")


def test_resolve_creates_and_returns_memory_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "user"))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    resolved = resolve_user_memory_dir(create=True)

    assert resolved == (tmp_path / "user" / "memory").resolve()
    assert resolved.is_dir()


def test_resolve_create_false_missing_dir_does_not_raise_and_does_not_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "absent-user"))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    resolved = resolve_user_memory_dir(create=False)

    assert not resolved.exists(), "a presence probe must not create anything"


@_POSIX_ONLY
def test_resolve_refuses_world_writable_dir_owned_by_someone_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trw_memory._dir_trust as dir_trust

    base = tmp_path / "shared-user"
    memory_dir = base / "memory"
    memory_dir.mkdir(parents=True, mode=0o777)
    os.chmod(memory_dir, 0o777)
    real_uid = memory_dir.stat().st_uid
    monkeypatch.setattr(dir_trust.os, "geteuid", lambda: real_uid + 1)
    monkeypatch.setenv("TRW_USER_DIR", str(base))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    with pytest.raises(UntrustedDirectoryError, match="group/world-writable"):
        resolve_user_memory_dir(create=True)


@_POSIX_ONLY
def test_resolve_auto_hardens_own_permissive_dir(tmp_path: Path) -> None:
    base = tmp_path / "own-permissive"
    memory_dir = base / "memory"
    memory_dir.mkdir(parents=True, mode=0o775)
    os.chmod(memory_dir, 0o775)
    old = os.environ.get("TRW_USER_DIR")
    old_xdg = os.environ.get("XDG_DATA_HOME")
    os.environ["TRW_USER_DIR"] = str(base)
    os.environ.pop("XDG_DATA_HOME", None)
    try:
        resolved = resolve_user_memory_dir(create=True)
        assert stat.S_IMODE(resolved.stat().st_mode) == 0o700
    finally:
        if old is None:
            os.environ.pop("TRW_USER_DIR", None)
        else:
            os.environ["TRW_USER_DIR"] = old
        if old_xdg is not None:
            os.environ["XDG_DATA_HOME"] = old_xdg


@_POSIX_ONLY
def test_resolve_refuses_a_directory_swapped_for_a_foreign_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulates the TOCTOU window this PRD closes: a caller resolves the
    directory once (creating it), then a local attacker swaps it for a
    symlink to an attacker-owned, world-writable location before a second
    resolution trusts it again. ``Path.resolve()`` follows the symlink to
    the real target directory -- the defense is that the OWNERSHIP check on
    that real target still refuses it, not that the symlink itself is
    detected as a symlink."""
    import trw_memory._dir_trust as dir_trust

    base = tmp_path / "swap-user"
    monkeypatch.setenv("TRW_USER_DIR", str(base))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    first = resolve_user_memory_dir(create=True)
    assert first.is_dir()

    # Attacker's window: swap the real directory for a symlink to a
    # world-writable location that is NOT owned by the current process.
    evil_target = tmp_path / "evil-target"
    evil_target.mkdir(mode=0o777)
    os.chmod(evil_target, 0o777)
    real_uid = evil_target.stat().st_uid
    first.rmdir()
    first.symlink_to(evil_target, target_is_directory=True)
    monkeypatch.setattr(dir_trust.os, "geteuid", lambda: real_uid + 1)

    with pytest.raises((OSError, UntrustedDirectoryError)):
        resolve_user_memory_dir(create=True)
