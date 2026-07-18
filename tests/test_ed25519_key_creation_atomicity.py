"""Atomicity and permission tests for Ed25519 signing-key creation."""

from __future__ import annotations

import contextlib
import os
import stat
import threading
from pathlib import Path

import pytest

from trw_memory.security import keys


def test_concurrent_creators_return_the_persisted_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key_path = tmp_path / "security" / "signing.bin"
    original_generate = keys.generate_ed25519_signing_key
    generation_count = 0
    generation_lock = threading.Lock()

    def _counted_generate() -> bytes:
        nonlocal generation_count
        with generation_lock:
            generation_count += 1
        return original_generate()

    monkeypatch.setattr(keys, "generate_ed25519_signing_key", _counted_generate)
    monkeypatch.setattr(keys, "load_ed25519_signing_key", lambda path: path.read_bytes())
    start = threading.Barrier(3)
    returned: list[bytes] = []

    def _create() -> None:
        start.wait()
        returned.append(keys.get_or_create_ed25519_key_at_path(key_path))

    threads = [threading.Thread(target=_create) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert generation_count == 1
    assert len(returned) == 2
    assert returned[0] == returned[1] == key_path.read_bytes()


def test_unlocked_contenders_only_observe_complete_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key_path = tmp_path / "security" / "signing.bin"
    original_link = os.link
    first_ready = threading.Event()
    release_first = threading.Event()
    link_count = 0
    link_lock = threading.Lock()
    returned: list[bytes] = []

    monkeypatch.setattr(keys, "lock_for_rmw", lambda path: contextlib.nullcontext(path))

    def _load_complete(path: Path) -> bytes:
        data = path.read_bytes()
        if len(data) != 32:
            raise keys.ConfigError("partial seed")
        return data

    def _pause_first_link(src: Path, dst: Path, *, follow_symlinks: bool = True) -> None:
        nonlocal link_count
        with link_lock:
            link_count += 1
            is_first = link_count == 1
        if is_first:
            first_ready.set()
            assert release_first.wait(timeout=2)
        original_link(src, dst, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(keys.os, "link", _pause_first_link)
    monkeypatch.setattr(keys, "load_ed25519_signing_key", _load_complete)

    first = threading.Thread(target=lambda: returned.append(keys.get_or_create_ed25519_key_at_path(key_path)))
    first.start()
    assert first_ready.wait(timeout=2)
    assert not key_path.exists()
    second_result = keys.get_or_create_ed25519_key_at_path(key_path)
    release_first.set()
    first.join(timeout=2)

    assert not first.is_alive()
    assert returned == [second_result]
    assert second_result == key_path.read_bytes()


def test_key_has_exact_private_mode_under_restrictive_umask(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key_path = tmp_path / "security" / "signing.bin"
    key_path.parent.mkdir(mode=0o700)
    observed_mode: list[int] = []

    def _observe_mode(path: Path) -> object:
        observed_mode.append(stat.S_IMODE(path.stat().st_mode))
        return object()

    monkeypatch.setattr(keys, "load_ed25519_signing_key", _observe_mode)
    previous_umask = os.umask(0o777)
    try:
        keys.get_or_create_ed25519_key_at_path(key_path)
    finally:
        os.umask(previous_umask)

    assert observed_mode == [0o600]


def test_key_permission_fallback_uses_portable_chmod(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key_path = tmp_path / "security" / "signing.bin"
    original_chmod = os.chmod
    chmod_calls: list[tuple[Path, int]] = []

    def _portable_chmod(path: Path, mode: int) -> None:
        chmod_calls.append((Path(path), mode))
        original_chmod(path, mode)

    monkeypatch.delattr(keys.os, "fchmod", raising=False)
    monkeypatch.setattr(keys.os, "chmod", _portable_chmod)
    monkeypatch.setattr(keys, "load_ed25519_signing_key", lambda path: path.read_bytes())

    assert keys.get_or_create_ed25519_key_at_path(key_path) == key_path.read_bytes()
    assert len(chmod_calls) == 1
    assert chmod_calls[0][1] == 0o600


def test_symlink_key_path_is_rejected_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"not-a-key")
    key_path = tmp_path / "security" / "signing.bin"
    key_path.parent.mkdir()
    key_path.symlink_to(target)

    assert keys.get_or_create_ed25519_key_at_path(key_path) is None
    assert target.read_bytes() == b"not-a-key"


def test_failed_write_leaves_no_partial_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key_path = tmp_path / "security" / "signing.bin"
    original_write = os.write
    calls = 0

    def _fail_after_partial(fd: int, data: bytes | memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(fd, data[:5])
        raise OSError("simulated write failure")

    monkeypatch.setattr(keys.os, "write", _fail_after_partial)
    with pytest.raises(OSError, match="simulated write failure"):
        keys.get_or_create_ed25519_key_at_path(key_path)

    assert not key_path.exists()
    assert not list(key_path.parent.glob(".*.tmp"))
    monkeypatch.setattr(keys.os, "write", original_write)
    monkeypatch.setattr(keys, "load_ed25519_signing_key", lambda path: path.read_bytes())
    assert keys.get_or_create_ed25519_key_at_path(key_path) == key_path.read_bytes()
    assert len(key_path.read_bytes()) == 32
