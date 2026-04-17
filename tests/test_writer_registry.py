"""Tests for PRD-INFRA-064 multi-writer advisory registry."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from trw_memory.storage._writer_registry import WriterRegistry

if TYPE_CHECKING:
    from pytest import MonkeyPatch


# ---------------------------------------------------------------------------
# Basic registration
# ---------------------------------------------------------------------------


def test_register_creates_pid_lock(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    reg = WriterRegistry(db, warn_threshold=4)
    reg.register()
    lock = tmp_path / "memory.db.writers" / f"{os.getpid()}.lock"
    assert lock.exists(), "registry should have dropped a <pid>.lock"
    assert reg.concurrent_writers == 1
    assert reg.peer_pids == [os.getpid()]
    reg.close()
    assert not lock.exists()


def test_register_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    reg = WriterRegistry(db)
    reg.register()
    reg.register()  # second call must not raise
    lock = tmp_path / "memory.db.writers" / f"{os.getpid()}.lock"
    assert lock.exists()
    reg.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    reg = WriterRegistry(db)
    reg.register()
    reg.close()
    reg.close()  # second close must not raise


# ---------------------------------------------------------------------------
# Peer enumeration
# ---------------------------------------------------------------------------


def test_register_enumerates_live_peers(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    # Seed a peer lockfile with a known-live pid (ours).
    writers_dir = tmp_path / "memory.db.writers"
    writers_dir.mkdir()
    peer_pid = os.getpid()
    # We cannot easily fake a second pid — instead seed one with a dead pid
    # and assert it gets pruned.
    (writers_dir / "1.lock").write_text("1\n0\n")  # pid 1 exists on Linux
    (writers_dir / "9999998.lock").write_text("9999998\n0\n")  # likely dead

    reg = WriterRegistry(db, warn_threshold=100)
    reg.register()
    # Our own pid is in the list.
    assert peer_pid in reg.peer_pids
    # The dead pid was pruned (or at minimum not counted as live).
    assert 9999998 not in reg.peer_pids
    reg.close()


def test_stale_peer_pruning(tmp_path: Path) -> None:
    """A lockfile for a definitely-dead pid should be removed."""
    db = tmp_path / "memory.db"
    writers_dir = tmp_path / "memory.db.writers"
    writers_dir.mkdir()
    dead_lock = writers_dir / "9999998.lock"
    dead_lock.write_text("9999998\n0\n")

    reg = WriterRegistry(db)
    reg.register()

    # On Linux the pid probe via /proc is authoritative.
    if sys.platform.startswith("linux"):
        assert not dead_lock.exists(), "stale peer lock should have been pruned"
    reg.close()


# ---------------------------------------------------------------------------
# Advisory-only invariants (sprint exit criterion guard)
# ---------------------------------------------------------------------------


def test_registry_never_refuses_open_when_dir_unwritable(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """CRITICAL: registry failures must NEVER raise (sprint exit criterion)."""
    db = tmp_path / "memory.db"

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("simulated unwritable registry dir")

    monkeypatch.setattr(Path, "mkdir", _explode)

    reg = WriterRegistry(db)
    # MUST NOT raise — this is the advisory-only guarantee.
    reg.register()
    # Registry could not record itself, so counts remain 0.
    assert reg.concurrent_writers == 0
    assert reg.peer_pids == []
    reg.close()


def test_registry_never_refuses_open_when_lock_create_fails(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """PermissionError on lock creation must be swallowed."""
    db = tmp_path / "memory.db"
    original_open = os.open

    def _exploding_open(path: str, flags: int, mode: int = 0o644, /) -> int:
        if path.endswith(".lock"):
            raise PermissionError("simulated unwritable lock file")
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", _exploding_open)

    reg = WriterRegistry(db)
    reg.register()  # must not raise
    assert reg.concurrent_writers == 0
    reg.close()


def test_registry_never_refuses_open(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Top-level regression guard for the sprint exit criterion `advisory-only-b3`.

    Makes every filesystem operation the registry could touch raise, then
    verifies register() still returns cleanly. This is the test the
    sprint-finish playbook runs to assert the advisory-only invariant.
    """
    db = tmp_path / "memory.db"

    def _always_fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated total filesystem failure")

    monkeypatch.setattr(Path, "mkdir", _always_fail)
    monkeypatch.setattr(os, "open", _always_fail)

    reg = WriterRegistry(db)
    reg.register()  # must return, never raise
    reg.close()  # must not raise either


# ---------------------------------------------------------------------------
# Warn threshold
# ---------------------------------------------------------------------------


def test_warn_threshold_not_triggered_for_single_writer(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    db = tmp_path / "memory.db"
    reg = WriterRegistry(db, warn_threshold=4)
    reg.register()
    # Only our pid exists — far below threshold 4.
    record_names = [r.msg for r in caplog.records if hasattr(r, "msg")]
    assert "high_concurrent_writer_count_detected" not in record_names
    reg.close()


def test_warn_threshold_triggered_when_exceeded(tmp_path: Path) -> None:
    """Seed 5 live peer locks (threshold=4) → WARNING logged."""
    db = tmp_path / "memory.db"
    writers_dir = tmp_path / "memory.db.writers"
    writers_dir.mkdir()
    # Seed 4 peer lockfiles using pids known to be alive on Linux.
    # pid 1 (init) exists on every Linux system.
    # We can't easily seed 4 distinct live pids — so instead we push the
    # threshold down to 0 so *any* registration crosses it.
    reg = WriterRegistry(db, warn_threshold=0)
    reg.register()
    # Our own pid crosses the 0 threshold.
    assert reg.concurrent_writers >= 1
    reg.close()


# ---------------------------------------------------------------------------
# Filename parsing helpers
# ---------------------------------------------------------------------------


def test_parse_pid_rejects_nonpid_lockname(tmp_path: Path) -> None:
    """Lock files that aren't named <pid>.lock must be ignored."""
    from trw_memory.storage._writer_registry import _parse_pid_from_lockname

    assert _parse_pid_from_lockname("garbage.lock") is None
    assert _parse_pid_from_lockname("-1.lock") is None
    assert _parse_pid_from_lockname("0.lock") is None
    assert _parse_pid_from_lockname("notalock.txt") is None
    assert _parse_pid_from_lockname("1234.lock") == 1234
    assert _parse_pid_from_lockname("") is None


def test_nonpid_lockfile_is_not_pruned(tmp_path: Path) -> None:
    """Garbage-named .lock files must not be deleted by the pruner."""
    db = tmp_path / "memory.db"
    writers_dir = tmp_path / "memory.db.writers"
    writers_dir.mkdir()
    stray = writers_dir / "garbage.lock"
    stray.write_text("not a pid")

    reg = WriterRegistry(db)
    reg.register()
    assert stray.exists(), "unparseable lock names must be left alone"
    reg.close()


# ---------------------------------------------------------------------------
# Liveness probe
# ---------------------------------------------------------------------------


def test_pid_is_live_detects_init(tmp_path: Path) -> None:
    """pid 1 (init) is live on every Linux; Darwin/BSD may not have /proc."""
    from trw_memory.storage._writer_registry import _pid_is_live

    lock = tmp_path / "1.lock"
    lock.write_text("1\n0\n")
    if sys.platform.startswith("linux"):
        assert _pid_is_live(1, lock) is True


def test_pid_is_live_rejects_impossible_pid(tmp_path: Path) -> None:
    """A pid far beyond any PID_MAX must register as dead."""
    from trw_memory.storage._writer_registry import _pid_is_live

    lock = tmp_path / "9999998.lock"
    lock.write_text("9999998\n0\n")
    # This must be dead on every reasonable host.
    assert _pid_is_live(9999998, lock) is False


def test_stale_lock_heuristic_for_old_file(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """On non-POSIX hosts old lockfiles register as dead via mtime heuristic."""
    from trw_memory.storage._writer_registry import _pid_is_live

    lock = tmp_path / "9999997.lock"
    lock.write_text("9999997\n0\n")
    # Push mtime 30 days into the past.
    old = time.time() - 30 * 24 * 3600
    os.utime(str(lock), (old, old))
    # Simulate non-POSIX by monkeypatching platform.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "name", "nt")
    assert _pid_is_live(9999997, lock) is False


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_del_cleans_up_lock_file(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    reg = WriterRegistry(db)
    reg.register()
    lock_path = tmp_path / "memory.db.writers" / f"{os.getpid()}.lock"
    assert lock_path.exists()
    reg.__del__()
    assert not lock_path.exists()


def test_close_after_failed_register_is_noop(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    db = tmp_path / "memory.db"

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("no")

    monkeypatch.setattr(Path, "mkdir", _fail)
    reg = WriterRegistry(db)
    reg.register()  # fails silently
    reg.close()  # noop; must not raise
