"""Focused timestamped-backup rotation tests for SQLite backend recovery."""

from __future__ import annotations

import inspect
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import structlog
from pydantic import ValidationError

from trw_memory.models.config import MemoryConfig
from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_sqlite_backend_recovery_support import (
    _TIMESTAMPED_BACKUP_RE,
    _freeze_utc,
    _trigger_recovery,
    _write_legacy_backup,
    _write_timestamped_backup,
)


def test_fr01_filename_is_iso8601_utc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_utc(monkeypatch, datetime(2026, 4, 12, 3, 14, 59, tzinfo=timezone.utc))
    backup = _trigger_recovery(tmp_path / "memory.db", monkeypatch)
    assert backup.name == "memory.db.corrupt.2026-04-12T03-14-59Z.bak"
    assert _TIMESTAMPED_BACKUP_RE.fullmatch(backup.name) is not None


def test_fr01_filename_uses_utc_not_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_utc(monkeypatch, datetime(2026, 6, 15, 23, 5, 0, tzinfo=timezone.utc))
    backup = _trigger_recovery(tmp_path / "memory.db", monkeypatch)
    assert backup.name.endswith("Z.bak")
    assert ":" not in backup.name
    assert "2026-06-15T23-05-00Z" in backup.name


def test_fr02_config_field_default() -> None:
    assert MemoryConfig().memory_corrupt_backup_keep == 5


def test_fr02_config_field_bounds() -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(memory_corrupt_backup_keep=0)
    with pytest.raises(ValidationError):
        MemoryConfig(memory_corrupt_backup_keep=51)
    assert MemoryConfig(memory_corrupt_backup_keep=1).memory_corrupt_backup_keep == 1
    assert MemoryConfig(memory_corrupt_backup_keep=50).memory_corrupt_backup_keep == 50


def test_fr02_init_accepts_corrupt_backup_keep_kwarg(tmp_path: Path) -> None:
    init_sig = inspect.signature(SQLiteBackend.__init__)
    assert "corrupt_backup_keep" in init_sig.parameters
    assert init_sig.parameters["corrupt_backup_keep"].default == 5
    backend = SQLiteBackend(tmp_path / "memory.db", corrupt_backup_keep=3)
    try:
        assert backend._corrupt_backup_keep == 3
    finally:
        backend.close()


def test_fr02_keep_n_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for day in range(1, 8):
        _write_timestamped_backup(tmp_path, f"2026-04-{day:02d}T00-00-00Z")
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=5)
    remaining = [p.name for p in tmp_path.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name)]
    assert len(remaining) == 5
    assert all("2026-04-01" not in n and "2026-04-02" not in n for n in remaining)


def test_fr03_prune_oldest_first(tmp_path: Path) -> None:
    t1 = _write_timestamped_backup(tmp_path, "2026-04-10T00-00-00Z")
    t2 = _write_timestamped_backup(tmp_path, "2026-04-11T00-00-00Z")
    t3 = _write_timestamped_backup(tmp_path, "2026-04-12T00-00-00Z")
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=2)
    assert not t1.exists()
    assert t2.exists()
    assert t3.exists()


def test_fr03_prune_uses_filename_not_mtime(tmp_path: Path) -> None:
    t1 = _write_timestamped_backup(tmp_path, "2026-04-10T00-00-00Z")
    t2 = _write_timestamped_backup(tmp_path, "2026-04-11T00-00-00Z")
    t3 = _write_timestamped_backup(tmp_path, "2026-04-12T00-00-00Z")
    now = time.time()
    os.utime(t1, (now, now))
    os.utime(t2, (now - 3600, now - 3600))
    os.utime(t3, (now - 7200, now - 7200))
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=2)
    assert not t1.exists(), "pruning picked wrong victim — it consulted mtime, not filename"
    assert t2.exists()
    assert t3.exists()


def test_fr03_malformed_filename_skipped(tmp_path: Path) -> None:
    malformed = tmp_path / "memory.db.corrupt.notatimestamp.bak"
    malformed.write_bytes(b"garbage")
    for day in range(10, 15):
        _write_timestamped_backup(tmp_path, f"2026-04-{day}T00-00-00Z")
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=5)
    assert malformed.exists()
    remaining = [p for p in tmp_path.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name)]
    assert len(remaining) == 5


def test_fr04_legacy_bak_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = _write_legacy_backup(tmp_path, "memory.db.corrupt.bak", b"legacy-sacred-bytes")
    legacy_content = legacy.read_bytes()
    _trigger_recovery(tmp_path / "memory.db", monkeypatch)
    assert legacy.exists()
    assert legacy.read_bytes() == legacy_content


def test_fr04_legacy_bak_1_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = _write_legacy_backup(tmp_path, "memory.db.corrupt.bak.1", b"legacy-v1-bytes")
    legacy_content = legacy.read_bytes()
    _trigger_recovery(tmp_path / "memory.db", monkeypatch)
    assert legacy.exists()
    assert legacy.read_bytes() == legacy_content


def test_fr04_legacy_counted_but_not_pruned(tmp_path: Path) -> None:
    legacy_0 = _write_legacy_backup(tmp_path, "memory.db.corrupt.bak", b"legacy-0")
    legacy_1 = _write_legacy_backup(tmp_path, "memory.db.corrupt.bak.1", b"legacy-1")
    for day in range(1, 6):
        _write_timestamped_backup(tmp_path, f"2026-04-{day:02d}T00-00-00Z")
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=5)
    assert legacy_0.exists()
    assert legacy_1.exists()
    remaining_ts = [p.name for p in tmp_path.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name)]
    assert len(remaining_ts) == 3
    assert all("2026-04-01" not in n and "2026-04-02" not in n for n in remaining_ts)


def test_fr04_legacy_only_overshoot_warns(tmp_path: Path) -> None:
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak")
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak.1")
    with structlog.testing.capture_logs() as logs:
        SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=1)
    overshoot = [log for log in logs if log.get("event") == "corrupt_backup_budget_exceeded_legacy_only"]
    assert len(overshoot) == 1
    assert overshoot[0]["keep"] == 1
    assert overshoot[0]["legacy_count"] == 2
    assert (tmp_path / "memory.db.corrupt.bak").exists()
    assert (tmp_path / "memory.db.corrupt.bak.1").exists()
