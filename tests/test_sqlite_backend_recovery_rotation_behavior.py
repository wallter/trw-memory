"""Behavior, integration, and observability tests for recovery rotation."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import structlog

from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_sqlite_backend_recovery_support import (
    _TIMESTAMPED_BACKUP_RE,
    _corrupt_sqlite_master,
    _find_timestamped_backup,
    _freeze_utc,
    _make_fake_row,
    _populate_db,
    _trigger_recovery,
    _write_legacy_backup,
    _write_timestamped_backup,
)


def test_fr05_salvage_semantics_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: []),
    )
    with structlog.testing.capture_logs() as logs:
        conn = SQLiteBackend.recover_db(db_path, recovery_policy="empty_ok")
        conn.close()
    events = [log for log in logs if log.get("event") == "db_recovered"]
    assert len(events) == 1
    entry = events[0]
    assert entry["rows_salvaged"] == 0
    assert "backup" in entry
    assert "db" in entry
    assert _TIMESTAMPED_BACKUP_RE.fullmatch(Path(entry["backup"]).name) is not None


def test_nfr01_rotation_latency_bounded(tmp_path: Path) -> None:
    for i in range(50):
        _write_timestamped_backup(tmp_path, f"2026-04-10T00-00-{i % 60:02d}Z-{i}")
    start = time.perf_counter()
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 100, f"pruning took {elapsed_ms:.2f}ms (target <20ms, hard cap 100ms)"


def test_nfr02_filename_collision_same_second(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_utc(monkeypatch, datetime(2026, 4, 12, 3, 14, 59, tzinfo=timezone.utc))
    db_1 = tmp_path / "memory.db"
    db_1.write_bytes(b"dummy-1" * 100)
    first = SQLiteBackend._rotate_corrupt_backup(db_1)
    db_2 = tmp_path / "memory.db"
    db_2.write_bytes(b"dummy-2" * 100)
    second = SQLiteBackend._rotate_corrupt_backup(db_2)
    assert first.name == "memory.db.corrupt.2026-04-12T03-14-59Z.bak"
    assert second.name == "memory.db.corrupt.2026-04-12T03-14-59Z-1.bak"
    assert _TIMESTAMPED_BACKUP_RE.fullmatch(first.name) is not None
    assert _TIMESTAMPED_BACKUP_RE.fullmatch(second.name) is not None


def test_nfr02_is_idempotent(tmp_path: Path) -> None:
    for day in range(1, 6):
        _write_timestamped_backup(tmp_path, f"2026-04-{day:02d}T00-00-00Z")
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=5)
    first_pass = sorted(p.name for p in tmp_path.iterdir())
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=5)
    second_pass = sorted(p.name for p in tmp_path.iterdir())
    assert first_pass == second_pass


def test_nfr04_rotation_emits_structured_log(tmp_path: Path) -> None:
    for day in range(1, 4):
        _write_timestamped_backup(tmp_path, f"2026-04-{day:02d}T00-00-00Z")
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak")
    with structlog.testing.capture_logs() as logs:
        SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=2)
    events = [log for log in logs if log.get("event") == "corrupt_backup_rotated"]
    assert len(events) == 1
    entry = events[0]
    assert entry["log_level"] == "info"
    assert entry["keep"] == 2
    assert entry["total_legacy"] == 1
    assert entry["total_timestamped"] == 1
    assert entry["pruned"] == 2
    assert entry["parent"] == str(tmp_path)


def test_negative_unlink_permission_error_is_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for day in range(1, 4):
        _write_timestamped_backup(tmp_path, f"2026-04-{day:02d}T00-00-00Z")
    original_unlink = Path.unlink

    def _raise_perm(self: Path, *args: Any, **kwargs: Any) -> None:
        if "2026-04-01" in self.name:
            raise PermissionError("denied")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _raise_perm)
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=1)


def test_negative_keep_equals_1_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_utc(monkeypatch, datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc))
    for day in range(1, 4):
        _write_timestamped_backup(tmp_path, f"2026-04-{day:02d}T00-00-00Z")
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=1)
    remaining = [p.name for p in tmp_path.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name)]
    assert len(remaining) == 1
    assert remaining[0] == "memory.db.corrupt.2026-04-03T00-00-00Z.bak"


def test_integration_end_to_end_7_recoveries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_row = _make_fake_row(
        {
            "id": "L-seq",
            "content": "seq",
            "created_at": "2026-04-13T00:00:00+00:00",
            "updated_at": "2026-04-13T00:00:00+00:00",
        }
    )
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: [fake_row]),
    )
    db_path = tmp_path / "memory.db"
    for i in range(7):
        _freeze_utc(monkeypatch, datetime(2026, 4, 12, 3, 14, 50 + i, tzinfo=timezone.utc))
        _populate_db(db_path, entries=1)
        _corrupt_sqlite_master(db_path)
        conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict", corrupt_backup_keep=5)
        conn.close()
    remaining = sorted(p.name for p in tmp_path.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name))
    assert len(remaining) == 5
    assert all("14-50Z" not in n and "14-51Z" not in n for n in remaining)


def test_integration_mixed_legacy_and_new(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak", b"legacy-0")
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak.1", b"legacy-1")
    fake_row = _make_fake_row(
        {
            "id": "L-m",
            "content": "m",
            "created_at": "2026-04-13T00:00:00+00:00",
            "updated_at": "2026-04-13T00:00:00+00:00",
        }
    )
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: [fake_row]),
    )
    db_path = tmp_path / "memory.db"
    for i in range(4):
        _freeze_utc(monkeypatch, datetime(2026, 4, 12, 3, 14, 50 + i, tzinfo=timezone.utc))
        _populate_db(db_path, entries=1)
        _corrupt_sqlite_master(db_path)
        conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict", corrupt_backup_keep=5)
        conn.close()
    assert (tmp_path / "memory.db.corrupt.bak").read_bytes() == b"legacy-0"
    assert (tmp_path / "memory.db.corrupt.bak.1").read_bytes() == b"legacy-1"
    remaining_ts = [p for p in tmp_path.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name)]
    assert len(remaining_ts) == 3


def test_regression_2_wide_overwrite_no_longer_occurs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy_0 = _write_legacy_backup(tmp_path, "memory.db.corrupt.bak", b"first-event")
    legacy_1 = _write_legacy_backup(tmp_path, "memory.db.corrupt.bak.1", b"second-event")
    _trigger_recovery(tmp_path / "memory.db", monkeypatch)
    assert legacy_0.read_bytes() == b"first-event"
    assert legacy_1.read_bytes() == b"second-event"
    assert _find_timestamped_backup(tmp_path).exists()


def test_regression_healthy_open_path_no_rotation(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "memory.db")
    backend.close()
    corrupt_files = [p for p in tmp_path.iterdir() if p.name.startswith("memory.db.corrupt.")]
    assert corrupt_files == []


def test_migration_upgrade_preserves_legacy_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak", b"pre-upgrade-0")
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak.1", b"pre-upgrade-1")
    _trigger_recovery(tmp_path / "memory.db", monkeypatch)
    assert (tmp_path / "memory.db.corrupt.bak").read_bytes() == b"pre-upgrade-0"
    assert (tmp_path / "memory.db.corrupt.bak.1").read_bytes() == b"pre-upgrade-1"
    assert _find_timestamped_backup(tmp_path).exists()
