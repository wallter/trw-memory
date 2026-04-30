"""Focused policy tests for SQLite backend recovery."""

from __future__ import annotations

import inspect
import shutil
import sqlite3
from pathlib import Path

import pytest
import structlog
from pydantic import ValidationError

from trw_memory.exceptions import CorruptDatabaseUnsalvageableError, StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_sqlite_backend_recovery_support import (
    _corrupt_sqlite_master,
    _find_timestamped_backup,
    _make_entry,
    _populate_db,
)


def test_fr01_exception_class_exists_and_carries_backup_path(tmp_path: Path) -> None:
    backup = tmp_path / "memory.db.corrupt.bak"
    backup.write_bytes(b"\x00" * 8192)
    exc = CorruptDatabaseUnsalvageableError("salvage failed", backup_path=str(backup))
    assert str(backup) in str(exc)
    assert exc.backup_path == str(backup)
    assert exc.path == str(backup)


def test_fr01_exception_is_storage_error_subclass() -> None:
    assert issubclass(CorruptDatabaseUnsalvageableError, StorageError)


def test_fr02_config_default_is_strict() -> None:
    assert MemoryConfig().memory_recovery_policy == "strict"


def test_fr02_config_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(memory_recovery_policy="yolo")  # type: ignore[arg-type]


def test_fr02_config_accepts_empty_ok() -> None:
    assert MemoryConfig(memory_recovery_policy="empty_ok").memory_recovery_policy == "empty_ok"


def test_fr03_strict_refuses_silent_empty_on_destroyed_sqlite_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=3)
    _corrupt_sqlite_master(db_path)
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: []),
    )
    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")
    backup_path = _find_timestamped_backup(tmp_path)
    assert backup_path.exists()
    assert backup_path.stat().st_size > 4096
    assert not db_path.exists()


def test_fr03_strict_refusal_exception_contains_backup_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: []),
    )
    with pytest.raises(CorruptDatabaseUnsalvageableError) as exc_info:
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")
    backup_path = _find_timestamped_backup(tmp_path)
    assert str(backup_path) in str(exc_info.value)
    assert exc_info.value.backup_path == str(backup_path)


def test_fr05_empty_ok_preserves_legacy_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: []),
    )
    conn = SQLiteBackend.recover_db(db_path, recovery_policy="empty_ok")
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
    finally:
        conn.close()
    assert _find_timestamped_backup(tmp_path).exists()
    assert db_path.exists()


def test_fr05_empty_ok_logs_rows_salvaged_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert events[0]["rows_salvaged"] == 0


def test_fr06_init_accepts_recovery_policy_kwarg(tmp_path: Path) -> None:
    init_sig = inspect.signature(SQLiteBackend.__init__)
    assert "recovery_policy" in init_sig.parameters
    assert init_sig.parameters["recovery_policy"].default == "strict"


def test_fr06_recover_db_accepts_recovery_policy_kwarg() -> None:
    sig = inspect.signature(SQLiteBackend.recover_db)
    assert "recovery_policy" in sig.parameters
    assert sig.parameters["recovery_policy"].default == "strict"


def test_fr06_policy_threaded_through_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: []),
    )
    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend(db_path)
    backup_path = _find_timestamped_backup(tmp_path)
    assert backup_path.exists()
    shutil.copy(str(backup_path), str(db_path))
    backend = SQLiteBackend(db_path, recovery_policy="empty_ok")
    assert backend.recovered is True
    backend.close()


def test_nfr01_healthy_open_path_unchanged_latency(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    try:
        assert backend.recovered is False
        assert backend.integrity_warning is False
        assert backend._recovery_policy == "strict"
    finally:
        backend.close()


def test_nfr04_strict_refusal_emits_structured_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: []),
    )
    with structlog.testing.capture_logs() as logs:
        with pytest.raises(CorruptDatabaseUnsalvageableError):
            SQLiteBackend.recover_db(db_path, recovery_policy="strict")
    refusal_events = [log for log in logs if log.get("event") == "db_recovery_refused_strict"]
    assert len(refusal_events) == 1
    entry = refusal_events[0]
    assert entry["log_level"] == "error"
    assert entry["action"] == "refuse_empty_fallback"
    assert entry["db_path"] == str(db_path)
    assert entry["backup_path"] == str(_find_timestamped_backup(tmp_path))
    assert entry["backup_size_bytes"] > 4096
    assert entry["salvage_primary_failed"] is True
    assert entry["salvage_cli_failed"] is True


def test_regression_2026_04_12_silent_empty_fallback_now_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=5)
    _corrupt_sqlite_master(db_path)
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: []),
    )
    with pytest.raises(CorruptDatabaseUnsalvageableError) as exc_info:
        SQLiteBackend(db_path)
    assert exc_info.value.backup_path != ""
    assert Path(exc_info.value.backup_path).exists()


def test_regression_healthy_db_open_unchanged(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.store(_make_entry("L-ok", "healthy"))
    backend.close()
    backend2 = SQLiteBackend(db_path)
    assert backend2.recovered is False
    assert backend2.integrity_warning is False
    entry = backend2.get("L-ok")
    assert entry is not None
    assert entry.content == "healthy"
    backend2.close()


def test_negative_empty_backup_still_produces_empty_db_under_strict(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"\x00" * 2048)
    conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict")
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
    finally:
        conn.close()
