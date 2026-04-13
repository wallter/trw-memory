"""Tests for SQLiteBackend strict-salvage-refusal recovery policy (PRD-CORE-138).

Covers FR01-FR06 and NFR01-NFR04 from docs/requirements-aare-f/prds/PRD-CORE-138.md.
"""

from __future__ import annotations

import inspect
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import structlog
from pydantic import ValidationError

from trw_memory.exceptions import CorruptDatabaseUnsalvageableError, StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend


# ---------------------------------------------------------------------------
# Helpers (mirror patterns in test_db_recovery.py)
# ---------------------------------------------------------------------------


def _make_entry(entry_id: str = "L-0001", content: str = "test content") -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail="detail text",
        tags=["test"],
        importance=0.5,
        status=MemoryStatus.ACTIVE,
        namespace="default",
        source="agent",
        created_at=now,
        updated_at=now,
    )


def _corrupt_sqlite_master(db_path: Path) -> None:
    """Corrupt a DB by overwriting its header + first page.

    This is the failure mode the 2026-04-12 incident produced: sqlite_master
    destroyed, making ``SELECT * FROM memories`` raise ``DatabaseError``, while
    the backup file is structurally non-empty (> page_size).
    """
    data = db_path.read_bytes()
    # Overwrite the SQLite header (first 100 bytes) and the first page but
    # keep later pages intact so the backup remains non-empty.
    corrupted = b"\x00\xff\xfe\xfd" * 512 + data[2048:]
    db_path.write_bytes(corrupted)


def _populate_db(db_path: Path, *, entries: int = 3) -> None:
    """Create a populated SQLite DB so the backup has meaningful bytes."""
    backend = SQLiteBackend(db_path)
    for idx in range(entries):
        backend.store(_make_entry(entry_id=f"L-{idx:04d}", content=f"row {idx}"))
    backend.close()


# ---------------------------------------------------------------------------
# FR01 — exception class
# ---------------------------------------------------------------------------


def test_fr01_exception_class_exists_and_carries_backup_path(tmp_path: Path) -> None:
    """FR01: CorruptDatabaseUnsalvageableError embeds backup path in str(exc)."""
    backup = tmp_path / "memory.db.corrupt.bak"
    backup.write_bytes(b"\x00" * 8192)

    exc = CorruptDatabaseUnsalvageableError("salvage failed", backup_path=str(backup))

    assert str(backup) in str(exc)
    assert exc.backup_path == str(backup)
    # Base StorageError stores it on `.path` as well for consistency with peers.
    assert exc.path == str(backup)


def test_fr01_exception_is_storage_error_subclass() -> None:
    """FR01: new exception is a StorageError subclass."""
    assert issubclass(CorruptDatabaseUnsalvageableError, StorageError)


# ---------------------------------------------------------------------------
# FR02 — config field
# ---------------------------------------------------------------------------


def test_fr02_config_default_is_strict() -> None:
    """FR02: MemoryConfig().memory_recovery_policy == 'strict' by default."""
    config = MemoryConfig()
    assert config.memory_recovery_policy == "strict"


def test_fr02_config_rejects_invalid_value() -> None:
    """FR02: Pydantic rejects values outside Literal['strict','empty_ok']."""
    with pytest.raises(ValidationError):
        MemoryConfig(memory_recovery_policy="yolo")  # type: ignore[arg-type]


def test_fr02_config_accepts_empty_ok() -> None:
    """FR02: 'empty_ok' is accepted as the escape-hatch alternative."""
    config = MemoryConfig(memory_recovery_policy="empty_ok")
    assert config.memory_recovery_policy == "empty_ok"


# ---------------------------------------------------------------------------
# FR03 — strict refusal on sqlite_master destruction
# ---------------------------------------------------------------------------


def test_fr03_strict_refuses_silent_empty_on_destroyed_sqlite_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR03: strict mode raises CorruptDatabaseUnsalvageableError when salvage yields 0 rows."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=3)
    _corrupt_sqlite_master(db_path)

    # Force the .recover CLI path to return empty (second salvage path unavailable).
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: []),
    )

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")

    # Backup preserved intact on disk; no new DB written at db_path.
    backup_path = db_path.with_suffix(".db.corrupt.bak")
    assert backup_path.exists()
    assert backup_path.stat().st_size > 4096  # backup is non-empty
    assert not db_path.exists()  # no new DB was created on strict refusal


def test_fr03_strict_refusal_exception_contains_backup_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR03: exception message includes the absolute backup path."""
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

    backup_path = db_path.with_suffix(".db.corrupt.bak")
    assert str(backup_path) in str(exc_info.value)
    assert exc_info.value.backup_path == str(backup_path)


# ---------------------------------------------------------------------------
# FR04 — sqlite3 .recover CLI fallback
# ---------------------------------------------------------------------------


def test_fr04_recover_cli_salvage_succeeds_when_select_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR04: sqlite3 .recover CLI path restores rows when in-process SELECT fails."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)

    # Snapshot raw bytes to synthesize a CLI-recover success scenario later
    _corrupt_sqlite_master(db_path)

    # Craft a fake CLI dump that creates a memories table with one row.
    # Full DDL is brittle — return a populated row from _salvage_via_recover_cli directly
    # (the test proves the recovery path uses CLI rows, not that subprocess formatting is right).
    sentinel_row = sqlite3.Row
    # We can't easily construct sqlite3.Row instances without a cursor; mock with a
    # lightweight object exposing .keys() and iteration support like sqlite3.Row.
    class _FakeRow:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def keys(self) -> list[str]:
            return list(self._data.keys())

        def __iter__(self) -> Any:
            return iter(self._data.values())

    # Build a row with all NOT NULL fields populated (id, content, created_at, updated_at).
    now = "2026-04-13T00:00:00+00:00"
    fake_row = _FakeRow(
        {
            "id": "L-rescued-via-cli",
            "content": "rescued content",
            "created_at": now,
            "updated_at": now,
        }
    )
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: [fake_row]),
    )

    # Should NOT raise — CLI salvage returned rows.
    conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict")
    try:
        count_row = conn.execute("SELECT count(*) FROM memories").fetchone()
        assert count_row[0] == 1
        id_row = conn.execute("SELECT id FROM memories").fetchone()
        assert id_row[0] == "L-rescued-via-cli"
    finally:
        conn.close()


def test_fr04_recover_cli_unavailable_falls_through_to_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR04: CLI unavailable (FileNotFoundError) falls through to strict refusal."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _raise_fnf(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("sqlite3 not installed")

    monkeypatch.setattr(subprocess, "run", _raise_fnf)

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


def test_fr04_recover_cli_timeout_falls_through_to_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR04: CLI timeout falls through to strict-mode refusal."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _raise_timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="sqlite3", timeout=30)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


def test_fr04_recover_cli_nonzero_exit_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR04: CLI non-zero exit code is treated as salvage failure."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _nonzero(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=["sqlite3"], returncode=1, stdout=b"", stderr=b"err")

    monkeypatch.setattr(subprocess, "run", _nonzero)

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


def test_fr04_recover_cli_empty_dump_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR04: CLI returns 0 exit but empty stdout → strict refusal."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _empty_stdout(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=["sqlite3"], returncode=0, stdout=b"   \n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _empty_stdout)

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


# ---------------------------------------------------------------------------
# FR05 — empty_ok preserves legacy behavior
# ---------------------------------------------------------------------------


def test_fr05_empty_ok_preserves_legacy_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR05: empty_ok policy preserves pre-PRD silent-empty behavior."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    # Force CLI salvage to also fail (return empty).
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: []),
    )

    # Under empty_ok, no exception is raised; a fresh empty DB is created.
    conn = SQLiteBackend.recover_db(db_path, recovery_policy="empty_ok")
    try:
        # Fresh DB: schema exists, no rows.
        count_row = conn.execute("SELECT count(*) FROM memories").fetchone()
        assert count_row[0] == 0
    finally:
        conn.close()

    # Backup file is still on disk (legacy behavior preserves backups too).
    assert db_path.with_suffix(".db.corrupt.bak").exists()
    # New DB was created at the original path.
    assert db_path.exists()


def test_fr05_empty_ok_logs_rows_salvaged_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR05: empty_ok fallback logs db_recovered with rows_salvaged=0 (legacy WARNING)."""
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


# ---------------------------------------------------------------------------
# FR06 — policy threaded through __init__
# ---------------------------------------------------------------------------


def test_fr06_init_accepts_recovery_policy_kwarg(tmp_path: Path) -> None:
    """FR06: recovery_policy kwarg present on __init__ with strict default."""
    init_sig = inspect.signature(SQLiteBackend.__init__)
    assert "recovery_policy" in init_sig.parameters
    assert init_sig.parameters["recovery_policy"].default == "strict"


def test_fr06_recover_db_accepts_recovery_policy_kwarg() -> None:
    """FR06: recovery_policy kwarg present on recover_db with strict default."""
    sig = inspect.signature(SQLiteBackend.recover_db)
    assert "recovery_policy" in sig.parameters
    assert sig.parameters["recovery_policy"].default == "strict"


def test_fr06_policy_threaded_through_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR06: recovery_policy kwarg flows from __init__ to recover_db."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: []),
    )

    # Default (strict) → raises.
    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend(db_path)

    # Rebuild the corrupted state (the strict path left the backup but no DB).
    # Move backup back so we can test empty_ok reopening.
    backup_path = db_path.with_suffix(".db.corrupt.bak")
    assert backup_path.exists()
    import shutil

    shutil.copy(str(backup_path), str(db_path))

    # Remove the old .corrupt.bak so rotation doesn't preserve stale state
    # and the strict path signal matches empty_ok expectations.
    # (With empty_ok: constructor completes, recovered=True.)
    backend = SQLiteBackend(db_path, recovery_policy="empty_ok")
    assert backend.recovered is True
    backend.close()


# ---------------------------------------------------------------------------
# NFR01 — healthy path unchanged
# ---------------------------------------------------------------------------


def test_nfr01_healthy_open_path_unchanged_latency(tmp_path: Path) -> None:
    """NFR01: healthy DB open path is unchanged by the new recovery_policy kwarg."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    try:
        assert backend.recovered is False
        assert backend.integrity_warning is False
        # Default policy is "strict" — healthy paths ignore this entirely.
        assert backend._recovery_policy == "strict"
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# NFR03 — subprocess shell-safety
# ---------------------------------------------------------------------------


def test_nfr03_subprocess_called_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NFR03: subprocess.run invocation passes args as a list, never shell=True."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    recorded: dict[str, Any] = {}

    def _record(*args: Any, **kwargs: Any) -> Any:
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        # Return a benign non-match so the rest of the flow continues.
        return subprocess.CompletedProcess(args=args[0] if args else [], returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _record)

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")

    # Positional args: first arg is the command LIST, not a string.
    cmd = recorded["args"][0]
    assert isinstance(cmd, list)
    assert cmd[0] == "sqlite3"
    # No shell=True anywhere.
    assert recorded["kwargs"].get("shell", False) is False


# ---------------------------------------------------------------------------
# NFR04 — structured log on strict refusal
# ---------------------------------------------------------------------------


def test_nfr04_strict_refusal_emits_structured_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NFR04: strict-refusal path emits db_recovery_refused_strict at ERROR level."""
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
    assert entry["backup_path"] == str(db_path.with_suffix(".db.corrupt.bak"))
    assert entry["backup_size_bytes"] > 4096
    assert entry["salvage_primary_failed"] is True
    assert entry["salvage_cli_failed"] is True


# ---------------------------------------------------------------------------
# Regression — 2026-04-12 silent-empty fallback
# ---------------------------------------------------------------------------


def test_regression_2026_04_12_silent_empty_fallback_now_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: reproduce the 2026-04-12 failure mode; strict default must raise.

    This is the canary test for the incident this PRD addresses. Before PRD-CORE-138,
    destroying sqlite_master silently produced an empty DB. Now strict default raises.
    """
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=5)
    _corrupt_sqlite_master(db_path)

    # Force CLI salvage unavailable to mirror the 2026-04-12 environment
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: []),
    )

    with pytest.raises(CorruptDatabaseUnsalvageableError) as exc_info:
        SQLiteBackend(db_path)  # default config → strict

    # Evidence preserved.
    assert exc_info.value.backup_path != ""
    assert Path(exc_info.value.backup_path).exists()


def test_regression_healthy_db_open_unchanged(tmp_path: Path) -> None:
    """Regression: healthy DB open still succeeds with no recovery triggered."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    assert backend.recovered is False
    backend.store(_make_entry("L-ok", "healthy"))
    backend.close()

    # Reopen — still healthy.
    backend2 = SQLiteBackend(db_path)
    assert backend2.recovered is False
    assert backend2.integrity_warning is False
    entry = backend2.get("L-ok")
    assert entry is not None
    assert entry.content == "healthy"
    backend2.close()


# ---------------------------------------------------------------------------
# Negative paths
# ---------------------------------------------------------------------------


def test_negative_empty_backup_still_produces_empty_db_under_strict(tmp_path: Path) -> None:
    """If the corrupt backup is tiny (<= page_size), strict mode falls through to empty DB."""
    db_path = tmp_path / "memory.db"
    # Write total garbage smaller than 4096 bytes → backup will be < page_size.
    db_path.write_bytes(b"\x00" * 2048)

    # Strict mode should NOT raise on a genuinely empty backup.
    conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict")
    try:
        count_row = conn.execute("SELECT count(*) FROM memories").fetchone()
        assert count_row[0] == 0
    finally:
        conn.close()
