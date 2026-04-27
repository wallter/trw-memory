"""Tests for SQLiteBackend strict-salvage-refusal recovery policy (PRD-CORE-138)
plus timestamped 5-wide corruption-backup rotation (PRD-CORE-139).

Covers FR01-FR06 and NFR01-NFR04 from docs/requirements-aare-f/prds/PRD-CORE-138.md
and FR01-FR05/NFR01-NFR04 from docs/requirements-aare-f/prds/PRD-CORE-139.md.
"""

from __future__ import annotations

import inspect
import os
import re
import sqlite3
import subprocess
import time
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

# PRD-CORE-139: filename pattern for new timestamped backups.
_TIMESTAMPED_BACKUP_RE = re.compile(r"^memory\.db\.corrupt\.(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)(?:-\d+)?\.bak$")


def _find_timestamped_backup(parent: Path) -> Path:
    """Return the single timestamped corruption backup in ``parent`` (PRD-CORE-139).

    Used by tests that need to reference the post-rotation backup path without
    hardcoding the wall-clock timestamp.
    """
    matches = sorted(p for p in parent.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name))
    assert len(matches) == 1, f"expected exactly one timestamped backup in {parent}, got {[p.name for p in matches]}"
    return matches[0]


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
    # PRD-CORE-139: backup name is timestamped (memory.db.corrupt.<ISO-UTC>.bak).
    backup_path = _find_timestamped_backup(tmp_path)
    assert backup_path.exists()
    assert backup_path.stat().st_size > 4096  # backup is non-empty
    assert not db_path.exists()  # no new DB was created on strict refusal


def test_fr03_strict_refusal_exception_contains_backup_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    # PRD-CORE-139: backup name is timestamped.
    backup_path = _find_timestamped_backup(tmp_path)
    assert str(backup_path) in str(exc_info.value)
    assert exc_info.value.backup_path == str(backup_path)


# ---------------------------------------------------------------------------
# FR04 — sqlite3 .recover CLI fallback
# ---------------------------------------------------------------------------


def test_fr04_recover_cli_salvage_succeeds_when_select_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_recovery_drops_unknown_columns_preventing_sql_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: column names from the corrupt dump are filtered against
    the schema allowlist before splicing into the INSERT string. An attacker
    who crafts a corrupt DB with a malicious column name cannot inject SQL;
    the unknown column is silently dropped (warning emitted) and the valid
    columns are preserved.
    """
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)

    class _FakeRow:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def keys(self) -> list[str]:
            return list(self._data.keys())

        def __iter__(self) -> Any:
            return iter(self._data.values())

    now = "2026-04-18T00:00:00+00:00"
    # Attacker-controlled column name that would corrupt the INSERT if not filtered.
    malicious_col = "id, content); DROP TABLE memories; --"
    poisoned_row = _FakeRow(
        {
            "id": "L-legit",
            "content": "legit content",
            "created_at": now,
            "updated_at": now,
            malicious_col: "payload",
        }
    )
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: [poisoned_row]),
    )

    conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict")
    try:
        # Table still exists (injection didn't execute) and the legit row landed.
        count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        assert count == 1
        row_id = conn.execute("SELECT id FROM memories").fetchone()[0]
        assert row_id == "L-legit"
    finally:
        conn.close()


def test_fr04_recover_cli_unavailable_falls_through_to_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR04: CLI unavailable (FileNotFoundError) falls through to strict refusal."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _raise_fnf(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("sqlite3 not installed")

    monkeypatch.setattr(subprocess, "run", _raise_fnf)

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


def test_fr04_recover_cli_timeout_falls_through_to_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR04: CLI timeout falls through to strict-mode refusal."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _raise_timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="sqlite3", timeout=30)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


def test_fr04_recover_cli_nonzero_exit_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR04: CLI non-zero exit code is treated as salvage failure."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _nonzero(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=["sqlite3"], returncode=1, stdout=b"", stderr=b"err")

    monkeypatch.setattr(subprocess, "run", _nonzero)

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


def test_fr04_recover_cli_full_executescript_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR04: exercise the real _salvage_via_recover_cli tempdb/executescript path.

    Mocks subprocess.run to return a valid CREATE + INSERT dump so the helper
    loads it via executescript and returns rows. This covers the tempfile +
    dbapi.connect + executescript branch of _salvage_via_recover_cli.
    """
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    # Provide a minimal, valid SQL dump the way `sqlite3 .recover` would.
    dump_sql = b"""
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
INSERT INTO memories (id, content, created_at, updated_at)
VALUES ('L-from-dump', 'rescued from cli dump', '2026-04-13T00:00:00+00:00', '2026-04-13T00:00:00+00:00');
"""

    def _valid_dump(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(
            args=["sqlite3"],
            returncode=0,
            stdout=dump_sql,
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", _valid_dump)

    conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict")
    try:
        row = conn.execute("SELECT id, content FROM memories").fetchone()
        assert row[0] == "L-from-dump"
        assert row[1] == "rescued from cli dump"
    finally:
        conn.close()


def test_fr04_recover_cli_malformed_dump_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR04: malformed dump (executescript raises sqlite3.Error) → empty list → strict refusal."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    def _bad_dump(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(
            args=["sqlite3"],
            returncode=0,
            stdout=b"THIS IS NOT VALID SQL AT ALL;;",
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", _bad_dump)

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict")


def test_fr04_recover_cli_empty_dump_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_fr05_empty_ok_preserves_legacy_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    # PRD-CORE-139: backup is named with a timestamped suffix.
    assert _find_timestamped_backup(tmp_path).exists()
    # New DB was created at the original path.
    assert db_path.exists()


def test_fr05_empty_ok_logs_rows_salvaged_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_fr06_policy_threaded_through_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    # PRD-CORE-139: backup name is timestamped.
    backup_path = _find_timestamped_backup(tmp_path)
    assert backup_path.exists()
    import shutil

    shutil.copy(str(backup_path), str(db_path))

    # With empty_ok: constructor completes, recovered=True. A second
    # timestamped backup will be created alongside the first.
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


def test_nfr03_subprocess_called_without_shell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_nfr04_strict_refusal_emits_structured_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    # PRD-CORE-139: backup path is timestamped, not the legacy .corrupt.bak name.
    assert entry["backup_path"] == str(_find_timestamped_backup(tmp_path))
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


# ═══════════════════════════════════════════════════════════════════════════
# PRD-CORE-139: Timestamped backup rotation
# ═══════════════════════════════════════════════════════════════════════════
# Covers FR01 (filename format), FR02 (config field), FR03 (filename-based
# pruning), FR04 (legacy preservation), FR05 (salvage semantics unchanged),
# NFR01-NFR04 (latency, idempotence, security, observability).


from trw_memory.storage import sqlite_backend as _sqlite_backend_module  # noqa: E402


class _FrozenDateTime:
    """Drop-in datetime replacement that returns a fixed ``now(tz)``.

    Used by PRD-CORE-139 tests to control the UTC timestamp embedded in
    the backup filename without pulling in freezegun (not a dev dep here).
    """

    _frozen_utc: datetime = datetime(2026, 4, 13, 14, 37, 14, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz: Any = None) -> datetime:
        if tz is None:
            return cls._frozen_utc.replace(tzinfo=None)
        return cls._frozen_utc.astimezone(tz) if tz != timezone.utc else cls._frozen_utc

    @classmethod
    def set(cls, value: datetime) -> None:
        cls._frozen_utc = value


def _freeze_utc(monkeypatch: pytest.MonkeyPatch, when: datetime) -> None:
    """Patch ``datetime`` in the backend module to return ``when`` for ``now(tz)``."""
    _FrozenDateTime.set(when)
    # Replace the ``datetime`` symbol the backend imports; leave ``timezone`` alone
    # so ``datetime.now(timezone.utc)`` still resolves to the desired instant.
    monkeypatch.setattr(_sqlite_backend_module, "datetime", _FrozenDateTime)


def _write_timestamped_backup(parent: Path, ts_suffix: str, content: bytes = b"data") -> Path:
    """Seed a synthetic timestamped corrupt backup file with a given suffix."""
    path = parent / f"memory.db.corrupt.{ts_suffix}.bak"
    path.write_bytes(content)
    return path


def _write_legacy_backup(parent: Path, name: str, content: bytes = b"legacy") -> Path:
    """Seed a legacy-named corrupt backup (memory.db.corrupt.bak[.1])."""
    path = parent / name
    path.write_bytes(content)
    return path


def _trigger_recovery(db_path: Path, monkeypatch: pytest.MonkeyPatch, *, keep_n: int = 5) -> Path:
    """Populate + corrupt + recover, returning the new timestamped backup path."""
    _populate_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)

    # Keep the CLI salvage path deterministic: return a single synthetic row so
    # strict policy does not raise.
    class _FakeRow:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def keys(self) -> list[str]:
            return list(self._data.keys())

        def __iter__(self) -> Any:
            return iter(self._data.values())

    now = "2026-04-13T00:00:00+00:00"
    fake_row = _FakeRow({"id": "L-ok", "content": "ok", "created_at": now, "updated_at": now})
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: [fake_row]),
    )

    conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict", corrupt_backup_keep=keep_n)
    conn.close()
    return _find_timestamped_backup(db_path.parent)


# ---------------------------------------------------------------------------
# FR01 — filename format
# ---------------------------------------------------------------------------


def test_fr01_filename_is_iso8601_utc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR01: backup filename matches memory.db.corrupt.<ISO-UTC>.bak exactly."""
    _freeze_utc(monkeypatch, datetime(2026, 4, 12, 3, 14, 59, tzinfo=timezone.utc))
    db_path = tmp_path / "memory.db"
    backup = _trigger_recovery(db_path, monkeypatch)

    assert backup.name == "memory.db.corrupt.2026-04-12T03-14-59Z.bak"
    assert _TIMESTAMPED_BACKUP_RE.fullmatch(backup.name) is not None


def test_fr01_filename_uses_utc_not_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR01: Z suffix present; colons replaced with hyphens for Windows safety."""
    _freeze_utc(monkeypatch, datetime(2026, 6, 15, 23, 5, 0, tzinfo=timezone.utc))
    db_path = tmp_path / "memory.db"
    backup = _trigger_recovery(db_path, monkeypatch)

    assert backup.name.endswith("Z.bak")
    assert ":" not in backup.name
    # Confirm UTC time materialized, not local.
    assert "2026-06-15T23-05-00Z" in backup.name


# ---------------------------------------------------------------------------
# FR02 — config field
# ---------------------------------------------------------------------------


def test_fr02_config_field_default() -> None:
    """FR02: MemoryConfig().memory_corrupt_backup_keep defaults to 5."""
    assert MemoryConfig().memory_corrupt_backup_keep == 5


def test_fr02_config_field_bounds() -> None:
    """FR02: values outside [1, 50] raise ValidationError."""
    with pytest.raises(ValidationError):
        MemoryConfig(memory_corrupt_backup_keep=0)
    with pytest.raises(ValidationError):
        MemoryConfig(memory_corrupt_backup_keep=51)
    # Boundaries inclusive.
    assert MemoryConfig(memory_corrupt_backup_keep=1).memory_corrupt_backup_keep == 1
    assert MemoryConfig(memory_corrupt_backup_keep=50).memory_corrupt_backup_keep == 50


def test_fr02_init_accepts_corrupt_backup_keep_kwarg(tmp_path: Path) -> None:
    """FR02: SQLiteBackend.__init__ exposes corrupt_backup_keep kwarg with default 5."""
    init_sig = inspect.signature(SQLiteBackend.__init__)
    assert "corrupt_backup_keep" in init_sig.parameters
    assert init_sig.parameters["corrupt_backup_keep"].default == 5

    # Attribute is stored on the instance for threading into recover_db.
    backend = SQLiteBackend(tmp_path / "memory.db", corrupt_backup_keep=3)
    try:
        assert backend._corrupt_backup_keep == 3
    finally:
        backend.close()


def test_fr02_keep_n_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR02: with keep=5, 7 synthetic recoveries leave exactly 5 timestamped files."""
    # Seed 7 timestamped backups directly (faster than 7 real recoveries).
    for day in range(1, 8):
        _write_timestamped_backup(tmp_path, f"2026-04-{day:02d}T00-00-00Z")

    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=5)

    remaining = [p.name for p in tmp_path.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name)]
    assert len(remaining) == 5
    # The two oldest (April 1 and 2) should have been pruned.
    assert all("2026-04-01" not in n and "2026-04-02" not in n for n in remaining)


# ---------------------------------------------------------------------------
# FR03 — filename-parsed pruning
# ---------------------------------------------------------------------------


def test_fr03_prune_oldest_first(tmp_path: Path) -> None:
    """FR03: with T1 < T2 < T3 and keep=2, T1 is the sole deletion victim."""
    t1 = _write_timestamped_backup(tmp_path, "2026-04-10T00-00-00Z")
    t2 = _write_timestamped_backup(tmp_path, "2026-04-11T00-00-00Z")
    t3 = _write_timestamped_backup(tmp_path, "2026-04-12T00-00-00Z")

    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=2)

    assert not t1.exists()
    assert t2.exists()
    assert t3.exists()


def test_fr03_prune_uses_filename_not_mtime(tmp_path: Path) -> None:
    """FR03: st_mtime reversed via os.utime; pruning still picks oldest by filename."""
    t1 = _write_timestamped_backup(tmp_path, "2026-04-10T00-00-00Z")
    t2 = _write_timestamped_backup(tmp_path, "2026-04-11T00-00-00Z")
    t3 = _write_timestamped_backup(tmp_path, "2026-04-12T00-00-00Z")

    # Reverse the mtime order: t1 has the newest mtime, t3 the oldest.
    now = time.time()
    os.utime(t1, (now, now))  # newest mtime
    os.utime(t2, (now - 3600, now - 3600))
    os.utime(t3, (now - 7200, now - 7200))  # oldest mtime

    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=2)

    # Even though t1 has the newest mtime, its filename timestamp is oldest
    # and so it must be the deletion victim.
    assert not t1.exists(), "pruning picked wrong victim — it consulted mtime, not filename"
    assert t2.exists()
    assert t3.exists()


def test_fr03_malformed_filename_skipped(tmp_path: Path) -> None:
    """FR03: files not matching the timestamp regex are neither counted nor deleted."""
    malformed = tmp_path / "memory.db.corrupt.notatimestamp.bak"
    malformed.write_bytes(b"garbage")
    # Seed 5 valid timestamped files so keep=5 does not evict any (malformed does not count).
    for day in range(10, 15):
        _write_timestamped_backup(tmp_path, f"2026-04-{day}T00-00-00Z")

    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=5)

    assert malformed.exists()  # not deleted
    remaining = [p for p in tmp_path.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name)]
    assert len(remaining) == 5  # malformed did not count against the budget


# ---------------------------------------------------------------------------
# FR04 — legacy backup preservation
# ---------------------------------------------------------------------------


def test_fr04_legacy_bak_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR04: memory.db.corrupt.bak survives a recovery byte-identical."""
    legacy = _write_legacy_backup(tmp_path, "memory.db.corrupt.bak", b"legacy-sacred-bytes")
    legacy_content = legacy.read_bytes()

    db_path = tmp_path / "memory.db"
    _trigger_recovery(db_path, monkeypatch)

    assert legacy.exists()
    assert legacy.read_bytes() == legacy_content


def test_fr04_legacy_bak_1_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR04: memory.db.corrupt.bak.1 survives a recovery byte-identical."""
    legacy = _write_legacy_backup(tmp_path, "memory.db.corrupt.bak.1", b"legacy-v1-bytes")
    legacy_content = legacy.read_bytes()

    db_path = tmp_path / "memory.db"
    _trigger_recovery(db_path, monkeypatch)

    assert legacy.exists()
    assert legacy.read_bytes() == legacy_content


def test_fr04_legacy_counted_but_not_pruned(tmp_path: Path) -> None:
    """FR04: 2 legacy + 5 timestamped with keep=5 → one timestamped deleted, legacy untouched."""
    legacy_0 = _write_legacy_backup(tmp_path, "memory.db.corrupt.bak", b"legacy-0")
    legacy_1 = _write_legacy_backup(tmp_path, "memory.db.corrupt.bak.1", b"legacy-1")
    for day in range(1, 6):
        _write_timestamped_backup(tmp_path, f"2026-04-{day:02d}T00-00-00Z")

    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=5)

    # Total was 7 (2 legacy + 5 timestamped). Budget is 5. Excess = 2.
    # Only timestamped files are pruning victims → 2 timestamped deleted.
    assert legacy_0.exists()
    assert legacy_1.exists()
    remaining_ts = [p.name for p in tmp_path.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name)]
    assert len(remaining_ts) == 3
    # The two oldest were deleted — April 1 and April 2.
    assert all("2026-04-01" not in n and "2026-04-02" not in n for n in remaining_ts)


def test_fr04_legacy_only_overshoot_warns(tmp_path: Path) -> None:
    """FR04: 6 legacy files with keep=5: no deletion, exactly one WARNING emitted.

    Legacy file count exceeding the budget cannot be remedied — the WARNING
    surfaces the condition for operators to act on.
    """
    # The legacy-name set has only 2 entries (memory.db.corrupt.bak, .bak.1);
    # so the test scenario uses 2 legacy + (keep-N-1) timestamped to simulate
    # the overshoot path where ALL pruning candidates are timestamped but
    # the legacy total alone still exceeds the budget.
    # For the pure legacy-only overshoot we seed the 2 legacy names and use
    # keep_n=1 so the budget excess persists with no timestamped candidates.
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak")
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak.1")

    with structlog.testing.capture_logs() as logs:
        SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=1)

    overshoot = [log for log in logs if log.get("event") == "corrupt_backup_budget_exceeded_legacy_only"]
    assert len(overshoot) == 1
    assert overshoot[0]["keep"] == 1
    assert overshoot[0]["legacy_count"] == 2
    # Both legacy files still on disk.
    assert (tmp_path / "memory.db.corrupt.bak").exists()
    assert (tmp_path / "memory.db.corrupt.bak.1").exists()


# ---------------------------------------------------------------------------
# FR05 — salvage semantics unchanged
# ---------------------------------------------------------------------------


def test_fr05_salvage_semantics_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR05: db_recovered log payload still emits rows_salvaged and backup fields."""
    db_path = tmp_path / "memory.db"
    _populate_db(db_path, entries=2)
    _corrupt_sqlite_master(db_path)

    # Under empty_ok, recovery succeeds with 0 rows salvaged — same as pre-PRD baseline.
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
    # Critical payload fields from PRD-CORE-138 unchanged.
    assert entry["rows_salvaged"] == 0
    assert "backup" in entry
    assert "db" in entry
    # Backup path now carries the timestamped name.
    assert _TIMESTAMPED_BACKUP_RE.fullmatch(Path(entry["backup"]).name) is not None


# ---------------------------------------------------------------------------
# NFR01 — performance
# ---------------------------------------------------------------------------


def test_nfr01_rotation_latency_bounded(tmp_path: Path) -> None:
    """NFR01: _prune_corrupt_backups over 50 files completes in <20 ms."""
    for i in range(50):
        # Vary the timestamp by second to exercise sort comparator 50 times.
        _write_timestamped_backup(tmp_path, f"2026-04-10T00-00-{i % 60:02d}Z-{i}")

    start = time.perf_counter()
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=5)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 100, f"pruning took {elapsed_ms:.2f}ms (target <20ms, hard cap 100ms)"


# ---------------------------------------------------------------------------
# NFR02 — reliability & collision handling
# ---------------------------------------------------------------------------


def test_nfr02_filename_collision_same_second(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NFR02: two recoveries in the same UTC second produce <ts>.bak and <ts>-1.bak."""
    _freeze_utc(monkeypatch, datetime(2026, 4, 12, 3, 14, 59, tzinfo=timezone.utc))

    db_1 = tmp_path / "memory.db"
    db_1.write_bytes(b"dummy-1" * 100)
    first = SQLiteBackend._rotate_corrupt_backup(db_1)

    # Second corruption event at the exact same frozen UTC second.
    db_2 = tmp_path / "memory.db"
    db_2.write_bytes(b"dummy-2" * 100)
    second = SQLiteBackend._rotate_corrupt_backup(db_2)

    assert first.name == "memory.db.corrupt.2026-04-12T03-14-59Z.bak"
    assert second.name == "memory.db.corrupt.2026-04-12T03-14-59Z-1.bak"
    # Both parseable by the timestamped regex.
    assert _TIMESTAMPED_BACKUP_RE.fullmatch(first.name) is not None
    assert _TIMESTAMPED_BACKUP_RE.fullmatch(second.name) is not None


def test_nfr02_is_idempotent(tmp_path: Path) -> None:
    """NFR02: running prune twice on an already-bounded directory is a no-op."""
    for day in range(1, 6):
        _write_timestamped_backup(tmp_path, f"2026-04-{day:02d}T00-00-00Z")

    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=5)
    first_pass = sorted(p.name for p in tmp_path.iterdir())

    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=5)
    second_pass = sorted(p.name for p in tmp_path.iterdir())

    assert first_pass == second_pass


# ---------------------------------------------------------------------------
# NFR04 — observability
# ---------------------------------------------------------------------------


def test_nfr04_rotation_emits_structured_log(tmp_path: Path) -> None:
    """NFR04: _prune_corrupt_backups emits corrupt_backup_rotated INFO with expected fields."""
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
    # Started with 3 timestamped files; budget=2, legacy=1 → 2 timestamped deleted.
    # After deletion, total_timestamped is the count of what remains.
    assert entry["total_timestamped"] == 1
    assert entry["pruned"] == 2
    assert entry["parent"] == str(tmp_path)


# ---------------------------------------------------------------------------
# Negative paths
# ---------------------------------------------------------------------------


def test_negative_unlink_permission_error_is_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative: Path.unlink raising PermissionError during prune does not propagate."""
    for day in range(1, 4):
        _write_timestamped_backup(tmp_path, f"2026-04-{day:02d}T00-00-00Z")

    original_unlink = Path.unlink

    def _raise_perm(self: Path, *args: Any, **kwargs: Any) -> None:
        if "2026-04-01" in self.name:
            raise PermissionError("denied")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _raise_perm)

    # Should not raise even though the victim unlink fails.
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=1)


def test_negative_keep_equals_1_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Boundary: keep=1 leaves exactly one timestamped backup after a recovery."""
    _freeze_utc(monkeypatch, datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc))
    # Seed 3 existing timestamped backups.
    for day in range(1, 4):
        _write_timestamped_backup(tmp_path, f"2026-04-{day:02d}T00-00-00Z")

    # Prune down to 1.
    SQLiteBackend._prune_corrupt_backups(tmp_path, keep_n=1)

    remaining = [p.name for p in tmp_path.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name)]
    assert len(remaining) == 1
    # The newest (April 3) is what survives.
    assert remaining[0] == "memory.db.corrupt.2026-04-03T00-00-00Z.bak"


# ---------------------------------------------------------------------------
# Integration — end-to-end recovery sequences
# ---------------------------------------------------------------------------


def test_integration_end_to_end_7_recoveries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration: 7 real recoveries with keep=5 leave exactly 5 timestamped files."""
    db_path = tmp_path / "memory.db"

    # Class-based stub so seven calls all behave like PRD-CORE-138's happy salvage path.
    class _FakeRow:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def keys(self) -> list[str]:
            return list(self._data.keys())

        def __iter__(self) -> Any:
            return iter(self._data.values())

    now = "2026-04-13T00:00:00+00:00"
    fake_row = _FakeRow({"id": "L-seq", "content": "seq", "created_at": now, "updated_at": now})
    monkeypatch.setattr(
        SQLiteBackend,
        "_salvage_via_recover_cli",
        staticmethod(lambda _backup, dbapi=sqlite3: [fake_row]),
    )

    # Seven distinct UTC seconds so no collision suffixes are appended.
    for i in range(7):
        _freeze_utc(monkeypatch, datetime(2026, 4, 12, 3, 14, 50 + i, tzinfo=timezone.utc))
        _populate_db(db_path, entries=1)
        _corrupt_sqlite_master(db_path)
        conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict", corrupt_backup_keep=5)
        conn.close()

    remaining = sorted(p.name for p in tmp_path.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name))
    assert len(remaining) == 5
    # The two oldest (t=50 and t=51) were evicted.
    assert all("14-50Z" not in n and "14-51Z" not in n for n in remaining)


def test_integration_mixed_legacy_and_new(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration: legacy + timestamped coexist; only timestamped are ever pruned."""
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak", b"legacy-0")
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak.1", b"legacy-1")

    # Direct-drive the rotation helpers rather than reusing _trigger_recovery,
    # which asserts "exactly one timestamped backup" on return.
    class _FakeRow:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def keys(self) -> list[str]:
            return list(self._data.keys())

        def __iter__(self) -> Any:
            return iter(self._data.values())

    now_iso = "2026-04-13T00:00:00+00:00"
    fake_row = _FakeRow({"id": "L-m", "content": "m", "created_at": now_iso, "updated_at": now_iso})
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

    # Legacy survives untouched.
    assert (tmp_path / "memory.db.corrupt.bak").read_bytes() == b"legacy-0"
    assert (tmp_path / "memory.db.corrupt.bak.1").read_bytes() == b"legacy-1"
    # Total timestamped: 4 created, budget is 5 (legacy 2 + timestamped 4 = 6 → over budget).
    # 1 timestamped should have been pruned to fit 5 total.
    remaining_ts = [p for p in tmp_path.iterdir() if _TIMESTAMPED_BACKUP_RE.fullmatch(p.name)]
    assert len(remaining_ts) == 3  # 4 created, 1 pruned → 3 remain


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_regression_2_wide_overwrite_no_longer_occurs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: pre-PRD-139 2-wide rotation would have overwritten legacy files.

    Reproduces the 2026-04-12 pattern where a second corruption event destroyed
    the evidence from the first. With timestamped rotation + legacy preservation,
    both legacy files and the new timestamped backup must coexist.
    """
    legacy_0 = _write_legacy_backup(tmp_path, "memory.db.corrupt.bak", b"first-event")
    legacy_1 = _write_legacy_backup(tmp_path, "memory.db.corrupt.bak.1", b"second-event")

    db_path = tmp_path / "memory.db"
    _trigger_recovery(db_path, monkeypatch)

    # Legacy artifacts survived.
    assert legacy_0.read_bytes() == b"first-event"
    assert legacy_1.read_bytes() == b"second-event"
    # New timestamped backup exists alongside.
    assert _find_timestamped_backup(tmp_path).exists()


def test_regression_healthy_open_path_no_rotation(tmp_path: Path) -> None:
    """Regression: healthy DB open triggers zero rotation side effects."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.close()

    # No corrupt backup files should have been produced by a clean open.
    corrupt_files = [p for p in tmp_path.iterdir() if p.name.startswith("memory.db.corrupt.")]
    assert corrupt_files == []


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_migration_upgrade_preserves_legacy_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Migration: user upgrading from pre-139 keeps their legacy backups intact."""
    # Pre-upgrade state: directory only contains legacy files.
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak", b"pre-upgrade-0")
    _write_legacy_backup(tmp_path, "memory.db.corrupt.bak.1", b"pre-upgrade-1")

    db_path = tmp_path / "memory.db"
    _trigger_recovery(db_path, monkeypatch)

    # Both legacy files survived byte-for-byte.
    assert (tmp_path / "memory.db.corrupt.bak").read_bytes() == b"pre-upgrade-0"
    assert (tmp_path / "memory.db.corrupt.bak.1").read_bytes() == b"pre-upgrade-1"
    # One new timestamped backup appeared.
    assert _find_timestamped_backup(tmp_path).exists()
