"""SQLiteBackend init-time recovery routing tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
import structlog

from trw_memory.exceptions import CorruptDatabaseUnsalvageableError
from trw_memory.storage._init_helpers import open_connection_with_recovery
from trw_memory.storage._recovery import classify_recovery_preflight, recovery_state_path, write_recovery_state


class _FakeBackend:
    def __init__(self, exc: sqlite3.DatabaseError) -> None:
        self.exc = exc
        self.recover_called = False
        self.open_without_called = False

    def _open_and_configure(self, _db_path: Path) -> Any:
        raise self.exc

    def _db_has_data(self, _db_path: Path, *, dbapi: Any, sqlcipher_key_hex: str | None) -> bool:
        return True

    def _open_without_integrity_check(self, _db_path: Path, *, dbapi: Any, sqlcipher_key_hex: str | None) -> Any:
        self.open_without_called = True
        return sqlite3.connect(":memory:")

    def recover_db(
        self,
        _db_path: Path,
        *,
        dbapi: Any,
        sqlcipher_key_hex: str | None,
        recovery_policy: str,
        corrupt_backup_keep: int,
        rebuild_from_cold: bool,
    ) -> Any:
        self.recover_called = True
        return sqlite3.connect(":memory:")


def test_quick_check_failure_with_rows_recovers_instead_of_opening_corrupt_db(tmp_path: Path) -> None:
    """A row-count probe is not a health check; failed quick_check must recover."""
    backend = _FakeBackend(sqlite3.DatabaseError("database disk image is malformed (quick_check failed twice)"))

    conn, integrity_warning, recovered = open_connection_with_recovery(
        backend,  # type: ignore[arg-type]
        tmp_path / "memory.db",
        dbapi=sqlite3,
        sqlcipher_key_hex=None,
        recovery_policy="strict",
        corrupt_backup_keep=5,
        rebuild_from_cold=True,
    )

    conn.close()
    assert backend.recover_called is True
    assert backend.open_without_called is False
    assert integrity_warning is False
    assert recovered is True


def test_lock_contention_with_rows_keeps_non_destructive_open_without_probe(tmp_path: Path) -> None:
    """Explicit SQLite lock/busy errors remain the non-destructive fallback case."""
    backend = _FakeBackend(sqlite3.DatabaseError("database is locked"))

    conn, integrity_warning, recovered = open_connection_with_recovery(
        backend,  # type: ignore[arg-type]
        tmp_path / "memory.db",
        dbapi=sqlite3,
        sqlcipher_key_hex=None,
        recovery_policy="strict",
        corrupt_backup_keep=5,
        rebuild_from_cold=True,
    )

    conn.close()
    assert backend.recover_called is False
    assert backend.open_without_called is True
    assert integrity_warning is True
    assert recovered is False
    assert recovery_state_path(tmp_path / "memory.db").exists()


def test_preflight_classifies_large_db_as_degraded_open(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")

    preflight = classify_recovery_preflight(db_path, inline_max_bytes=4)

    assert preflight.classification == "degraded_open_with_background_recovery"
    assert preflight.reason == "db_exceeds_inline_recovery_budget"
    assert preflight.db_size_bytes == 10


def test_degraded_preflight_blocks_inline_recovery_and_persists_state(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    backend = _FakeBackend(sqlite3.DatabaseError("database disk image is malformed"))

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        open_connection_with_recovery(
            backend,  # type: ignore[arg-type]
            db_path,
            dbapi=sqlite3,
            sqlcipher_key_hex=None,
            recovery_policy="strict",
            corrupt_backup_keep=5,
            rebuild_from_cold=True,
            recovery_inline_max_bytes=4,
        )

    assert backend.recover_called is False
    assert "degraded_open_with_background_recovery" in recovery_state_path(db_path).read_text(encoding="utf-8")


def test_recovery_state_write_is_valid_json_and_classifies_hard_fail(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")

    write_recovery_state(db_path, status="hard_fail", reason="inline_recovery_failed", db_size_bytes=10)

    state = recovery_state_path(db_path).read_text(encoding="utf-8")
    assert '"status": "hard_fail"' in state
    assert classify_recovery_preflight(db_path, inline_max_bytes=1024).classification == "hard_fail"


@pytest.mark.parametrize("status", ["pending", "running", "degraded_open_with_background_recovery"])
def test_valid_pending_status_yields_degraded_open(tmp_path: Path, status: str) -> None:
    """Valid in-flight statuses keep the degraded-open-with-background-recovery path."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    write_recovery_state(db_path, status=status, reason="t", db_size_bytes=10)

    # Inline budget large enough that size alone would classify fast_open;
    # the persisted in-flight status is what must force the degraded path.
    preflight = classify_recovery_preflight(db_path, inline_max_bytes=1024)

    assert preflight.classification == "degraded_open_with_background_recovery"
    assert preflight.reason == "recovery_already_pending"
    assert preflight.persisted_status == status


def test_malformed_json_state_does_not_raise_and_falls_through_to_size(tmp_path: Path) -> None:
    """A truncated/garbage sidecar must not crash; classification falls back to size."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    recovery_state_path(db_path).write_text("{not valid json", encoding="utf-8")

    # Oversized DB still degrades on the size budget; no hard_fail, no raise.
    over_budget = classify_recovery_preflight(db_path, inline_max_bytes=4)
    assert over_budget.classification == "degraded_open_with_background_recovery"
    assert over_budget.reason == "db_exceeds_inline_recovery_budget"
    assert over_budget.persisted_status == ""

    # Within budget falls through to fast_open — malformed state is ignored, not hard_fail.
    within_budget = classify_recovery_preflight(db_path, inline_max_bytes=1024)
    assert within_budget.classification == "fast_open"


def test_non_utf8_state_does_not_raise_and_falls_through_to_size(tmp_path: Path) -> None:
    """Non-UTF-8 bytes raise UnicodeDecodeError on read_text — the seam must absorb it."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    # 0xFF is invalid UTF-8; the prior read_text(encoding="utf-8") would crash here.
    recovery_state_path(db_path).write_bytes(b"\xff\xfe\x00\x80garbage")

    within_budget = classify_recovery_preflight(db_path, inline_max_bytes=1024)
    assert within_budget.classification == "fast_open"
    assert within_budget.persisted_status == ""

    over_budget = classify_recovery_preflight(db_path, inline_max_bytes=4)
    assert over_budget.classification == "degraded_open_with_background_recovery"


def test_non_object_json_state_does_not_hard_fail(tmp_path: Path) -> None:
    """A JSON array/scalar is not an object — status is absent, so no hard_fail."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    recovery_state_path(db_path).write_text('["hard_fail"]', encoding="utf-8")

    preflight = classify_recovery_preflight(db_path, inline_max_bytes=1024)

    assert preflight.classification == "fast_open"
    assert preflight.persisted_status == ""


def test_non_string_status_does_not_hard_fail(tmp_path: Path) -> None:
    """A non-string status (e.g. a nested object) must not be coerced into a hard_fail trigger."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    recovery_state_path(db_path).write_text('{"status": {"nested": "hard_fail"}}', encoding="utf-8")

    preflight = classify_recovery_preflight(db_path, inline_max_bytes=1024)

    assert preflight.classification == "fast_open"
    assert preflight.persisted_status == ""


def test_corrupt_state_diagnostics_are_content_free(tmp_path: Path) -> None:
    """Any logs emitted for a corrupt sidecar must leak neither the path nor the raw payload."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    state_path = recovery_state_path(db_path)
    secret_marker = "SECRET-sk-live-DEADBEEF"
    # Malformed JSON carrying a secret marker; the seam must not echo it or the path.
    state_path.write_text(f"{{not json {secret_marker}", encoding="utf-8")

    with structlog.testing.capture_logs() as logs:
        classify_recovery_preflight(db_path, inline_max_bytes=1024)

    for event in logs:
        rendered = repr(event)
        assert secret_marker not in rendered
        assert str(state_path) not in rendered
        assert state_path.name not in rendered
