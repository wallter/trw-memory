"""SQLiteBackend init-time recovery routing tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from trw_memory.storage._init_helpers import open_connection_with_recovery


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
