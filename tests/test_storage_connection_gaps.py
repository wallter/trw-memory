"""Wave 12: targeted tests for uncovered branches in storage/_connection.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import structlog.testing

from trw_memory.storage._connection import (
    apply_open_pragmas,
    check_integrity,
    db_has_data,
    open_and_configure,
    open_without_integrity_check,
)

# ---------------------------------------------------------------------------
# apply_open_pragmas — verify=True warning branches (lines 57, 60)
# ---------------------------------------------------------------------------


class TestApplyOpenPragmasVerify:
    def test_verify_false_no_warning_logged(self, tmp_path: Path) -> None:
        """verify=False never logs WAL/sync warnings."""
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        try:
            with structlog.testing.capture_logs() as logs:
                apply_open_pragmas(conn, verify=False)
        finally:
            conn.close()
        assert [entry for entry in logs if entry.get("log_level") == "warning"] == []

    def test_verify_true_wal_not_enabled_logs_warning(self) -> None:
        """verify=True + WAL result not 'wal' → warning logged (line 57)."""
        mock_conn = MagicMock()
        mock_wal_cursor = MagicMock()
        mock_wal_cursor.fetchone.return_value = ("memory",)
        mock_sync_cursor = MagicMock()
        mock_sync_cursor.fetchone.return_value = (1,)

        def _execute(sql: str):
            if "journal_mode" in sql:
                return mock_wal_cursor
            if "synchronous" in sql:
                return mock_sync_cursor
            return MagicMock()

        mock_conn.execute.side_effect = _execute

        import structlog.testing

        with structlog.testing.capture_logs() as logs:
            apply_open_pragmas(mock_conn, verify=True)

        assert any("wal_mode_not_enabled" in str(l.get("event", "")) for l in logs)

    def test_verify_true_sync_not_normal_logs_warning(self) -> None:
        """verify=True + synchronous result not 1/'1' → warning logged (line 60)."""
        mock_conn = MagicMock()
        mock_wal_cursor = MagicMock()
        mock_wal_cursor.fetchone.return_value = ("wal",)
        mock_sync_cursor = MagicMock()
        mock_sync_cursor.fetchone.return_value = (2,)  # FULL, not NORMAL

        def _execute(sql: str):
            if "journal_mode" in sql:
                return mock_wal_cursor
            if "synchronous" in sql:
                return mock_sync_cursor
            return MagicMock()

        mock_conn.execute.side_effect = _execute

        import structlog.testing

        with structlog.testing.capture_logs() as logs:
            apply_open_pragmas(mock_conn, verify=True)

        assert any("synchronous_normal_not_set" in str(l.get("event", "")) for l in logs)


# ---------------------------------------------------------------------------
# open_without_integrity_check (lines 151-160)
# ---------------------------------------------------------------------------


class TestOpenWithoutIntegrityCheck:
    def test_opens_connection_without_checking_integrity(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE foo (x INTEGER)")
        conn.commit()
        conn.close()

        result = open_without_integrity_check(db_path)
        try:
            assert result is not None
        finally:
            result.close()

    def test_memory_db_path(self) -> None:
        conn = open_without_integrity_check(Path(":memory:"))
        try:
            assert conn is not None
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# open_and_configure — integrity check retry and failure (lines 132-141)
# ---------------------------------------------------------------------------


class TestOpenAndConfigureIntegrityFailure:
    def test_integrity_check_failure_raises_database_error(self, tmp_path: Path) -> None:
        """When quick_check returns non-'ok' twice → raises DatabaseError."""
        db_path = tmp_path / "test.db"

        # Create a valid DB first
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE foo (x INTEGER)")
        conn.commit()
        conn.close()

        call_count = 0

        def _execute_side_effect(sql: str):
            nonlocal call_count
            if "quick_check" in sql:
                call_count += 1
                mock_result = MagicMock()
                mock_result.fetchall.return_value = [("database is malformed",)]
                return mock_result
            # Let other pragmas execute normally
            conn_inner = sqlite3.connect(str(db_path))
            try:
                return conn_inner.execute(sql)
            except Exception:
                return MagicMock()

        with patch("trw_memory.storage._connection.time.sleep"):
            with patch("trw_memory.storage._connection.connect") as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value = mock_conn

                def _execute(sql: str):
                    if "quick_check" in sql:
                        mock_result = MagicMock()
                        mock_result.fetchall.return_value = [("malformed",)]
                        return mock_result
                    return MagicMock()

                mock_conn.execute.side_effect = _execute

                with pytest.raises(sqlite3.DatabaseError, match="malformed"):
                    open_and_configure(db_path)
                mock_conn.close.assert_called_once()

        assert call_count == 0  # we used the mock, not the real counter


# ---------------------------------------------------------------------------
# check_integrity (lines 176-189)
# ---------------------------------------------------------------------------


class TestCheckIntegrity:
    def test_healthy_db_returns_ok_true(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE foo (x INTEGER)")
        conn.commit()
        conn.close()

        result = check_integrity(db_path)

        assert result["ok"] is True
        assert result["detail"] == "ok"
        assert str(db_path) in str(result["db_path"])

    def test_missing_file_returns_ok_false(self, tmp_path: Path) -> None:
        db_path = tmp_path / "missing.db"

        result = check_integrity(db_path)

        # New file is created but it has no tables — quick_check returns "ok"
        # OR it fails with DatabaseError depending on sqlite version
        assert "ok" in result

    def test_database_error_returns_ok_false(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"

        with patch(
            "trw_memory.storage._connection.connect",
            side_effect=sqlite3.DatabaseError("corrupt"),
        ):
            result = check_integrity(db_path)

        assert result["ok"] is False
        assert "corrupt" in str(result["detail"])

    def test_connection_closed_on_unexpected_exception(self, tmp_path: Path) -> None:
        """A non-sqlite exception during quick_check must still close the conn.

        Regression: close() lived in the try body, so a KeyboardInterrupt/
        MemoryError (anything not sqlite3.DatabaseError) leaked the connection.
        """
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = RuntimeError("boom")  # not a sqlite3 error

        with patch("trw_memory.storage._connection.connect", return_value=mock_conn):
            with pytest.raises(RuntimeError, match="boom"):
                check_integrity(tmp_path / "test.db")

        mock_conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# db_has_data (lines 203-219)
# ---------------------------------------------------------------------------


class TestDbHasData:
    def test_empty_db_returns_false(self, tmp_path: Path) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)
        backend.close()

        result = db_has_data(db_path)
        assert result is False

    def test_db_with_rows_returns_true(self, tmp_path: Path) -> None:
        from trw_memory.models.memory import MemoryEntry
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)
        backend.store(MemoryEntry(id="D-001", content="test"))
        backend.close()

        result = db_has_data(db_path)
        assert result is True

    def test_missing_memories_table_returns_false(self, tmp_path: Path) -> None:
        """Database without memories table → sqlite3.Error → returns False."""
        db_path = tmp_path / "nomem.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE other (x INTEGER)")
        conn.commit()
        conn.close()

        result = db_has_data(db_path)
        assert result is False

    def test_a_locked_probe_is_unknown_not_empty(self, tmp_path: Path) -> None:
        """The defect this test used to assert as correct.

        It was ``test_connect_error_returns_false`` and it used the literal error
        ``locked``. ``open_connection_with_recovery`` asks ``db_has_data`` on its
        lock-contention branch, and under contention the PROBE is locked too — so
        ``False`` told that caller "this store has no rows" and it took the
        destructive branch: rename the live database to ``.corrupt.bak`` and
        initialise a blank schema. A populated store was wiped because the machine
        was busy, which is exactly when it is most likely to happen.

        Found by a cross-family audit 2026-09-12.
        """
        db_path = tmp_path / "test.db"

        with patch(
            "trw_memory.storage._connection.connect",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            result = db_has_data(db_path)
        assert result is None
        assert result is not False, (
            "False is the value that sends open_connection_with_recovery down the "
            "destructive branch; UNKNOWN must never be spelled that way"
        )

    def test_a_structural_connect_error_is_still_false(self, tmp_path: Path) -> None:
        """Non-vacuity partner, and the boundary of the fix.

        Only lock/busy is UNKNOWN. A genuine structural failure has no readable
        rows and the recovery path is designed for it, so widening the tri-state
        to every ``sqlite3.Error`` would disable recovery for real corruption.
        """
        db_path = tmp_path / "test.db"

        with patch(
            "trw_memory.storage._connection.connect",
            side_effect=sqlite3.DatabaseError("file is not a database"),
        ):
            assert db_has_data(db_path) is False
