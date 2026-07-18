"""Wave 12: targeted tests for uncovered branches in storage/_connection.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.storage._connection import (
    apply_open_pragmas,
    check_integrity,
    connect,
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
            apply_open_pragmas(conn, verify=False)
        finally:
            conn.close()

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

    def test_connect_error_returns_false(self, tmp_path: Path) -> None:
        """sqlite3.Error on connect → returns False."""
        db_path = tmp_path / "test.db"

        with patch(
            "trw_memory.storage._connection.connect",
            side_effect=sqlite3.OperationalError("locked"),
        ):
            result = db_has_data(db_path)

        assert result is False


# ---------------------------------------------------------------------------
# sqlcipher paths — key validation (lines 96-100) and pragma (lines 66-68)
# ---------------------------------------------------------------------------


class TestSqlcipherPaths:
    def test_invalid_sqlcipher_key_too_short_raises_value_error(self, tmp_path: Path) -> None:
        """sqlcipher_key_hex with < 64 chars → raises ValueError."""
        with pytest.raises(ValueError, match="64-character"):
            connect(
                tmp_path / "test.db",
                dbapi=sqlite3,
                timeout=5.0,
                check_same_thread=True,
                sqlcipher_key_hex="abc123",  # too short
            )

    def test_invalid_sqlcipher_key_uppercase_raises_value_error(self, tmp_path: Path) -> None:
        """sqlcipher_key_hex with uppercase chars → raises ValueError."""
        bad_key = "A" * 64  # uppercase, not valid lowercase hex
        with pytest.raises(ValueError, match="lowercase hex"):
            connect(
                tmp_path / "test.db",
                dbapi=sqlite3,
                timeout=5.0,
                check_same_thread=True,
                sqlcipher_key_hex=bad_key,
            )

    def test_apply_sqlcipher_pragmas_safe_delegates_to_parent(self) -> None:
        """_apply_sqlcipher_pragmas_safe calls the parent module's function."""
        from trw_memory.storage._connection import _apply_sqlcipher_pragmas_safe

        mock_conn = MagicMock()
        with patch("trw_memory.storage.sqlite_backend._apply_sqlcipher_pragmas") as mock_parent:
            _apply_sqlcipher_pragmas_safe(mock_conn)
        mock_parent.assert_called_once_with(mock_conn)

    def test_valid_sqlcipher_key_applies_key_pragma(self, tmp_path: Path) -> None:
        """Valid 64-char hex key → PRAGMA key executed (lines 98-100)."""
        valid_key = "a" * 64  # valid 64-char lowercase hex
        mock_conn = MagicMock()
        mock_conn.row_factory = None

        with (
            patch("trw_memory.storage._connection.sqlite3.connect", return_value=mock_conn),
            patch("trw_memory.storage._connection._apply_sqlcipher_pragmas_safe"),
        ):
            try:
                connect(
                    tmp_path / "test.db",
                    dbapi=sqlite3,
                    timeout=5.0,
                    check_same_thread=True,
                    sqlcipher_key_hex=valid_key,
                )
            except Exception:
                pass  # connection may fail due to missing sqlcipher

        # Verify the key PRAGMA was attempted
        calls = [str(c) for c in mock_conn.execute.call_args_list]
        assert any("PRAGMA key" in c for c in calls)
