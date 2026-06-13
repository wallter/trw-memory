"""Wave 15: coverage gap-fill for storage/_stale_handle.py (lines 42, 57, 99-100)."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

from trw_memory.storage._stale_handle import (
    handle_integrity_regression,
    reconnect,
    run_integrity_check,
)


class TestHandleIntegrityRegression:
    def test_sets_integrity_warning_flag(self) -> None:
        """handle_integrity_regression sets backend.integrity_warning = True (line 42)."""
        backend = MagicMock()
        handle_integrity_regression(backend)
        assert backend.integrity_warning is True


class TestReconnectSqlCipherPath:
    def test_sqlcipher_path_calls_open_and_configure_with_key(self) -> None:
        """_sqlcipher_key_hex is not None → open_and_configure called with key (line 57)."""
        backend = MagicMock()
        backend._sqlcipher_key_hex = "deadbeef"
        backend._db_path = "/tmp/test.db"
        backend._dbapi = MagicMock()
        backend.reconnect_count = 0

        with patch("trw_memory.storage._stale_handle.ensure_schema"):
            reconnect(backend)

        backend._open_and_configure.assert_called_once_with(
            backend._db_path,
            dbapi=backend._dbapi,
            sqlcipher_key_hex="deadbeef",
        )
        assert backend.reconnect_count == 1


class TestRunIntegrityCheck:
    def test_database_error_returns_false(self) -> None:
        """sqlite3.DatabaseError in execute → return False (lines 99-100)."""
        backend = MagicMock()
        backend._conn.execute.side_effect = sqlite3.DatabaseError("corrupt")
        result = run_integrity_check(backend)
        assert result is False
