"""Wave 15: coverage gap-fill for storage/_stale_handle.py (lines 42, 57, 99-100)."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import StaleConnectionError
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
        backend._skip_commit_depth = 0
        backend._conn.in_transaction = False
        backend._dim = 384

        with (
            patch("trw_memory.storage._stale_handle.ensure_schema"),
            patch("trw_memory.storage._stale_handle.load_vec_extension", return_value=True),
            patch("trw_memory.storage._stale_handle.ensure_fts_table", return_value=True),
        ):
            reconnect(backend)

        backend._open_and_configure.assert_called_once_with(
            backend._db_path,
            dbapi=backend._dbapi,
            sqlcipher_key_hex="deadbeef",
        )
        assert backend.reconnect_count == 1

    @pytest.mark.parametrize(("depth", "in_transaction"), [(1, False), (0, True)])
    def test_active_transaction_blocks_reconnect(self, depth: int, in_transaction: bool) -> None:
        backend = MagicMock()
        backend._skip_commit_depth = depth
        backend._conn.in_transaction = in_transaction

        with pytest.raises(StaleConnectionError, match="transaction is active"):
            reconnect(backend)
        backend._open_and_configure.assert_not_called()

    def test_candidate_schema_failure_preserves_old_connection(self) -> None:
        backend = MagicMock()
        old_conn = backend._conn
        old_conn.in_transaction = False
        backend._skip_commit_depth = 0
        backend._sqlcipher_key_hex = None
        backend._db_path = "/tmp/test.db"
        backend.reconnect_count = 4
        candidate = backend._open_and_configure.return_value

        with (
            patch("trw_memory.storage._stale_handle.prepare_db_file_mode"),
            patch("trw_memory.storage._stale_handle.ensure_schema", side_effect=sqlite3.DatabaseError("bad schema")),
            pytest.raises(StaleConnectionError, match="bad schema"),
        ):
            reconnect(backend)

        assert backend._conn is old_conn
        assert backend.reconnect_count == 4
        candidate.close.assert_called_once()
        old_conn.close.assert_not_called()
        backend._stale_detector.reset.assert_not_called()


class TestRunIntegrityCheck:
    def test_database_error_returns_false(self) -> None:
        """sqlite3.DatabaseError in execute → return False (lines 99-100)."""
        backend = MagicMock()
        backend._conn.execute.side_effect = sqlite3.DatabaseError("corrupt")
        result = run_integrity_check(backend)
        assert result is False
