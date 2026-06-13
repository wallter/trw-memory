"""Wave 15: coverage gap-fill for storage/_vector_ops.py.

Target lines: 63, 84, 96, 124, 128-136, 160, 205, 220, 232-255, 267, 293, 296-301.
"""
from __future__ import annotations

import sqlite3
import struct
import threading
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.storage._vector_ops import (
    delete_vector,
    delete_vector_internal,
    existing_vector_ids,
    get_stored_embeddings,
    search_vectors,
    upsert_vector,
    vector_exists,
)


def _mock_conn() -> MagicMock:
    conn = MagicMock(spec=sqlite3.Connection)
    conn.total_changes = 0
    return conn


def _lock() -> threading.Lock:
    return threading.Lock()


# ---------------------------------------------------------------------------
# delete_vector_internal
# ---------------------------------------------------------------------------

class TestDeleteVectorInternalRaise:
    def test_non_vec_sqlite_error_is_reraised(self) -> None:
        """Non-optional sqlite error in delete_vector_internal → re-raise (line 63)."""
        conn = _mock_conn()
        conn.execute.side_effect = sqlite3.OperationalError("disk I/O error")
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            delete_vector_internal(conn, "M-001")


# ---------------------------------------------------------------------------
# delete_vector with vec_available=False
# ---------------------------------------------------------------------------

class TestDeleteVectorUnavailable:
    def test_vec_unavailable_returns_false(self) -> None:
        """delete_vector with vec_available=False → return False immediately (line 84)."""
        conn = _mock_conn()
        result = delete_vector(conn, _lock(), vec_available=False, entry_id="M-001")
        assert result is False
        conn.execute.assert_not_called()


# ---------------------------------------------------------------------------
# vector_exists with vec_available=False
# ---------------------------------------------------------------------------

class TestVectorExistsUnavailable:
    def test_vec_unavailable_returns_false(self) -> None:
        """vector_exists with vec_available=False → return False (line 96)."""
        conn = _mock_conn()
        result = vector_exists(conn, vec_available=False, entry_id="M-001")
        assert result is False
        conn.execute.assert_not_called()


# ---------------------------------------------------------------------------
# existing_vector_ids
# ---------------------------------------------------------------------------

class TestExistingVectorIdsUnavailable:
    def test_vec_unavailable_returns_empty_set(self) -> None:
        """existing_vector_ids with vec_available=False → return set() (line 124)."""
        conn = _mock_conn()
        result = existing_vector_ids(conn, _lock(), vec_available=False)
        assert result == set()

    def test_optional_vec_sqlite_error_returns_empty_set(self) -> None:
        """sqlite.Error (vec0 absent) during existing_vector_ids → debug + return set() (lines 128-136)."""
        conn = _mock_conn()
        conn.execute.side_effect = sqlite3.OperationalError("no such module: vec0")
        result = existing_vector_ids(conn, _lock(), vec_available=True)
        assert result == set()

    def test_non_vec_sqlite_error_returns_empty_set_with_warning(self) -> None:
        """Non-optional sqlite.Error during existing_vector_ids → warning + return set() (line 135)."""
        conn = _mock_conn()
        conn.execute.side_effect = sqlite3.OperationalError("disk I/O error")
        result = existing_vector_ids(conn, _lock(), vec_available=True)
        assert result == set()


# ---------------------------------------------------------------------------
# upsert_vector
# ---------------------------------------------------------------------------

class TestUpsertVectorUnavailable:
    def test_vec_unavailable_returns_early(self) -> None:
        """upsert_vector with vec_available=False → return without touching conn (line 160)."""
        conn = _mock_conn()
        upsert_vector(conn, _lock(), vec_available=False, dim=3, entry_id="M-001", embedding=[0.1, 0.2, 0.3])
        conn.execute.assert_not_called()

    def test_non_vec_sqlite_error_is_reraised(self) -> None:
        """Non-optional sqlite error in upsert_vector → re-raise (line 205)."""
        conn = _mock_conn()
        conn.execute.side_effect = sqlite3.OperationalError("disk I/O error")
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            upsert_vector(conn, _lock(), vec_available=True, dim=3, entry_id="M-001", embedding=[0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# search_vectors
# ---------------------------------------------------------------------------

class TestSearchVectorsUnavailable:
    def test_vec_unavailable_returns_empty_list(self) -> None:
        """search_vectors with vec_available=False → return [] (line 220)."""
        conn = _mock_conn()
        result = search_vectors(conn, _lock(), vec_available=False, dim=3, query_embedding=[0.1, 0.2, 0.3])
        assert result == []
        conn.execute.assert_not_called()

    def test_dimension_mismatch_returns_empty_list(self) -> None:
        """query_embedding length != dim → return [] (lines 226-231)."""
        conn = _mock_conn()
        result = search_vectors(conn, _lock(), vec_available=True, dim=3, query_embedding=[0.1, 0.2])
        assert result == []
        conn.execute.assert_not_called()

    def test_successful_knn_search_returns_results(self) -> None:
        """search_vectors executes KNN SQL and returns (entry_id, distance) pairs (lines 232-245)."""
        conn = _mock_conn()
        conn.execute.return_value.fetchall.return_value = [("M-001", 0.12), ("M-002", 0.45)]
        result = search_vectors(conn, _lock(), vec_available=True, dim=2, query_embedding=[0.1, 0.2])
        assert result == [("M-001", 0.12), ("M-002", 0.45)]

    def test_optional_vec_sqlite_error_returns_empty_list(self) -> None:
        """sqlite.Error (vec0 absent) during search_vectors → debug + return [] (lines 251-252)."""
        conn = _mock_conn()
        conn.execute.side_effect = sqlite3.OperationalError("no such module: vec0")
        result = search_vectors(conn, _lock(), vec_available=True, dim=2, query_embedding=[0.1, 0.2])
        assert result == []

    def test_non_vec_sqlite_error_returns_empty_list_with_warning(self) -> None:
        """Non-optional sqlite.Error during search_vectors → warning + return [] (lines 253-255)."""
        conn = _mock_conn()
        conn.execute.side_effect = sqlite3.OperationalError("disk I/O error")
        result = search_vectors(conn, _lock(), vec_available=True, dim=2, query_embedding=[0.1, 0.2])
        assert result == []


# ---------------------------------------------------------------------------
# get_stored_embeddings
# ---------------------------------------------------------------------------

class TestGetStoredEmbeddingsUnavailable:
    def test_vec_unavailable_returns_empty_dict(self) -> None:
        """get_stored_embeddings with vec_available=False → return {} (line 267)."""
        conn = _mock_conn()
        result = get_stored_embeddings(conn, _lock(), vec_available=False, entry_ids=["M-001"])
        assert result == {}
        conn.execute.assert_not_called()

    def test_empty_entry_ids_returns_empty_dict(self) -> None:
        """get_stored_embeddings with empty entry_ids → return {} (line 267)."""
        conn = _mock_conn()
        result = get_stored_embeddings(conn, _lock(), vec_available=True, entry_ids=[])
        assert result == {}

    def test_none_embedding_blob_is_skipped(self) -> None:
        """Row with None blob → continue (line 293)."""
        conn = _mock_conn()
        conn.execute.return_value.fetchall.return_value = [("M-001", None)]
        result = get_stored_embeddings(conn, _lock(), vec_available=True, entry_ids=["M-001"])
        assert result == {}

    def test_invalid_blob_length_is_skipped(self) -> None:
        """Blob with len % 4 != 0 → debug logged + continue (lines 296-301)."""
        conn = _mock_conn()
        bad_blob = b"\x01\x02\x03"  # 3 bytes, not divisible by 4
        conn.execute.return_value.fetchall.return_value = [("M-001", bad_blob)]
        result = get_stored_embeddings(conn, _lock(), vec_available=True, entry_ids=["M-001"])
        assert result == {}

    def test_valid_blob_is_decoded(self) -> None:
        """Valid embedding blob → decoded to float list (lines 302-303)."""
        conn = _mock_conn()
        embedding = [0.1, 0.2, 0.3]
        blob = struct.pack("3f", *embedding)
        conn.execute.return_value.fetchall.return_value = [("M-001", blob)]
        result = get_stored_embeddings(conn, _lock(), vec_available=True, entry_ids=["M-001"])
        assert "M-001" in result
        assert len(result["M-001"]) == 3
