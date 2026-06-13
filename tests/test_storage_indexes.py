"""Tests that enterprise-scale composite indexes exist in the schema."""
import sqlite3
from pathlib import Path
import pytest
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _get_index_names(backend: SQLiteBackend) -> set[str]:
    rows = backend._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    return {row[0] for row in rows}


class TestCompositeIndexes:
    def test_ns_status_index_exists(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "idx.db")
        assert "idx_memories_ns_status" in _get_index_names(backend)

    def test_ns_status_importance_index_exists(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "idx.db")
        assert "idx_memories_ns_status_imp" in _get_index_names(backend)

    def test_status_updated_index_exists(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "idx.db")
        assert "idx_memories_status_updated" in _get_index_names(backend)
