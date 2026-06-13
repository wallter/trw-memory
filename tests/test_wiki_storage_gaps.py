"""Wave 13: coverage gap-fill for wiki/storage.py (lines 78-79, 94-95, 132-133, 143-144)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry
from trw_memory.wiki.storage import (
    purge_wiki_refs_for_entry,
    query_wiki_inbound_refs,
    query_wiki_outbound_refs,
    replace_wiki_refs_for_entry,
)


def _wiki_entry(entry_id: str = "M-001") -> MemoryEntry:
    """Build a MemoryEntry with wiki.page metadata containing an outbound ref."""
    payload = json.dumps(
        {
            "kind": "topic",
            "slug": "topic/foo",
            "title": "Foo",
            "provenance": [],
            "confidence": "unverified",
            "evidence": [],
            "outbound_refs": [
                {"target_slug": "topic/bar", "ref_type": "related", "label": "", "bidirectional": True}
            ],
            "path": "",
        }
    )
    return MemoryEntry(id=entry_id, content="test", metadata={"wiki.page": payload})


def _failing_backend(tmp_path: Path):
    """Return a SQLiteBackend with _conn replaced by a MagicMock that raises sqlite3.Error."""
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    backend = SQLiteBackend(tmp_path / "test.db")
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.Error("disk error")
    backend._conn = mock_conn  # type: ignore[assignment]
    return backend


class TestReplaceWikiRefsError:
    def test_sqlite_error_raises_storage_error(self, tmp_path: Path) -> None:
        """sqlite3.Error in replace_wiki_refs_for_entry → StorageError (lines 78-79)."""
        backend = _failing_backend(tmp_path)
        entry = _wiki_entry()

        try:
            with pytest.raises(StorageError, match="Failed to replace"):
                replace_wiki_refs_for_entry(backend, entry)
        finally:
            backend.close()


class TestPurgeWikiRefsError:
    def test_sqlite_error_raises_storage_error(self, tmp_path: Path) -> None:
        """sqlite3.Error in purge_wiki_refs_for_entry → StorageError (lines 94-95)."""
        backend = _failing_backend(tmp_path)

        try:
            with pytest.raises(StorageError, match="Failed to purge"):
                purge_wiki_refs_for_entry(backend, "M-001")
        finally:
            backend.close()


class TestQueryWikiRefsNamespaceFilter:
    def test_query_outbound_with_namespace_filter_applies_clause(self, tmp_path: Path) -> None:
        """Passing namespace to query_wiki_outbound_refs hits namespace clause (lines 132-133)."""
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        backend = SQLiteBackend(tmp_path / "test.db")
        try:
            result = query_wiki_outbound_refs(backend, "topic/foo", namespace="project:default")
            assert isinstance(result, list)
        finally:
            backend.close()

    def test_query_inbound_with_namespace_filter_applies_clause(self, tmp_path: Path) -> None:
        """Passing namespace to query_wiki_inbound_refs hits namespace clause (lines 132-133)."""
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        backend = SQLiteBackend(tmp_path / "test.db")
        try:
            result = query_wiki_inbound_refs(backend, "topic/bar", namespace="project:default")
            assert isinstance(result, list)
        finally:
            backend.close()


class TestQueryWikiRefsError:
    def test_sqlite_error_raises_storage_error(self, tmp_path: Path) -> None:
        """sqlite3.Error in _query_refs → StorageError (lines 143-144)."""
        backend = _failing_backend(tmp_path)

        try:
            with pytest.raises(StorageError, match="Failed to query"):
                query_wiki_outbound_refs(backend, "topic/foo")
        finally:
            backend.close()
