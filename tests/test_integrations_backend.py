"""Tests for trw-memory backend integration helpers."""

from __future__ import annotations

from datetime import timezone
from typing import Any

from trw_memory.integrations._backend import create_backend, make_entry


class TestBackendHelper:
    """Tests for _backend.py helpers."""

    def test_make_entry_generates_id(self) -> None:
        entry = make_entry(content="test", namespace="ns")
        assert entry.id.startswith("M-")
        assert len(entry.id) == 18

    def test_make_entry_sets_timestamps(self) -> None:
        entry = make_entry(content="test")
        assert entry.created_at is not None
        assert entry.updated_at is not None
        assert entry.created_at.tzinfo == timezone.utc

    def test_make_entry_sets_tags(self) -> None:
        entry = make_entry(content="test", tags=["a", "b"])
        assert entry.tags == ["a", "b"]

    def test_create_backend_returns_storage_backend(self, tmp_path: Any) -> None:
        from trw_memory.storage.interface import StorageBackend

        backend = create_backend("test", storage_path=str(tmp_path))
        try:
            assert isinstance(backend, StorageBackend)
        finally:
            backend.close()
