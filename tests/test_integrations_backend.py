"""Tests for trw-memory backend integration helpers."""

from __future__ import annotations

from datetime import timezone
from typing import Any
from unittest.mock import MagicMock

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

    def test_resolve_backend_with_provided_backend(self) -> None:
        """resolve_backend returns provided backend without ownership."""
        from trw_memory.integrations._backend import resolve_backend

        mock_backend = MagicMock()
        backend, owns = resolve_backend("ns", None, mock_backend)
        assert backend is mock_backend
        assert not owns

    def test_resolve_backend_creates_new_backend(self, tmp_path: Any) -> None:
        """resolve_backend creates and owns backend when none provided."""
        from trw_memory.integrations._backend import resolve_backend
        from trw_memory.storage.interface import StorageBackend

        backend, owns = resolve_backend("test", str(tmp_path), None)
        try:
            assert isinstance(backend, StorageBackend)
            assert owns
        finally:
            backend.close()
