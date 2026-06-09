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


class TestReadNamespaceMetadata:
    """``_read_namespace_metadata`` must fail open on a corrupt sidecar.

    ``discover_namespace_backends`` iterates every on-disk namespace dir and,
    under encryption, reads ``namespace.txt`` per candidate. A single
    unreadable or non-UTF-8 sidecar must isolate to that one namespace
    (the caller skips it) rather than aborting cross-namespace discovery that
    backs status / graph / consolidate.
    """

    def test_missing_file_returns_none(self, tmp_path: Any) -> None:
        from trw_memory.integrations._backend import _read_namespace_metadata

        assert _read_namespace_metadata(tmp_path) is None

    def test_roundtrip_returns_namespace(self, tmp_path: Any) -> None:
        from trw_memory.integrations._backend import (
            _read_namespace_metadata,
            _write_namespace_metadata,
        )

        _write_namespace_metadata(tmp_path, "project:my-app")
        assert _read_namespace_metadata(tmp_path) == "project:my-app"

    def test_whitespace_only_returns_none(self, tmp_path: Any) -> None:
        from trw_memory.integrations._backend import (
            _NAMESPACE_METADATA_FILE,
            _read_namespace_metadata,
        )

        (tmp_path / _NAMESPACE_METADATA_FILE).write_text("  \n", encoding="utf-8")
        assert _read_namespace_metadata(tmp_path) is None

    def test_non_utf8_fails_open(self, tmp_path: Any) -> None:
        """A torn/partial write leaving non-UTF-8 bytes yields None, not a raise."""
        import structlog

        from trw_memory.integrations._backend import (
            _NAMESPACE_METADATA_FILE,
            _read_namespace_metadata,
        )

        # 0x80 is a UTF-8 continuation byte with no lead byte — invalid.
        (tmp_path / _NAMESPACE_METADATA_FILE).write_bytes(b"project:\x80\xffbad")
        with structlog.testing.capture_logs() as logs:
            assert _read_namespace_metadata(tmp_path) is None
        dropped = [r for r in logs if r["event"] == "namespace_metadata_read_failed"]
        assert len(dropped) == 1
        assert dropped[0]["error"] == "UnicodeDecodeError"
        # Content-free diagnostic: never leak the (sensitive) namespace bytes.
        assert "project" not in repr(dropped[0])

    def test_unreadable_sidecar_fails_open(self, tmp_path: Any) -> None:
        """An ``OSError`` (sidecar path is a directory) yields None, not a raise."""
        from trw_memory.integrations._backend import (
            _NAMESPACE_METADATA_FILE,
            _read_namespace_metadata,
        )

        # Make ``namespace.txt`` a directory so .exists() passes but read_text
        # raises OSError (IsADirectoryError).
        (tmp_path / _NAMESPACE_METADATA_FILE).mkdir()
        assert _read_namespace_metadata(tmp_path) is None
