# ruff: noqa: F401,F811
"""SQLiteBackend update-path tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_storage_sqlite_support import backend, make_entry


class TestUpdate:
    def test_update_importance(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u1", importance=0.3))
        result = backend.update("u1", importance=0.9, namespace="default")
        assert result is not None
        assert result.importance == pytest.approx(0.9)

    def test_update_nonexistent_returns_none(self, backend: SQLiteBackend) -> None:
        result = backend.update("no-such-id", importance=0.9, namespace="default")
        assert result is None

    def test_update_no_fields_returns_current(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u2", "original"))
        result = backend.update("u2", namespace="default")
        assert result is not None
        assert result.content == "original"

    def test_update_status(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u3", status=MemoryStatus.ACTIVE))
        result = backend.update("u3", status=MemoryStatus.RESOLVED, namespace="default")
        assert result is not None
        assert result.status == MemoryStatus.RESOLVED

    def test_update_tags(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u4", tags=["old"]))
        result = backend.update("u4", tags=["new", "updated"], namespace="default")
        assert result is not None
        assert result.tags == ["new", "updated"]


class TestUpdateBranchCoverage:
    """FR03: Branch coverage for update() with various field types."""

    def test_update_list_field_tags(self, backend: SQLiteBackend) -> None:
        """update(entry_id, tags=['new']) serializes tags as JSON."""
        backend.store(make_entry("u-tags", tags=["old"]))
        result = backend.update("u-tags", tags=["new", "added"], namespace="default")
        assert result is not None
        assert result.tags == ["new", "added"]

    def test_update_list_field_evidence(self, backend: SQLiteBackend) -> None:
        """update with evidence list serializes correctly."""
        backend.store(make_entry("u-ev"))
        result = backend.update("u-ev", evidence=["log1.txt", "trace.json"], namespace="default")
        assert result is not None
        assert result.evidence == ["log1.txt", "trace.json"]

    def test_update_dict_field_metadata(self, backend: SQLiteBackend) -> None:
        """update(entry_id, metadata={'key': 'val'}) serializes as JSON."""
        backend.store(make_entry("u-meta"))
        result = backend.update("u-meta", metadata={"sprint": "52", "agent": "tm-b"}, namespace="default")
        assert result is not None
        assert result.metadata == {"sprint": "52", "agent": "tm-b"}

    def test_update_dict_field_vector_clock(self, backend: SQLiteBackend) -> None:
        """update with vector_clock dict serializes correctly."""
        backend.store(make_entry("u-vc"))
        result = backend.update("u-vc", vector_clock={"node-1": 3, "node-2": 1}, namespace="default")
        assert result is not None
        assert result.vector_clock["node-1"] == 3
        assert result.vector_clock["node-2"] == 1

    def test_update_datetime_field(self, backend: SQLiteBackend) -> None:
        """update with datetime value converts to isoformat."""
        backend.store(make_entry("u-dt"))
        new_time = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
        result = backend.update("u-dt", last_accessed_at=new_time, namespace="default")
        assert result is not None
        assert result.last_accessed_at is not None
        assert result.last_accessed_at.year == 2026
        assert result.last_accessed_at.month == 3

    def test_update_memory_status_enum(self, backend: SQLiteBackend) -> None:
        """update with MemoryStatus.RESOLVED extracts .value correctly."""
        backend.store(make_entry("u-status", status=MemoryStatus.ACTIVE))
        result = backend.update("u-status", status=MemoryStatus.RESOLVED, namespace="default")
        assert result is not None
        assert result.status == MemoryStatus.RESOLVED

    def test_update_invalid_column_raises(self, backend: SQLiteBackend) -> None:
        """update(entry_id, id='hacked') raises StorageError for immutable field."""
        backend.store(make_entry("u-invalid"))
        with pytest.raises(StorageError, match="Invalid update field"):
            backend.update("u-invalid", id="hacked", namespace="default")

    def test_update_auto_sets_updated_at(self, backend: SQLiteBackend) -> None:
        """update without explicit updated_at auto-sets it to now."""
        entry = make_entry("u-auto")
        backend.store(entry)
        original_updated = entry.updated_at

        result = backend.update("u-auto", importance=0.9, namespace="default")
        assert result is not None
        assert result.updated_at >= original_updated


class TestUpdateEnumStringValidation:
    """memory-storage-4: raw strings for enum-typed fields are validated.

    An invalid raw string must be rejected at write time rather than persisting
    and making the row un-deserializable (permanent quarantine) on next read.
    """

    def test_update_invalid_status_string_rejected(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u-bad-status", status=MemoryStatus.ACTIVE))
        with pytest.raises(StorageError, match="acttive"):
            backend.update("u-bad-status", status="acttive", namespace="default")
        # The bad write must not have landed — the row still reads back cleanly.
        reread = backend.get("u-bad-status", namespace="default")
        assert reread is not None
        assert reread.status == MemoryStatus.ACTIVE

    def test_update_valid_status_string_persists_as_value(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u-good-status", status=MemoryStatus.ACTIVE))
        result = backend.update("u-good-status", status="resolved", namespace="default")
        assert result is not None
        assert result.status == MemoryStatus.RESOLVED

    def test_update_invalid_confidence_string_rejected(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u-bad-conf"))
        with pytest.raises(StorageError, match="not-a-confidence"):
            backend.update("u-bad-conf", confidence="not-a-confidence", namespace="default")
        assert backend.get("u-bad-conf", namespace="default") is not None

    def test_update_invalid_protection_tier_string_rejected(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u-bad-tier"))
        with pytest.raises(StorageError, match="not-a-tier"):
            backend.update("u-bad-tier", protection_tier="not-a-tier", namespace="default")

    def test_update_invalid_type_string_rejected(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("u-bad-type"))
        with pytest.raises(StorageError, match="not-a-type"):
            backend.update("u-bad-type", type="not-a-type", namespace="default")
