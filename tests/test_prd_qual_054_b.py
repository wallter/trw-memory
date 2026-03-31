"""Tests for PRD-QUAL-054 FR-04 through FR-07.

FR-04: Export completeness — entry_to_export_dict returns all MemoryEntry fields.
FR-05: Audit log durability — flush() called after each append.
FR-06: Key file path validation — rejects path traversal.
FR-07: Code quality — __all__ exports, MemoryConfig __repr__.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from trw_memory.cli_formatters import entry_to_export_dict
from trw_memory.models.memory import MemoryEntry, MemoryStatus


def _make_entry(**overrides: Any) -> MemoryEntry:
    """Create a MemoryEntry with sensible defaults for testing."""
    defaults: dict[str, Any] = {
        "id": "M-test123456789a",
        "content": "Test content",
        "detail": "Test detail",
        "tags": ["tag1", "tag2"],
        "evidence": ["evidence1"],
        "importance": 0.8,
        "status": MemoryStatus.ACTIVE,
        "recurrence": 2,
        "namespace": "test",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "last_accessed_at": datetime(2026, 1, 3, tzinfo=timezone.utc),
        "access_count": 5,
        "q_value": 0.7,
        "q_observations": 3,
        "source": "agent",
        "source_identity": "test-agent",
        "merged_from": ["M-old1"],
        "consolidated_from": ["M-old2"],
        "consolidated_into": "M-new1",
        "metadata": {"key": "value"},
        "vector_clock": {"node1": 1},
        "remote_id": "remote-123",
        "published_to_platform": True,
        "pending_delete": False,
        "cross_validated": True,
        "outcome_history": ["boost:+0.1"],
        "assertions": [],
    }
    defaults.update(overrides)
    return MemoryEntry(**defaults)


# -----------------------------------------------------------------------
# FR-04: Export completeness
# -----------------------------------------------------------------------


class TestExportCompleteness:
    """FR-04: entry_to_export_dict returns all MemoryEntry fields."""

    def test_export_dict_all_fields(self) -> None:
        """Exported dict must contain all 28 MemoryEntry.to_dict() keys."""
        entry = _make_entry()
        exported = entry_to_export_dict(entry)

        # Get the full to_dict output (no field filter)
        full_dict = entry.to_dict()

        # Every key in to_dict() must appear in the export
        missing = set(full_dict.keys()) - set(exported.keys())
        assert not missing, f"Export is missing fields: {missing}"

        # Values must match
        for key in full_dict:
            assert exported[key] == full_dict[key], f"Mismatch on field {key!r}"

    def test_export_dict_round_trip(self) -> None:
        """Exported dict should contain enough data to recreate a valid entry."""
        entry = _make_entry()
        exported = entry_to_export_dict(entry)

        # Must have the id and content at minimum
        assert "id" in exported
        assert "content" in exported
        assert exported["id"] == "M-test123456789a"
        assert exported["content"] == "Test content"

        # Must have provenance fields
        assert "source" in exported
        assert "source_identity" in exported

        # Must have sync fields
        assert "vector_clock" in exported
        assert "remote_id" in exported

        # Must have graph fields
        assert "cross_validated" in exported
        assert "outcome_history" in exported

        # Must have merge tracking
        assert "merged_from" in exported
        assert "consolidated_from" in exported
        assert "consolidated_into" in exported

        # Must have assertions
        assert "assertions" in exported


# -----------------------------------------------------------------------
# FR-05: Audit log durability
# -----------------------------------------------------------------------


class TestAuditFlush:
    """FR-05: audit log flush() is called after each write."""

    def test_audit_flush_called(self, tmp_path: Path) -> None:
        """Verify flush() is called after writing an audit record."""
        from trw_memory.security.audit import AuditLog

        log_path = tmp_path / "audit.jsonl"
        audit = AuditLog(log_path=log_path)

        # Patch open to track flush calls
        original_open = Path.open

        flush_called = False

        class TrackingWriter:
            """Wrapper that tracks flush() calls."""

            def __init__(self, fh: Any) -> None:
                self._fh = fh

            def write(self, data: str) -> int:
                return self._fh.write(data)

            def flush(self) -> None:
                nonlocal flush_called
                flush_called = True
                self._fh.flush()

            def __enter__(self) -> TrackingWriter:
                return self

            def __exit__(self, *args: Any) -> None:
                self._fh.__exit__(*args)

        def patched_open(self_path: Any, *args: Any, **kwargs: Any) -> Any:
            fh = original_open(self_path, *args, **kwargs)
            if "a" in (args[0] if args else kwargs.get("mode", "r")):
                return TrackingWriter(fh)
            return fh

        with patch.object(Path, "open", patched_open):
            audit.append(action="store", target_id="M-test1")

        assert flush_called, "flush() was not called after writing audit record"


# -----------------------------------------------------------------------
# FR-06: Key file path validation
# -----------------------------------------------------------------------


class TestKeyPathValidation:
    """FR-06: key file paths must reject traversal attacks."""

    def test_key_path_traversal_rejected(self) -> None:
        """Paths containing '..' must be rejected."""
        from trw_memory.security.keys import _validate_key_path

        with pytest.raises(Exception, match=r"[Tt]raversal"):
            _validate_key_path(Path("../../etc/passwd"))

    def test_key_path_traversal_in_middle_rejected(self) -> None:
        """Paths with '..' in the middle must be rejected."""
        from trw_memory.security.keys import _validate_key_path

        with pytest.raises(Exception, match=r"[Tt]raversal"):
            _validate_key_path(Path("/home/user/../../../etc/shadow"))

    def test_key_path_valid_accepted(self) -> None:
        """Valid paths like ~/.trw-memory/master.key must be accepted."""
        from trw_memory.security.keys import _validate_key_path

        result = _validate_key_path(Path("~/.trw-memory/master.key"))
        assert result.is_absolute(), "Result must be an absolute resolved path"
        assert ".." not in result.parts


# -----------------------------------------------------------------------
# FR-07: Code quality — __all__ and __repr__
# -----------------------------------------------------------------------


class TestCodeQuality:
    """FR-07: __all__ exports and MemoryConfig __repr__."""

    def test_client_has_all(self) -> None:
        """client module must define __all__."""
        import trw_memory.client as client_mod

        assert hasattr(client_mod, "__all__"), "client.py must define __all__"
        all_exports = client_mod.__all__
        assert "MemoryClient" in all_exports
        assert "MemoryResultDict" in all_exports
        assert "StoreResultDict" in all_exports
        assert "ForgetResultDict" in all_exports

    def test_graph_has_all(self) -> None:
        """graph module must define __all__."""
        import trw_memory.graph as graph_mod

        assert hasattr(graph_mod, "__all__"), "graph.py must define __all__"
        all_exports = graph_mod.__all__
        # Must include the main public functions
        assert "graph_query" in all_exports
        assert "create_similarity_edges" in all_exports
        assert "create_tag_cooccurrence_edges" in all_exports

    def test_memory_config_repr(self) -> None:
        """MemoryConfig must have a concise __repr__."""
        from trw_memory.models.config import MemoryConfig

        config = MemoryConfig()
        r = repr(config)
        # Must show key settings
        assert "sqlite" in r or "storage_backend" in r
        assert "encryption" in r.lower() or "encryption_enabled" in r
