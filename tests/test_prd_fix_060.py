"""Tests for PRD-FIX-060: Structural Cleanup.

Covers all 5 FRs:
- FR-01: namespace.py shim removal
- FR-02: Consolidated entry serialization via to_dict()
- FR-03: Lock file cleanup
- FR-04: Dead ruff config removal
- FR-05: Boolean conversion simplification
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import Assertion, AssertionType, MemoryEntry, MemoryStatus

# ---------------------------------------------------------------------------
# FR-01: namespace.py shim removed
# ---------------------------------------------------------------------------


class TestFR01NamespaceShimRemoval:
    """Verify namespace.py is deleted and imports come from canonical modules."""

    def test_namespace_py_deleted(self) -> None:
        """The backward-compat shim namespace.py must no longer exist."""
        shim_path = Path(__file__).resolve().parent.parent / "src" / "trw_memory" / "namespace.py"
        assert not shim_path.exists(), f"namespace.py shim should be deleted but found at {shim_path}"

    def test_namespace_import_from_init(self) -> None:
        """from trw_memory import validate_namespace, namespace_to_path still works."""
        from trw_memory import namespace_to_path, validate_namespace

        assert validate_namespace("default") == "default"
        assert namespace_to_path("global") == Path("global")

    def test_imports_come_from_canonical_modules(self) -> None:
        """The __init__.py re-exports should come from namespaces.* not namespace."""
        import trw_memory

        # The module where the function is defined should be in namespaces.*
        validate_ns_module = trw_memory.validate_namespace.__module__
        ns_to_path_module = trw_memory.namespace_to_path.__module__
        assert "namespaces" in validate_ns_module, f"Expected namespaces module, got {validate_ns_module}"
        assert "namespaces" in ns_to_path_module, f"Expected namespaces module, got {ns_to_path_module}"


# ---------------------------------------------------------------------------
# FR-02: Consolidated entry serialization
# ---------------------------------------------------------------------------


class TestFR02EntryToDict:
    """Verify MemoryEntry.to_dict() method and its consumers."""

    @pytest.fixture()
    def sample_entry(self) -> MemoryEntry:
        """Create a fully-populated MemoryEntry for testing."""
        now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        return MemoryEntry(
            id="M-TEST-001",
            content="Test memory content",
            detail="Extended detail here",
            tags=["tag1", "tag2"],
            evidence=["evidence-1"],
            importance=0.8,
            status=MemoryStatus.ACTIVE,
            recurrence=3,
            namespace="project:test",
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
            access_count=5,
            q_value=0.7,
            q_observations=10,
            source="agent",
            source_identity="test-agent",
            merged_from=["M-OLD-001"],
            consolidated_from=["M-OLD-002"],
            consolidated_into=None,
            metadata={"key": "value"},
            vector_clock={"node1": 3},
            remote_id="R-001",
            published_to_platform=True,
            pending_delete=False,
            cross_validated=True,
            outcome_history=["boost:0.1"],
            assertions=[
                Assertion(
                    type=AssertionType.GREP_PRESENT,
                    pattern="test_pattern",
                    target="src/**/*.py",
                )
            ],
        )

    def test_memory_entry_to_dict_all_fields(self, sample_entry: MemoryEntry) -> None:
        """to_dict() includes all 48 fields of MemoryEntry (PRD-CORE-110/111 + PRD-INFRA-051)."""
        result = sample_entry.to_dict()

        expected_keys = {
            "id",
            "content",
            "detail",
            "tags",
            "evidence",
            "importance",
            "status",
            "recurrence",
            "namespace",
            "created_at",
            "updated_at",
            "last_accessed_at",
            "access_count",
            "q_value",
            "q_observations",
            "source",
            "source_identity",
            "client_profile",
            "model_id",
            "merged_from",
            "consolidated_from",
            "consolidated_into",
            "metadata",
            "vector_clock",
            "remote_id",
            "published_to_platform",
            "pending_delete",
            "sync_hash",
            "sync_seq",
            "last_synced_at",
            "cross_validated",
            # bi-temporal validity (PRD-CORE-194 FR01/FR02)
            "valid_from",
            "invalid_from",
            "invalidated_by",
            "outcome_history",
            "assertions",
            "anchors",
            "anchor_validity",
            "type",
            "nudge_line",
            "expires",
            "confidence",
            "task_type",
            "domain",
            "phase_origin",
            "phase_affinity",
            "team_origin",
            "protection_tier",
            "recall_count",
            "helpful_count",
            "unhelpful_count",
            "session_count",
            "verification_status",
            "verification_checked_at",
        }
        assert set(result.keys()) == expected_keys
        # PRD-CORE-244-FR08 dropped three unproduced attribution fields and
        # PRD-CORE-244-FR03 added verification_checked_at, in the schema-5 rebuild.
        assert len(result) == 54

        # Verify types of serialized values
        assert result["id"] == "M-TEST-001"
        assert result["content"] == "Test memory content"
        assert isinstance(result["tags"], list)
        assert isinstance(result["created_at"], str)  # ISO string
        assert isinstance(result["assertions"], list)
        assert result["published_to_platform"] is True
        assert result["cross_validated"] is True

    def test_memory_entry_to_dict_subset(self, sample_entry: MemoryEntry) -> None:
        """to_dict(fields=...) returns only the requested fields."""
        result = sample_entry.to_dict(fields={"id", "content"})

        assert set(result.keys()) == {"id", "content"}
        assert result["id"] == "M-TEST-001"
        assert result["content"] == "Test memory content"

    def test_memory_entry_to_dict_empty_subset(self, sample_entry: MemoryEntry) -> None:
        """to_dict(fields=set()) returns empty dict."""
        result = sample_entry.to_dict(fields=set())
        assert result == {}

    def test_memory_entry_to_dict_none_last_accessed(self) -> None:
        """to_dict handles None last_accessed_at correctly."""
        now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        entry = MemoryEntry(
            id="M-TEST-002",
            content="No access",
            created_at=now,
            updated_at=now,
        )
        result = entry.to_dict()
        assert result["last_accessed_at"] is None

    def test_yaml_backend_uses_to_dict(self, tmp_path: Path) -> None:
        """YAML backend store/get roundtrip still works after refactor."""
        from trw_memory.storage.yaml_backend import YAMLBackend

        now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        entry = MemoryEntry(
            id="M-YAML-001",
            content="YAML roundtrip test",
            tags=["yaml", "test"],
            importance=0.6,
            created_at=now,
            updated_at=now,
        )

        backend = YAMLBackend(tmp_path / "entries")
        backend.store(entry)
        result = backend.get("M-YAML-001", namespace="default")

        assert result is not None
        assert result.id == "M-YAML-001"
        assert result.content == "YAML roundtrip test"
        assert result.tags == ["yaml", "test"]
        assert result.importance == 0.6

    def test_export_dict_uses_to_dict(self, sample_entry: MemoryEntry) -> None:
        """entry_to_export_dict returns the expected subset of fields."""
        from trw_memory.cli_formatters import entry_to_export_dict

        result = entry_to_export_dict(sample_entry)

        # Should contain all MemoryEntry fields (lossless export)
        assert "id" in result
        assert "content" in result
        assert "assertions" in result  # was previously missing from export
        assert len(result) >= 27  # all fields present
        assert result["id"] == "M-TEST-001"
        assert result["importance"] == 0.8


# ---------------------------------------------------------------------------
# FR-03: Stable advisory lock inode
# ---------------------------------------------------------------------------


class TestFR03LockFilePersistence:
    """Verify lock files retain one stable inode across contenders."""

    def test_lock_file_remains_after_release(self, tmp_path: Path) -> None:
        """Deleting the lock file would let a third process bypass a waiter."""
        from trw_memory.storage.persistence import lock_for_rmw

        target_file = tmp_path / "test_entry.yaml"
        target_file.write_text("test: data\n")
        lock_path = tmp_path / "test_entry.yaml.lock"

        with lock_for_rmw(target_file):
            # Lock file should exist during the context
            assert lock_path.exists(), "Lock file should exist during lock_for_rmw"

        assert lock_path.exists(), "Lock file inode must remain stable across lock acquisitions"

    def test_lock_file_remains_after_exception(self, tmp_path: Path) -> None:
        """An exceptional holder still releases without replacing the inode."""
        from trw_memory.storage.persistence import lock_for_rmw

        target_file = tmp_path / "test_entry.yaml"
        target_file.write_text("test: data\n")
        lock_path = tmp_path / "test_entry.yaml.lock"

        with pytest.raises(RuntimeError, match="intentional"):
            with lock_for_rmw(target_file):
                raise RuntimeError("intentional error")

        assert lock_path.exists(), "Lock file inode must remain stable after exceptional release"


# ---------------------------------------------------------------------------
# FR-04: Dead ruff config removal
# ---------------------------------------------------------------------------


class TestFR04RuffConfig:
    """Verify ANN101/ANN102 are removed from pyproject.toml."""

    def test_ruff_no_ann101(self) -> None:
        """ANN101 should not appear in pyproject.toml ignore list."""
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        assert "ANN101" not in content, "ANN101 (deprecated self annotation) should be removed from ruff ignore"

    def test_ruff_no_ann102(self) -> None:
        """ANN102 should not appear in pyproject.toml ignore list."""
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        assert "ANN102" not in content, "ANN102 (deprecated cls annotation) should be removed from ruff ignore"


# ---------------------------------------------------------------------------
# FR-05: Boolean conversion simplification
# ---------------------------------------------------------------------------


class TestFR05BooleanConversion:
    """Verify _row_mapper boolean conversion handles SQLite 0/1 correctly."""

    def test_row_to_entry_boolean_fields(self) -> None:
        """row_to_entry correctly converts SQLite 0/1 to bool for boolean fields."""
        from trw_memory.storage._row_mapper import row_to_entry

        now_iso = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        # Build a row tuple matching ENTRY_COLUMNS order (52 columns)
        row = (
            "M-BOOL-001",  # id
            "content",  # content
            "detail",  # detail
            "[]",  # tags (JSON)
            "[]",  # evidence (JSON)
            0.5,  # importance
            "active",  # status
            1,  # recurrence
            "default",  # namespace
            now_iso,  # created_at
            now_iso,  # updated_at
            None,  # last_accessed_at
            None,  # valid_from (PRD-CORE-194)
            None,  # invalid_from (PRD-CORE-194)
            None,  # invalidated_by (PRD-CORE-194)
            0,  # access_count
            0,  # session_count
            0.5,  # q_value
            0,  # q_observations
            "agent",  # source
            "",  # source_identity
            "",  # client_profile
            "",  # model_id
            "[]",  # merged_from (JSON)
            "[]",  # consolidated_from (JSON)
            None,  # consolidated_into
            "{}",  # metadata (JSON)
            "{}",  # vector_clock (JSON)
            None,  # remote_id
            1,  # published_to_platform (SQLite int)
            0,  # pending_delete (SQLite int)
            1,  # cross_validated (SQLite int)
            "[]",  # outcome_history (JSON)
            "[]",  # assertions (JSON)
            "[]",  # anchors (JSON)
            1.0,  # anchor_validity
            "pattern",  # type
            "",  # nudge_line
            "",  # expires
            "unverified",  # confidence
            "",  # task_type
            "[]",  # domain_json
            "",  # phase_origin
            "[]",  # phase_affinity_json
            "",  # team_origin
            "normal",  # protection_tier
            "",  # sync_hash (PRD-INFRA-051)
            0,  # sync_seq (PRD-INFRA-051)
            None,  # last_synced_at (PRD-INFRA-051)
            0,  # recall_count (PRD-CORE-132)
            0,  # helpful_count (PRD-CORE-132)
            0,  # unhelpful_count (PRD-CORE-132)
            None,  # verification_status (PRD-CORE-231-FR02)
            "",  # verification_checked_at (PRD-CORE-244-FR03)
        )

        entry = row_to_entry(row)
        assert entry.published_to_platform is True
        assert entry.pending_delete is False
        assert entry.cross_validated is True

    def test_row_to_entry_boolean_none_values(self) -> None:
        """row_to_entry handles None for boolean fields (defaults to False)."""
        from trw_memory.storage._row_mapper import row_to_entry

        now_iso = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        row = (
            "M-BOOL-002",
            "content",
            "detail",
            "[]",
            "[]",
            0.5,
            "active",
            1,
            "default",
            now_iso,
            now_iso,
            None,
            None,  # valid_from (PRD-CORE-194)
            None,  # invalid_from (PRD-CORE-194)
            None,  # invalidated_by (PRD-CORE-194)
            0,  # access_count
            0,  # session_count
            0.5,
            0,
            "agent",
            "",
            "",  # client_profile
            "",  # model_id
            "[]",
            "[]",
            None,
            "{}",
            "{}",
            None,
            None,  # published_to_platform = None
            None,  # pending_delete = None
            None,  # cross_validated = None
            "[]",
            "[]",
            "[]",  # anchors_json
            None,  # anchor_validity
            "pattern",  # type_
            "",  # nudge_line
            "",  # expires
            "medium",  # confidence
            None,  # task_type
            "[]",  # domain_json
            None,  # phase_origin
            "[]",  # phase_affinity_json
            "",  # team_origin
            "normal",  # protection_tier
            "",  # sync_hash (PRD-INFRA-051)
            0,  # sync_seq (PRD-INFRA-051)
            None,  # last_synced_at (PRD-INFRA-051)
            0,  # recall_count (PRD-CORE-132)
            0,  # helpful_count (PRD-CORE-132)
            0,  # unhelpful_count (PRD-CORE-132)
            None,  # verification_status (PRD-CORE-231-FR02)
            "",  # verification_checked_at (PRD-CORE-244-FR03)
        )

        entry = row_to_entry(row)
        assert entry.published_to_platform is False
        assert entry.pending_delete is False
        assert entry.cross_validated is False

    def test_row_mapper_no_verbose_cast(self) -> None:
        """_row_mapper.py should not contain verbose bool(int(str(...))) patterns."""
        row_mapper_path = Path(__file__).resolve().parent.parent / "src" / "trw_memory" / "storage" / "_row_mapper.py"
        content = row_mapper_path.read_text()
        assert "bool(int(str(" not in content, "Verbose bool(int(str(...))) should be simplified to bool()"
