"""Tests for YAML backend field parity with SQLite backend (PRD-FIX-058).

Verifies that all 7 previously-missing fields survive YAML round-trip:
- vector_clock (FR01/FR03)
- remote_id (FR01/FR04)
- published_to_platform (FR01/FR04)
- pending_delete (FR01/FR04)
- cross_validated (FR01/FR05)
- outcome_history (FR01/FR05)
- assertions (FR02/FR06)

Also tests backward compatibility: old YAML files without new fields
load with correct defaults.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import (
    Assertion,
    AssertionType,
    MemoryEntry,
    MemoryStatus,
)
from trw_memory.storage.yaml_backend import YAMLBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend(tmp_path: Path) -> YAMLBackend:
    return YAMLBackend(tmp_path / "entries")


# ---------------------------------------------------------------------------
# FR01 + FR03-FR06: round-trip all 7 fields
# ---------------------------------------------------------------------------


class TestSyncFieldRoundTrip:
    """FR01-FR06: All 7 previously-missing fields survive YAML round-trip."""

    def test_vector_clock_round_trip(self, backend: YAMLBackend) -> None:
        """FR01/FR03: vector_clock dict[str, int] survives serialisation."""
        entry = MemoryEntry(
            id="M-vc-1",
            content="vector clock test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            vector_clock={"node1": 3, "node2": 5},
        )
        backend.store(entry)
        loaded = backend.get("M-vc-1")
        assert loaded is not None
        assert loaded.vector_clock == {"node1": 3, "node2": 5}

    def test_remote_id_round_trip(self, backend: YAMLBackend) -> None:
        """FR01/FR04: remote_id string survives serialisation."""
        entry = MemoryEntry(
            id="M-rid-1",
            content="remote id test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            remote_id="remote-abc-123",
        )
        backend.store(entry)
        loaded = backend.get("M-rid-1")
        assert loaded is not None
        assert loaded.remote_id == "remote-abc-123"

    def test_published_to_platform_round_trip(self, backend: YAMLBackend) -> None:
        """FR01/FR04: published_to_platform bool survives serialisation."""
        entry = MemoryEntry(
            id="M-pub-1",
            content="published test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            published_to_platform=True,
        )
        backend.store(entry)
        loaded = backend.get("M-pub-1")
        assert loaded is not None
        assert loaded.published_to_platform is True

    def test_pending_delete_round_trip(self, backend: YAMLBackend) -> None:
        """FR01/FR04: pending_delete bool survives serialisation."""
        entry = MemoryEntry(
            id="M-pd-1",
            content="pending delete test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            pending_delete=True,
        )
        backend.store(entry)
        loaded = backend.get("M-pd-1")
        assert loaded is not None
        assert loaded.pending_delete is True

    def test_cross_validated_round_trip(self, backend: YAMLBackend) -> None:
        """FR01/FR05: cross_validated bool survives serialisation."""
        entry = MemoryEntry(
            id="M-cv-1",
            content="cross validated test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            cross_validated=True,
        )
        backend.store(entry)
        loaded = backend.get("M-cv-1")
        assert loaded is not None
        assert loaded.cross_validated is True

    def test_outcome_history_round_trip(self, backend: YAMLBackend) -> None:
        """FR01/FR05: outcome_history list[str] survives serialisation."""
        entry = MemoryEntry(
            id="M-oh-1",
            content="outcome history test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            outcome_history=["boost:+0.05", "decay:-0.10"],
        )
        backend.store(entry)
        loaded = backend.get("M-oh-1")
        assert loaded is not None
        assert loaded.outcome_history == ["boost:+0.05", "decay:-0.10"]

    def test_assertions_round_trip(self, backend: YAMLBackend) -> None:
        """FR02/FR06: assertions list[Assertion] survives serialisation."""
        entry = MemoryEntry(
            id="M-assert-1",
            content="assertions test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            assertions=[
                Assertion(
                    type=AssertionType.GREP_PRESENT,
                    pattern="def test_",
                    target="tests/**/*.py",
                ),
                Assertion(
                    type=AssertionType.GLOB_EXISTS,
                    pattern="",
                    target="src/**/*.py",
                ),
            ],
        )
        backend.store(entry)
        loaded = backend.get("M-assert-1")
        assert loaded is not None
        assert len(loaded.assertions) == 2
        assert loaded.assertions[0].pattern == "def test_"
        assert loaded.assertions[0].target == "tests/**/*.py"
        assert loaded.assertions[0].type == AssertionType.GREP_PRESENT
        assert loaded.assertions[1].type == AssertionType.GLOB_EXISTS
        assert loaded.assertions[1].target == "src/**/*.py"


class TestAllFieldsCombined:
    """Integration: all 7 fields set on one entry survive round-trip."""

    def test_all_seven_fields_round_trip(self, backend: YAMLBackend) -> None:
        entry = MemoryEntry(
            id="M-parity-1",
            content="test field parity",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            vector_clock={"node1": 3, "node2": 5},
            remote_id="remote-abc",
            published_to_platform=True,
            pending_delete=True,
            cross_validated=True,
            outcome_history=["boost:+0.05", "decay:-0.10"],
            assertions=[
                Assertion(
                    type=AssertionType.GREP_PRESENT,
                    pattern="def test_",
                    target="tests/**/*.py",
                )
            ],
        )

        backend.store(entry)
        loaded = backend.get("M-parity-1")

        assert loaded is not None
        assert loaded.vector_clock == {"node1": 3, "node2": 5}
        assert loaded.remote_id == "remote-abc"
        assert loaded.published_to_platform is True
        assert loaded.pending_delete is True
        assert loaded.cross_validated is True
        assert loaded.outcome_history == ["boost:+0.05", "decay:-0.10"]
        assert len(loaded.assertions) == 1
        assert loaded.assertions[0].pattern == "def test_"


# ---------------------------------------------------------------------------
# Backward compatibility: missing fields default correctly
# ---------------------------------------------------------------------------


class TestBackwardCompatDefaults:
    """Old YAML files without new fields load with correct defaults."""

    def test_missing_fields_have_defaults(self, backend: YAMLBackend) -> None:
        entry = MemoryEntry(
            id="M-old-1",
            content="legacy entry",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        backend.store(entry)
        loaded = backend.get("M-old-1")

        assert loaded is not None
        assert loaded.vector_clock == {}
        assert loaded.remote_id is None
        assert loaded.published_to_platform is False
        assert loaded.pending_delete is False
        assert loaded.cross_validated is False
        assert loaded.outcome_history == []
        assert loaded.assertions == []

    def test_remote_id_none_round_trip(self, backend: YAMLBackend) -> None:
        """remote_id=None stores and loads back as None, not empty string."""
        entry = MemoryEntry(
            id="M-none-rid",
            content="no remote id",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            remote_id=None,
        )
        backend.store(entry)
        loaded = backend.get("M-none-rid")
        assert loaded is not None
        assert loaded.remote_id is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the 7 new fields."""

    def test_empty_vector_clock_round_trip(self, backend: YAMLBackend) -> None:
        entry = MemoryEntry(
            id="M-empty-vc",
            content="empty vc",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            vector_clock={},
        )
        backend.store(entry)
        loaded = backend.get("M-empty-vc")
        assert loaded is not None
        assert loaded.vector_clock == {}

    def test_empty_outcome_history_round_trip(self, backend: YAMLBackend) -> None:
        entry = MemoryEntry(
            id="M-empty-oh",
            content="empty outcome",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            outcome_history=[],
        )
        backend.store(entry)
        loaded = backend.get("M-empty-oh")
        assert loaded is not None
        assert loaded.outcome_history == []

    def test_empty_assertions_round_trip(self, backend: YAMLBackend) -> None:
        entry = MemoryEntry(
            id="M-empty-assert",
            content="empty assertions",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            assertions=[],
        )
        backend.store(entry)
        loaded = backend.get("M-empty-assert")
        assert loaded is not None
        assert loaded.assertions == []

    def test_assertions_with_all_types(self, backend: YAMLBackend) -> None:
        """All AssertionType variants survive round-trip."""
        entry = MemoryEntry(
            id="M-all-types",
            content="all assertion types",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            assertions=[
                Assertion(type=AssertionType.GREP_PRESENT, pattern="def main", target="src/**/*.py"),
                Assertion(type=AssertionType.GREP_ABSENT, pattern="import os", target="src/**/*.py"),
                Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="README.md"),
                Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="*.bak"),
            ],
        )
        backend.store(entry)
        loaded = backend.get("M-all-types")
        assert loaded is not None
        assert len(loaded.assertions) == 4
        loaded_types = [a.type for a in loaded.assertions]
        assert AssertionType.GREP_PRESENT in loaded_types
        assert AssertionType.GREP_ABSENT in loaded_types
        assert AssertionType.GLOB_EXISTS in loaded_types
        assert AssertionType.GLOB_ABSENT in loaded_types

    def test_assertion_preserves_last_result(self, backend: YAMLBackend) -> None:
        """Assertion metadata fields (last_result, last_evidence) survive."""
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id="M-assert-meta",
            content="assertion metadata",
            created_at=now,
            updated_at=now,
            assertions=[
                Assertion(
                    type=AssertionType.GREP_PRESENT,
                    pattern="class Foo",
                    target="src/**/*.py",
                    last_result=True,
                    last_verified_at=now,
                    last_evidence="found at src/foo.py:42",
                ),
            ],
        )
        backend.store(entry)
        loaded = backend.get("M-assert-meta")
        assert loaded is not None
        assert len(loaded.assertions) == 1
        a = loaded.assertions[0]
        assert a.last_result is True
        assert a.last_evidence == "found at src/foo.py:42"
        assert a.last_verified_at is not None

    def test_vector_clock_large_values(self, backend: YAMLBackend) -> None:
        """vector_clock with many nodes and large counters."""
        vc = {f"node-{i}": i * 1000 for i in range(10)}
        entry = MemoryEntry(
            id="M-large-vc",
            content="large vector clock",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            vector_clock=vc,
        )
        backend.store(entry)
        loaded = backend.get("M-large-vc")
        assert loaded is not None
        assert loaded.vector_clock == vc


# ---------------------------------------------------------------------------
# Update path: verify update() preserves new fields
# ---------------------------------------------------------------------------


class TestUpdatePreservesNewFields:
    """After update(), the 7 new fields must not be lost."""

    def test_update_preserves_vector_clock(self, backend: YAMLBackend) -> None:
        entry = MemoryEntry(
            id="M-upd-vc",
            content="update test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            vector_clock={"node1": 10},
            published_to_platform=True,
            assertions=[
                Assertion(
                    type=AssertionType.GREP_PRESENT,
                    pattern="def test_",
                    target="tests/**/*.py",
                )
            ],
        )
        backend.store(entry)

        # Update an unrelated field
        result = backend.update("M-upd-vc", importance=0.9)
        assert result is not None
        assert result.vector_clock == {"node1": 10}
        assert result.published_to_platform is True
        assert len(result.assertions) == 1

        # Verify persistence: re-read from disk
        loaded = backend.get("M-upd-vc")
        assert loaded is not None
        assert loaded.vector_clock == {"node1": 10}
        assert loaded.published_to_platform is True
        assert len(loaded.assertions) == 1

    def test_update_assertions_directly(self, backend: YAMLBackend) -> None:
        """Regression: update(assertions=[...]) must not destroy Assertion objects.

        serialize_update_value() previously called str() on Assertion models,
        producing repr strings instead of dicts, silently dropping all assertions.
        """
        entry = MemoryEntry(
            id="M-upd-assert",
            content="direct assertion update test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        backend.store(entry)

        new_assertions = [
            Assertion(
                type=AssertionType.GLOB_EXISTS,
                pattern="",
                target="src/**/*.py",
            )
        ]
        result = backend.update("M-upd-assert", assertions=new_assertions)
        assert result is not None
        assert len(result.assertions) == 1
        assert result.assertions[0].target == "src/**/*.py"

        # Verify persistence
        loaded = backend.get("M-upd-assert")
        assert loaded is not None
        assert len(loaded.assertions) == 1
        assert loaded.assertions[0].target == "src/**/*.py"
