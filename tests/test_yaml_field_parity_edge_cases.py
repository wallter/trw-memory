# ruff: noqa: F401,F811
"""YAML field parity edge-case tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trw_memory.models.memory import Assertion, AssertionType, MemoryEntry
from trw_memory.storage.yaml_backend import YAMLBackend

from ._test_yaml_field_parity_support import backend, make_entry


@pytest.mark.unit
class TestEdgeCases:
    def test_empty_vector_clock_round_trip(self, backend: YAMLBackend) -> None:
        entry = make_entry("M-empty-vc", "empty vc")
        backend.store(entry)
        loaded = backend.get("M-empty-vc", namespace="default")
        assert loaded is not None and loaded.vector_clock == {}

    def test_empty_outcome_history_round_trip(self, backend: YAMLBackend) -> None:
        entry = make_entry("M-empty-oh", "empty outcome")
        backend.store(entry)
        loaded = backend.get("M-empty-oh", namespace="default")
        assert loaded is not None and loaded.outcome_history == []

    def test_empty_assertions_round_trip(self, backend: YAMLBackend) -> None:
        entry = make_entry("M-empty-assert", "empty assertions")
        backend.store(entry)
        loaded = backend.get("M-empty-assert", namespace="default")
        assert loaded is not None and loaded.assertions == []

    def test_assertions_with_all_types(self, backend: YAMLBackend) -> None:
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
        loaded = backend.get("M-all-types", namespace="default")
        assert loaded is not None
        assert len(loaded.assertions) == 4
        loaded_types = [assertion.type for assertion in loaded.assertions]
        assert AssertionType.GREP_PRESENT in loaded_types
        assert AssertionType.GREP_ABSENT in loaded_types
        assert AssertionType.GLOB_EXISTS in loaded_types
        assert AssertionType.GLOB_ABSENT in loaded_types

    def test_assertion_preserves_last_result(self, backend: YAMLBackend) -> None:
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
        loaded = backend.get("M-assert-meta", namespace="default")
        assert loaded is not None
        assert len(loaded.assertions) == 1
        assertion = loaded.assertions[0]
        assert assertion.last_result is True
        assert assertion.last_evidence == "found at src/foo.py:42"
        assert assertion.last_verified_at is not None

    def test_vector_clock_large_values(self, backend: YAMLBackend) -> None:
        vector_clock = {f"node-{i}": i * 1000 for i in range(10)}
        entry = make_entry("M-large-vc", "large vector clock", vector_clock=vector_clock)
        backend.store(entry)
        loaded = backend.get("M-large-vc", namespace="default")
        assert loaded is not None and loaded.vector_clock == vector_clock
