# ruff: noqa: F401,F811
"""YAML field parity round-trip tests."""

from __future__ import annotations

import pytest

from trw_memory.models.memory import Assertion, AssertionType
from trw_memory.storage.yaml_backend import YAMLBackend

from ._test_yaml_field_parity_support import backend, make_entry


@pytest.mark.unit
class TestSyncFieldRoundTrip:
    def test_vector_clock_round_trip(self, backend: YAMLBackend) -> None:
        entry = make_entry("M-vc-1", "vector clock test", vector_clock={"node1": 3, "node2": 5})
        backend.store(entry)
        loaded = backend.get("M-vc-1", namespace="default")
        assert loaded is not None and loaded.vector_clock == {"node1": 3, "node2": 5}

    def test_remote_id_round_trip(self, backend: YAMLBackend) -> None:
        entry = make_entry("M-rid-1", "remote id test", remote_id="remote-abc-123")
        backend.store(entry)
        loaded = backend.get("M-rid-1", namespace="default")
        assert loaded is not None and loaded.remote_id == "remote-abc-123"

    def test_published_to_platform_round_trip(self, backend: YAMLBackend) -> None:
        entry = make_entry("M-pub-1", "published test", published_to_platform=True)
        backend.store(entry)
        loaded = backend.get("M-pub-1", namespace="default")
        assert loaded is not None and loaded.published_to_platform is True

    def test_pending_delete_round_trip(self, backend: YAMLBackend) -> None:
        entry = make_entry("M-pd-1", "pending delete test", pending_delete=True)
        backend.store(entry)
        loaded = backend.get("M-pd-1", namespace="default")
        assert loaded is not None and loaded.pending_delete is True

    def test_cross_validated_round_trip(self, backend: YAMLBackend) -> None:
        entry = make_entry("M-cv-1", "cross validated test", cross_validated=True)
        backend.store(entry)
        loaded = backend.get("M-cv-1", namespace="default")
        assert loaded is not None and loaded.cross_validated is True

    def test_outcome_history_round_trip(self, backend: YAMLBackend) -> None:
        entry = make_entry("M-oh-1", "outcome history test", outcome_history=["boost:+0.05", "decay:-0.10"])
        backend.store(entry)
        loaded = backend.get("M-oh-1", namespace="default")
        assert loaded is not None and loaded.outcome_history == ["boost:+0.05", "decay:-0.10"]

    def test_assertions_round_trip(self, backend: YAMLBackend) -> None:
        entry = make_entry(
            "M-assert-1",
            "assertions test",
            assertions=[
                Assertion(type=AssertionType.GREP_PRESENT, pattern="def test_", target="tests/**/*.py"),
                Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/**/*.py"),
            ],
        )
        backend.store(entry)
        loaded = backend.get("M-assert-1", namespace="default")
        assert loaded is not None
        assert len(loaded.assertions) == 2
        assert loaded.assertions[0].type == AssertionType.GREP_PRESENT
        assert loaded.assertions[0].pattern == "def test_"
        assert loaded.assertions[0].target == "tests/**/*.py"
        assert loaded.assertions[1].type == AssertionType.GLOB_EXISTS
        assert loaded.assertions[1].target == "src/**/*.py"


@pytest.mark.unit
class TestAllFieldsCombined:
    def test_all_seven_fields_round_trip(self, backend: YAMLBackend) -> None:
        entry = make_entry(
            "M-parity-1",
            "test field parity",
            vector_clock={"node1": 3, "node2": 5},
            remote_id="remote-abc",
            published_to_platform=True,
            pending_delete=True,
            cross_validated=True,
            outcome_history=["boost:+0.05", "decay:-0.10"],
            assertions=[Assertion(type=AssertionType.GREP_PRESENT, pattern="def test_", target="tests/**/*.py")],
        )
        backend.store(entry)
        loaded = backend.get("M-parity-1", namespace="default")
        assert loaded is not None
        assert loaded.vector_clock == {"node1": 3, "node2": 5}
        assert loaded.remote_id == "remote-abc"
        assert loaded.published_to_platform is True
        assert loaded.pending_delete is True
        assert loaded.cross_validated is True
        assert loaded.outcome_history == ["boost:+0.05", "decay:-0.10"]
        assert len(loaded.assertions) == 1 and loaded.assertions[0].pattern == "def test_"
