# ruff: noqa: F401,F811
"""YAML field parity update-path and malformed-data tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import Assertion, AssertionType, MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend

from ._test_yaml_field_parity_support import backend, write_entry_yaml


@pytest.mark.unit
class TestUpdatePreservesNewFields:
    def test_update_preserves_vector_clock(self, backend: YAMLBackend) -> None:
        entry = MemoryEntry(
            id="M-upd-vc",
            content="update test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            vector_clock={"node1": 10},
            published_to_platform=True,
            assertions=[Assertion(type=AssertionType.GREP_PRESENT, pattern="def test_", target="tests/**/*.py")],
        )
        backend.store(entry)
        result = backend.update("M-upd-vc", importance=0.9)
        assert result is not None
        assert result.vector_clock == {"node1": 10}
        assert result.published_to_platform is True
        assert len(result.assertions) == 1
        loaded = backend.get("M-upd-vc")
        assert loaded is not None
        assert loaded.vector_clock == {"node1": 10}
        assert loaded.published_to_platform is True
        assert len(loaded.assertions) == 1

    def test_update_assertions_directly(self, backend: YAMLBackend) -> None:
        entry = MemoryEntry(
            id="M-upd-assert",
            content="direct assertion update test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        backend.store(entry)
        result = backend.update(
            "M-upd-assert", assertions=[Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/**/*.py")]
        )
        assert result is not None
        assert len(result.assertions) == 1
        assert result.assertions[0].target == "src/**/*.py"
        loaded = backend.get("M-upd-assert")
        assert loaded is not None
        assert len(loaded.assertions) == 1
        assert loaded.assertions[0].target == "src/**/*.py"


@pytest.mark.unit
class TestMalformedAssertions:
    def test_assertions_with_non_dict_items(self, backend: YAMLBackend) -> None:
        data: dict[str, object] = {
            "id": "M-bad-assert",
            "content": "bad assertions",
            "created_at": "2026-01-15T10:00:00+00:00",
            "updated_at": "2026-01-15T10:00:00+00:00",
            "assertions": ["not_a_dict", 42, None],
        }
        write_entry_yaml(backend, "M-bad-assert", data)
        loaded = backend.get("M-bad-assert")
        assert loaded is not None and loaded.assertions == []

    def test_assertions_with_invalid_dict(self, backend: YAMLBackend) -> None:
        data: dict[str, object] = {
            "id": "M-mixed-assert",
            "content": "mixed assertions",
            "created_at": "2026-01-15T10:00:00+00:00",
            "updated_at": "2026-01-15T10:00:00+00:00",
            "assertions": [
                {"type": "grep_present", "pattern": "valid", "target": "src/**/*.py"},
                {"type": "invalid_type", "pattern": "bad", "target": "x"},
            ],
        }
        write_entry_yaml(backend, "M-mixed-assert", data)
        loaded = backend.get("M-mixed-assert")
        assert loaded is not None
        assert len(loaded.assertions) == 1
        assert loaded.assertions[0].pattern == "valid"


@pytest.mark.unit
class TestMalformedVectorClock:
    def test_vector_clock_with_non_int_value_degrades(self, backend: YAMLBackend) -> None:
        # A corrupt secondary store can hand back an already-parsed dict whose
        # value is not int-coercible. The load must fail open to an empty clock
        # rather than crashing the whole entry read with a ValueError.
        data: dict[str, object] = {
            "id": "M-bad-vc",
            "content": "bad vector clock",
            "created_at": "2026-01-15T10:00:00+00:00",
            "updated_at": "2026-01-15T10:00:00+00:00",
            "vector_clock": {"node1": "not-an-int"},
        }
        write_entry_yaml(backend, "M-bad-vc", data)
        loaded = backend.get("M-bad-vc")
        assert loaded is not None
        assert loaded.vector_clock == {}

    def test_vector_clock_with_null_value_degrades(self, backend: YAMLBackend) -> None:
        data: dict[str, object] = {
            "id": "M-null-vc",
            "content": "null vector clock value",
            "created_at": "2026-01-15T10:00:00+00:00",
            "updated_at": "2026-01-15T10:00:00+00:00",
            "vector_clock": {"node1": None},
        }
        write_entry_yaml(backend, "M-null-vc", data)
        loaded = backend.get("M-null-vc")
        assert loaded is not None
        assert loaded.vector_clock == {}

    def test_valid_vector_clock_still_loads(self, backend: YAMLBackend) -> None:
        data: dict[str, object] = {
            "id": "M-good-vc",
            "content": "valid vector clock",
            "created_at": "2026-01-15T10:00:00+00:00",
            "updated_at": "2026-01-15T10:00:00+00:00",
            "vector_clock": {"node1": 7},
        }
        write_entry_yaml(backend, "M-good-vc", data)
        loaded = backend.get("M-good-vc")
        assert loaded is not None
        assert loaded.vector_clock == {"node1": 7}


@pytest.mark.unit
class TestSQLiteAssertionsUpdate:
    def test_sqlite_update_assertions_directly(self, tmp_path: Path) -> None:
        db = SQLiteBackend(tmp_path / "test.db")
        entry = MemoryEntry(
            id="M-sql-assert",
            content="sqlite assertions update",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.store(entry)
        result = db.update(
            "M-sql-assert", assertions=[Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/**/*.py")]
        )
        assert result is not None
        assert len(result.assertions) == 1
        assert result.assertions[0].target == "src/**/*.py"
        loaded = db.get("M-sql-assert")
        assert loaded is not None
        assert len(loaded.assertions) == 1
        db.close()
