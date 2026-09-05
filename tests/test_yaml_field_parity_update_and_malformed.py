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
        result = backend.update("M-upd-vc", importance=0.9, namespace="default")
        assert result is not None
        assert result.vector_clock == {"node1": 10}
        assert result.published_to_platform is True
        assert len(result.assertions) == 1
        loaded = backend.get("M-upd-vc", namespace="default")
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
            "M-upd-assert",
            assertions=[Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/**/*.py")],
            namespace="default",
        )
        assert result is not None
        assert len(result.assertions) == 1
        assert result.assertions[0].target == "src/**/*.py"
        loaded = backend.get("M-upd-assert", namespace="default")
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
        loaded = backend.get("M-bad-assert", namespace="default")
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
        loaded = backend.get("M-mixed-assert", namespace="default")
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
        loaded = backend.get("M-bad-vc", namespace="default")
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
        loaded = backend.get("M-null-vc", namespace="default")
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
        loaded = backend.get("M-good-vc", namespace="default")
        assert loaded is not None
        assert loaded.vector_clock == {"node1": 7}


@pytest.mark.unit
class TestAnchorValidity:
    def test_zero_validity_survives_round_trip(self, backend: YAMLBackend) -> None:
        # anchor_validity=0.0 means "all code anchors stale" — a real signal.
        # The old ``float(raw) if raw else 1.0`` falsy-check resurrected it to
        # 1.0 (fresh), inverting staleness. It must round-trip as 0.0.
        data: dict[str, object] = {
            "id": "M-anc-zero",
            "content": "all anchors stale",
            "created_at": "2026-01-15T10:00:00+00:00",
            "updated_at": "2026-01-15T10:00:00+00:00",
            "anchor_validity": 0.0,
        }
        write_entry_yaml(backend, "M-anc-zero", data)
        loaded = backend.get("M-anc-zero", namespace="default")
        assert loaded is not None
        assert loaded.anchor_validity == 0.0

    def test_corrupt_validity_degrades_to_default(self, backend: YAMLBackend) -> None:
        data: dict[str, object] = {
            "id": "M-anc-bad",
            "content": "corrupt validity",
            "created_at": "2026-01-15T10:00:00+00:00",
            "updated_at": "2026-01-15T10:00:00+00:00",
            "anchor_validity": "not-a-number",
        }
        write_entry_yaml(backend, "M-anc-bad", data)
        loaded = backend.get("M-anc-bad", namespace="default")
        assert loaded is not None
        # PRD-CORE-244-FR01: an unparseable score is "never assessed", not a
        # perfect 1.0 the entry never earned.
        assert loaded.anchor_validity is None

    def test_missing_validity_uses_default(self, backend: YAMLBackend) -> None:
        data: dict[str, object] = {
            "id": "M-anc-missing",
            "content": "no validity field",
            "created_at": "2026-01-15T10:00:00+00:00",
            "updated_at": "2026-01-15T10:00:00+00:00",
        }
        write_entry_yaml(backend, "M-anc-missing", data)
        loaded = backend.get("M-anc-missing", namespace="default")
        assert loaded is not None
        assert loaded.anchor_validity is None


@pytest.mark.unit
class TestCorruptTimestampDegrades:
    """A byte-shifted / unparseable timestamp must NOT crash entry load.

    Regression for the 2026-06-10 corruption class: a single row carried a
    mangled ISO timestamp ``'026-04-13T00:00:00+00:002'`` (leading '2' lost,
    stray trailing '2' — a SQLite 3.51.1 WAL-reset byte-shift). The unguarded
    ``parse_dt`` in ``_dict_to_entry`` raised ``Invalid isoformat string`` and
    took down the whole listing. Loading one corrupt entry must degrade the
    bad field to a sentinel + WARN, consistent with how the same mapper
    already fail-opens status / anchors / ints / floats.
    """

    _BAD_TS = "026-04-13T00:00:00+00:002"

    def test_corrupt_created_at_degrades_to_now(self, backend: YAMLBackend) -> None:
        data: dict[str, object] = {
            "id": "M-ts-bad-created",
            "content": "byte-shifted created_at",
            "created_at": self._BAD_TS,
            "updated_at": "2026-04-13T00:00:00+00:00",
        }
        write_entry_yaml(backend, "M-ts-bad-created", data)
        loaded = backend.get("M-ts-bad-created", namespace="default")
        # Entry survives (no crash); the corrupt field degrades to a usable
        # tz-aware datetime rather than raising.
        assert loaded is not None
        assert loaded.created_at.tzinfo is not None
        # The valid sibling field is preserved untouched.
        assert loaded.updated_at.year == 2026

    def test_corrupt_updated_at_degrades(self, backend: YAMLBackend) -> None:
        data: dict[str, object] = {
            "id": "M-ts-bad-updated",
            "content": "byte-shifted updated_at",
            "created_at": "2026-04-13T00:00:00+00:00",
            "updated_at": self._BAD_TS,
        }
        write_entry_yaml(backend, "M-ts-bad-updated", data)
        loaded = backend.get("M-ts-bad-updated", namespace="default")
        assert loaded is not None
        assert loaded.updated_at.tzinfo is not None

    def test_corrupt_last_accessed_degrades_to_none(self, backend: YAMLBackend) -> None:
        data: dict[str, object] = {
            "id": "M-ts-bad-accessed",
            "content": "byte-shifted last_accessed_at",
            "created_at": "2026-04-13T00:00:00+00:00",
            "updated_at": "2026-04-13T00:00:00+00:00",
            "last_accessed_at": self._BAD_TS,
        }
        write_entry_yaml(backend, "M-ts-bad-accessed", data)
        loaded = backend.get("M-ts-bad-accessed", namespace="default")
        assert loaded is not None
        # An unparseable optional timestamp degrades to None, not a crash.
        assert loaded.last_accessed_at is None

    def test_one_corrupt_entry_does_not_collapse_listing(self, backend: YAMLBackend) -> None:
        good: dict[str, object] = {
            "id": "M-ts-good",
            "content": "valid entry",
            "created_at": "2026-04-13T00:00:00+00:00",
            "updated_at": "2026-04-13T00:00:00+00:00",
        }
        bad: dict[str, object] = {
            "id": "M-ts-bad",
            "content": "corrupt entry",
            "created_at": self._BAD_TS,
            "updated_at": "2026-04-13T00:00:00+00:00",
        }
        write_entry_yaml(backend, "M-ts-good", good)
        write_entry_yaml(backend, "M-ts-bad", bad)
        entries = backend.list_entries()
        # Both load — the good one always, and the corrupt one now degrades
        # in-place instead of being silently dropped by the catch-all skip.
        ids = {e.id for e in entries}
        assert "M-ts-good" in ids
        assert "M-ts-bad" in ids


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
            "M-sql-assert",
            assertions=[Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/**/*.py")],
            namespace="default",
        )
        assert result is not None
        assert len(result.assertions) == 1
        assert result.assertions[0].target == "src/**/*.py"
        loaded = db.get("M-sql-assert", namespace="default")
        assert loaded is not None
        assert len(loaded.assertions) == 1
        db.close()


@pytest.mark.integration
class TestSQLiteCorruptTimestampDegrades:
    """SQLite row mapper must fail open on a corrupt timestamp like yaml_backend.

    Parity with TestCorruptTimestampDegrades: the 2026-06-10 WAL-reset byte-shift
    corruption class (``'026-04-13T00:00:00+00:002'``) previously crashed the
    whole SQLite ``get``/``list_entries`` read because ``_row_mapper`` used the
    strict ``parse_dt``. It now degrades the bad field instead of raising.
    """

    _BAD_TS = "026-04-13T00:00:00+00:002"

    def _store(self, db: SQLiteBackend, entry_id: str) -> None:
        db.store(
            MemoryEntry(
                id=entry_id,
                content=f"content for {entry_id}",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        )

    def test_corrupt_created_at_does_not_crash_get(self, tmp_path: Path) -> None:
        db = SQLiteBackend(tmp_path / "test.db")
        self._store(db, "M-sql-ts-bad")
        # Corrupt the persisted created_at directly, mimicking the WAL-reset shift.
        db._conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            (self._BAD_TS, "M-sql-ts-bad"),
        )
        db._conn.commit()

        loaded = db.get("M-sql-ts-bad", namespace="default")

        assert loaded is not None  # no ValueError / crash
        assert loaded.created_at.tzinfo is not None  # degraded to a usable tz-aware datetime
        assert loaded.updated_at.year >= 2026  # valid sibling field untouched
        db.close()

    def test_one_corrupt_row_does_not_collapse_listing(self, tmp_path: Path) -> None:
        db = SQLiteBackend(tmp_path / "test.db")
        self._store(db, "M-sql-good")
        self._store(db, "M-sql-bad")
        db._conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            (self._BAD_TS, "M-sql-bad"),
        )
        db._conn.commit()

        ids = {e.id for e in db.list_entries()}

        assert "M-sql-good" in ids
        assert "M-sql-bad" in ids  # corrupt row degrades in place, not dropped
        db.close()
