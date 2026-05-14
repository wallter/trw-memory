# ruff: noqa: F401,F811
"""Core SQLiteBackend storage behavior tests."""

from __future__ import annotations

import time

import pytest

from trw_memory.models.memory import Assertion, AssertionType, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_storage_sqlite_support import backend, make_entry


class TestStoreAndGet:
    def test_store_and_get_round_trip(self, backend: SQLiteBackend) -> None:
        entry = make_entry(
            "e1",
            "pydantic validation error",
            detail="Use model_config strict=True",
            tags=["pydantic", "validation"],
            importance=0.8,
            namespace="proj",
        )
        backend.store(entry)
        result = backend.get("e1")

        assert result is not None
        assert result.id == "e1"
        assert result.content == "pydantic validation error"
        assert result.detail == "Use model_config strict=True"
        assert result.tags == ["pydantic", "validation"]
        assert result.importance == pytest.approx(0.8)
        assert result.namespace == "proj"

    def test_get_preserves_status(self, backend: SQLiteBackend) -> None:
        entry = make_entry("e2", status=MemoryStatus.RESOLVED)
        backend.store(entry)
        result = backend.get("e2")
        assert result is not None
        assert result.status == MemoryStatus.RESOLVED

    def test_get_preserves_metadata(self, backend: SQLiteBackend) -> None:
        entry = make_entry("e3")
        entry = entry.model_copy(update={"metadata": {"sprint": "31", "pr": "42"}})
        backend.store(entry)
        result = backend.get("e3")
        assert result is not None
        assert result.metadata == {"sprint": "31", "pr": "42"}

    def test_get_preserves_evidence(self, backend: SQLiteBackend) -> None:
        entry = make_entry("e4")
        entry = entry.model_copy(update={"evidence": ["log1", "log2"]})
        backend.store(entry)
        result = backend.get("e4")
        assert result is not None
        assert result.evidence == ["log1", "log2"]


class TestGetNonexistent:
    def test_get_nonexistent_returns_none(self, backend: SQLiteBackend) -> None:
        result = backend.get("does-not-exist")
        assert result is None

    def test_get_empty_string_returns_none(self, backend: SQLiteBackend) -> None:
        result = backend.get("")
        assert result is None


class TestStoreOverwrite:
    def test_store_overwrites_existing_entry(self, backend: SQLiteBackend) -> None:
        entry_v1 = make_entry("dup", "first version")
        backend.store(entry_v1)

        entry_v2 = make_entry("dup", "second version — updated")
        backend.store(entry_v2)

        result = backend.get("dup")
        assert result is not None
        assert result.content == "second version — updated"

    def test_overwrite_preserves_new_fields(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("dup2", importance=0.3))
        backend.store(make_entry("dup2", importance=0.9))
        result = backend.get("dup2")
        assert result is not None
        assert result.importance == pytest.approx(0.9)


class TestDelete:
    def test_delete_removes_entry(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("del1"))
        assert backend.delete("del1")
        assert backend.get("del1") is None

    def test_delete_returns_true_when_existed(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("del2"))
        assert backend.delete("del2")

    def test_delete_nonexistent_returns_false(self, backend: SQLiteBackend) -> None:
        assert not backend.delete("no-such-entry")

    def test_double_delete_returns_false_second_time(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("del3"))
        backend.delete("del3")
        assert backend.delete("del3") is False


class TestCount:
    def test_count_empty_store(self, backend: SQLiteBackend) -> None:
        assert backend.count() == 0

    def test_count_returns_total(self, backend: SQLiteBackend) -> None:
        for i in range(3):
            backend.store(make_entry(f"c{i}"))
        assert backend.count() == 3

    def test_count_with_namespace(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("ns1a", namespace="ns1"))
        backend.store(make_entry("ns1b", namespace="ns1"))
        backend.store(make_entry("ns2a", namespace="ns2"))

        assert backend.count("ns1") == 2
        assert backend.count("ns2") == 1

    def test_count_namespace_no_match_returns_zero(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("x", namespace="other"))
        assert backend.count("nonexistent") == 0

    def test_count_decrements_after_delete(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("d1"))
        backend.store(make_entry("d2"))
        backend.delete("d1")
        assert backend.count() == 1


class TestListEntries:
    def test_list_entries_returns_all(self, backend: SQLiteBackend) -> None:
        for i in range(4):
            backend.store(make_entry(f"le{i}"))
        entries = backend.list_entries()
        assert len(entries) == 4

    def test_list_entries_filtered_by_status_active(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("active1", status=MemoryStatus.ACTIVE))
        backend.store(make_entry("active2", status=MemoryStatus.ACTIVE))
        backend.store(make_entry("resolved1", status=MemoryStatus.RESOLVED))

        results = backend.list_entries(status=MemoryStatus.ACTIVE)
        ids = {entry.id for entry in results}
        assert "active1" in ids
        assert "active2" in ids
        assert "resolved1" not in ids

    def test_list_entries_filtered_by_status_resolved(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("a", status=MemoryStatus.ACTIVE))
        backend.store(make_entry("r", status=MemoryStatus.RESOLVED))

        results = backend.list_entries(status=MemoryStatus.RESOLVED)
        assert len(results) == 1
        assert results[0].id == "r"

    def test_list_entries_respects_limit(self, backend: SQLiteBackend) -> None:
        for i in range(10):
            backend.store(make_entry(f"lim{i}"))
        results = backend.list_entries(limit=3)
        assert len(results) == 3

    def test_list_entries_empty_store_returns_empty(self, backend: SQLiteBackend) -> None:
        assert backend.list_entries() == []


class TestEntriesWithAssertions:
    def test_count_with_assertions_returns_only_assertion_entries(self, backend: SQLiteBackend) -> None:
        with_assertions = make_entry("a1").model_copy(
            update={"assertions": [Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/main.py")]}
        )
        without_assertions = make_entry("a2")

        backend.store(with_assertions)
        backend.store(without_assertions)

        results = backend.count_with_assertions()
        assert len(results) == 1
        assert results[0].id == "a1"


class TestIncrementSessionCounts:
    def test_increment_session_counts_updates_existing_rows(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("L-sess001"))
        backend.store(make_entry("L-sess002"))

        updated = backend.increment_session_counts(["L-sess001", "L-sess002"])

        assert updated == 2
        first = backend.get("L-sess001")
        second = backend.get("L-sess002")
        assert first is not None
        assert first.session_count == 1
        assert second is not None
        assert second.session_count == 1

    def test_increment_session_counts_uses_single_commit(self, backend: SQLiteBackend) -> None:
        backend.store(make_entry("L-batch01"))
        backend.store(make_entry("L-batch02"))
        backend.store(make_entry("L-batch03"))

        statements: list[str] = []
        backend._conn.set_trace_callback(statements.append)
        try:
            backend.increment_session_counts(["L-batch01", "L-batch02", "L-batch03"])
        finally:
            backend._conn.set_trace_callback(None)

        commit_count = sum(1 for statement in statements if statement.upper().startswith("COMMIT"))
        assert commit_count == 1

    def test_increment_session_counts_stays_under_latency_budget_for_25_rows(self, backend: SQLiteBackend) -> None:
        for index in range(25):
            backend.store(make_entry(f"L-lat{index:04d}"))

        entry_ids = [f"L-lat{index:04d}" for index in range(25)]

        start = time.perf_counter()
        updated = backend.increment_session_counts(entry_ids)
        elapsed = time.perf_counter() - start

        assert updated == 25
        assert elapsed < 0.05
