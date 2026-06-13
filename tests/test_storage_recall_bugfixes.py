"""Behavior tests for the STORAGE/LIFECYCLE recall bug fixes.

Covers the read-only-audit findings:

- F6  — expired entries (``expires`` parses as a PAST ISO datetime) are dropped
        from the recall path; empty / non-date / future ``expires`` are kept.
- F7  — ``entries_with_assertions`` filters to ``status='active'`` by default so
        obsolete entries' stale assertions don't pollute the summary.
- F11 — updating an entry's status to a terminal (non-active) value removes its
        dense vector from the KNN index (entry row preserved for audit).
- F-008 — ``record_recall_access`` batches the recall bookkeeping into ONE
        commit instead of a per-entry get+update loop, with correct increments.
- F-007 — composite recall indexes exist after schema migration.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trw_memory.lifecycle._recall import (
    _expires_in_past,
    drop_expired_entries,
    rank_by_utility,
    record_recall_access,
)
from trw_memory.models.memory import (
    Assertion,
    AssertionType,
    MemoryEntry,
    MemoryStatus,
)
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# F6 — expiry filtering on the recall path
# ---------------------------------------------------------------------------


class TestF6ExpiryFiltering:
    def test_past_iso_date_is_expired(self) -> None:
        assert _expires_in_past("2025-01-01") is True

    def test_past_iso_datetime_with_offset_is_expired(self) -> None:
        assert _expires_in_past("2025-01-01T00:00:00+00:00") is True

    def test_past_iso_datetime_with_z_is_expired(self) -> None:
        assert _expires_in_past("2025-01-01T00:00:00Z") is True

    def test_future_iso_date_is_not_expired(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=365)).date().isoformat()
        assert _expires_in_past(future) is False

    def test_empty_expires_is_not_expired(self) -> None:
        assert _expires_in_past("") is False
        assert _expires_in_past("   ") is False

    def test_non_date_condition_is_not_expired(self) -> None:
        # Free-form condition strings must never be treated as expired.
        assert _expires_in_past("when the migration ships") is False
        assert _expires_in_past("never") is False
        assert _expires_in_past("2025-13-99") is False

    def test_non_string_is_not_expired(self) -> None:
        assert _expires_in_past(None) is False
        assert _expires_in_past(12345) is False

    def test_drop_expired_excludes_only_past_dates(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=10)).date().isoformat()
        matches: list[dict[str, object]] = [
            {"id": "past", "expires": "2025-01-01"},
            {"id": "empty", "expires": ""},
            {"id": "condition", "expires": "when X done"},
            {"id": "future", "expires": future},
            {"id": "missing"},
        ]
        kept_ids = {str(e["id"]) for e in drop_expired_entries(matches)}
        assert kept_ids == {"empty", "condition", "future", "missing"}
        assert "past" not in kept_ids

    def test_rank_by_utility_drops_expired(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=10)).date().isoformat()
        matches: list[dict[str, object]] = [
            {"id": "past", "content": "stale", "expires": "2025-01-01", "importance": 0.9},
            {"id": "fresh", "content": "fresh", "expires": future, "importance": 0.5},
            {"id": "plain", "content": "plain"},
        ]
        ranked = rank_by_utility(matches, ["fresh"], lambda_weight=0.4)
        ranked_ids = {str(e["id"]) for e in ranked}
        assert "past" not in ranked_ids
        assert ranked_ids == {"fresh", "plain"}

    def test_drop_expired_empty_input(self) -> None:
        assert drop_expired_entries([]) == []


# ---------------------------------------------------------------------------
# F7 — entries_with_assertions status filter
# ---------------------------------------------------------------------------


def _entry_with_assertion(entry_id: str, status: MemoryStatus) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=f"entry {entry_id}",
        status=status,
        assertions=[Assertion(type=AssertionType.GREP_PRESENT, target="bar.py", pattern="foo")],
    )


class TestF7AssertionStatusFilter:
    def test_excludes_obsolete_entries_by_default(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            backend.store(_entry_with_assertion("a-active", MemoryStatus.ACTIVE))
            backend.store(_entry_with_assertion("a-obsolete", MemoryStatus.OBSOLETE))
            backend.store(_entry_with_assertion("a-archived", MemoryStatus.ARCHIVED))

            result_ids = {e.id for e in backend.entries_with_assertions()}
            assert result_ids == {"a-active"}
        finally:
            backend.close()

    def test_status_none_includes_all_statuses(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            backend.store(_entry_with_assertion("b-active", MemoryStatus.ACTIVE))
            backend.store(_entry_with_assertion("b-obsolete", MemoryStatus.OBSOLETE))

            result_ids = {e.id for e in backend.entries_with_assertions(status=None)}
            assert result_ids == {"b-active", "b-obsolete"}
        finally:
            backend.close()

    def test_explicit_status_filter(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            backend.store(_entry_with_assertion("c-active", MemoryStatus.ACTIVE))
            backend.store(_entry_with_assertion("c-obsolete", MemoryStatus.OBSOLETE))

            result_ids = {e.id for e in backend.entries_with_assertions(status=MemoryStatus.OBSOLETE)}
            assert result_ids == {"c-obsolete"}
        finally:
            backend.close()

    def test_entries_without_assertions_excluded(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            backend.store(_entry_with_assertion("d-active", MemoryStatus.ACTIVE))
            backend.store(MemoryEntry(id="d-noassert", content="no assertions", status=MemoryStatus.ACTIVE))

            result_ids = {e.id for e in backend.entries_with_assertions()}
            assert result_ids == {"d-active"}
        finally:
            backend.close()

    def test_entries_ordered_by_updated_at_desc(self) -> None:
        """entries_with_assertions must return rows in updated_at DESC order.

        The primary SQL path previously had no ORDER BY clause, giving
        non-deterministic ordering while the resilient fallback path DID
        order by updated_at DESC. This verifies the primary path is now
        consistent with the documented ordering contract.
        """
        from datetime import timedelta

        backend = SQLiteBackend(Path(":memory:"))
        try:
            now = datetime.now(timezone.utc)
            older = now - timedelta(seconds=10)

            old_entry = _entry_with_assertion("e-old", MemoryStatus.ACTIVE)
            old_entry = old_entry.model_copy(update={"updated_at": older})
            backend.store(old_entry)

            new_entry = _entry_with_assertion("e-new", MemoryStatus.ACTIVE)
            new_entry = new_entry.model_copy(update={"updated_at": now})
            backend.store(new_entry)

            results = backend.entries_with_assertions()
            assert len(results) == 2
            # Most recently updated must come first.
            assert results[0].id == "e-new", f"Expected 'e-new' first (newer updated_at) but got {results[0].id!r}"
            assert results[1].id == "e-old"
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# F11 — stale vector removed on terminal-status update
# ---------------------------------------------------------------------------


class TestF11VectorPruneOnTerminalStatus:
    def _backend(self) -> SQLiteBackend:
        pytest.importorskip("sqlite_vec")
        backend = SQLiteBackend(Path(":memory:"))
        if not backend.vec_available:
            backend.close()
            pytest.skip("sqlite-vec virtual table unavailable")
        return backend

    def test_vector_deleted_when_marked_obsolete(self) -> None:
        backend = self._backend()
        try:
            backend.store(MemoryEntry(id="v-1", content="vector entry"))
            backend.upsert_vector("v-1", [0.1] * backend._dim)
            assert backend.vector_exists("v-1") is True

            backend.update("v-1", status="obsolete")

            # Vector pruned, but the entry row is preserved for audit.
            assert backend.vector_exists("v-1") is False
            row = backend.get("v-1")
            assert row is not None
            assert row.status == MemoryStatus.OBSOLETE
        finally:
            backend.close()

    def test_vector_deleted_for_archived_and_poisoned_and_resolved(self) -> None:
        for idx, terminal in enumerate(["archived", "obsolete_poisoned", "resolved"]):
            backend = self._backend()
            try:
                eid = f"v-term-{idx}"
                backend.store(MemoryEntry(id=eid, content="entry"))
                backend.upsert_vector(eid, [0.2] * backend._dim)
                assert backend.vector_exists(eid) is True

                backend.update(eid, status=terminal)
                assert backend.vector_exists(eid) is False, f"vector should be pruned for {terminal}"
            finally:
                backend.close()

    def test_vector_preserved_on_non_status_update(self) -> None:
        backend = self._backend()
        try:
            backend.store(MemoryEntry(id="v-2", content="keep vector"))
            backend.upsert_vector("v-2", [0.3] * backend._dim)

            backend.update("v-2", importance=0.9)

            assert backend.vector_exists("v-2") is True
        finally:
            backend.close()

    def test_vector_preserved_when_status_stays_active(self) -> None:
        backend = self._backend()
        try:
            backend.store(MemoryEntry(id="v-3", content="still active"))
            backend.upsert_vector("v-3", [0.4] * backend._dim)

            backend.update("v-3", status="active")

            assert backend.vector_exists("v-3") is True
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# F-008 — batched recall access (single commit, correct increments)
# ---------------------------------------------------------------------------


class TestF008BatchedRecallAccess:
    def test_single_update_statement_for_n_entries(self) -> None:
        """F-008: N recalled entries -> exactly ONE UPDATE statement (one WAL append).

        The old loop issued one SELECT + one UPDATE per entry (2N statements).
        The batched path issues a single ``UPDATE ... WHERE id IN (...)``.
        """
        backend = SQLiteBackend(Path(":memory:"))
        try:
            ids = [f"r-{i}" for i in range(5)]
            for eid in ids:
                backend.store(MemoryEntry(id=eid, content="recall"))

            update_count = {"n": 0}
            select_count = {"n": 0}
            real_conn = backend._conn

            class _CountingConn:
                def execute(self, sql: str, *args: object) -> object:
                    stripped = sql.lstrip().upper()
                    if stripped.startswith("UPDATE MEMORIES"):
                        update_count["n"] += 1
                    elif stripped.startswith("SELECT") and "FROM MEMORIES" in stripped:
                        select_count["n"] += 1
                    return real_conn.execute(sql, *args)

                def __getattr__(self, name: str) -> object:
                    return getattr(real_conn, name)

            backend._conn = _CountingConn()
            try:
                record_recall_access(backend, ids)
            finally:
                backend._conn = real_conn

            # Exactly one UPDATE for the whole batch; no per-entry SELECT loop.
            assert update_count["n"] == 1
            assert select_count["n"] == 0
        finally:
            backend.close()

    def test_increments_are_correct(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            for eid in ["r-a", "r-b"]:
                backend.store(MemoryEntry(id=eid, content="recall"))

            record_recall_access(backend, ["r-a", "r-b"])

            for eid in ["r-a", "r-b"]:
                loaded = backend.get(eid)
                assert loaded is not None
                assert loaded.recall_count == 1
                assert loaded.access_count == 1
                assert loaded.last_accessed_at is not None
        finally:
            backend.close()

    def test_dedup_within_single_call(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            backend.store(MemoryEntry(id="r-dup", content="dedup"))

            record_recall_access(backend, ["r-dup", "r-dup", "r-dup"])

            loaded = backend.get("r-dup")
            assert loaded is not None
            assert loaded.recall_count == 1
            assert loaded.access_count == 1
        finally:
            backend.close()

    def test_repeated_calls_accumulate(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            backend.store(MemoryEntry(id="r-acc", content="accumulate"))

            for _ in range(3):
                record_recall_access(backend, ["r-acc"])

            loaded = backend.get("r-acc")
            assert loaded is not None
            assert loaded.recall_count == 3
            assert loaded.access_count == 3
        finally:
            backend.close()

    def test_empty_ids_is_noop(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            assert backend.increment_recall_access([]) == 0
        finally:
            backend.close()

    def test_returns_rows_updated(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            backend.store(MemoryEntry(id="r-x", content="x"))
            backend.store(MemoryEntry(id="r-y", content="y"))
            # r-missing does not exist — not counted.
            updated = backend.increment_recall_access(["r-x", "r-y", "r-missing"])
            assert updated == 2
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# F-007 — composite recall indexes present after migration
# ---------------------------------------------------------------------------


class TestF007CompositeIndexes:
    def _index_names(self, backend: SQLiteBackend) -> set[str]:
        rows = backend._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'memories'"
        ).fetchall()
        return {str(r[0]) for r in rows}

    def test_indexes_present_in_fresh_db(self) -> None:
        backend = SQLiteBackend(Path(":memory:"))
        try:
            names = self._index_names(backend)
            assert "idx_memories_ns_updated" in names
            assert "idx_memories_ns_importance" in names
        finally:
            backend.close()

    def test_indexes_present_after_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "idx.db"
        backend = SQLiteBackend(db_path)
        try:
            backend.store(MemoryEntry(id="idx-1", content="content"))
        finally:
            backend.close()

        backend2 = SQLiteBackend(db_path)
        try:
            names = self._index_names(backend2)
            assert "idx_memories_ns_updated" in names
            assert "idx_memories_ns_importance" in names
        finally:
            backend2.close()
