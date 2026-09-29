"""Graph decay pass tests."""

from __future__ import annotations

import threading
import tracemalloc
from datetime import datetime, timezone

import pytest

from trw_memory.graph import memory_decay_pass

from ._test_graph_support import _insert_memory_row, _make_conn


class TestMemoryDecayPass:
    def test_processes_qualifying_entries(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        _insert_memory_row(conn, "e1", cross_validated=1, last_accessed_at=old_date, importance=0.8)
        _insert_memory_row(conn, "e2", cross_validated=1, last_accessed_at=old_date, importance=0.6)

        result = memory_decay_pass(conn, cutoff_days=90)

        assert result["processed"] == 2
        assert result["total_decayed"] == 2

        row = conn.execute("SELECT importance FROM memories WHERE id = 'e1'").fetchone()
        assert row is not None
        assert abs(row[0] - 0.7) < 0.001

        history_row = conn.execute("SELECT outcome_history FROM memories WHERE id = 'e1'").fetchone()
        assert history_row is not None
        assert "new_value=0.7000" in str(history_row[0])

    def test_decays_non_cross_validated_stale_unused_entries(self) -> None:
        """PRD-CORE-244 FR09: ``cross_validated`` is not part of the decay
        predicate. It was re-measured at 0 of 9,366 rows on 2026-09-03 — a
        gate no production entry could ever satisfy — so a stale, unused
        entry decays regardless of cross-validation status.
        """
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        _insert_memory_row(conn, "e1", cross_validated=0, last_accessed_at=old_date, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90)

        assert result["processed"] == 1
        assert result["total_decayed"] == 1
        row = conn.execute("SELECT importance FROM memories WHERE id = 'e1'").fetchone()
        assert row is not None
        assert abs(row[0] - 0.7) < 0.001

    def test_skips_recently_accessed_non_cross_validated_entries(self) -> None:
        """Negative control: recency, not cross-validation, gates decay — a
        recently used entry is skipped even when never cross-validated.
        """
        conn = _make_conn()
        recent = datetime.now(timezone.utc).isoformat()
        _insert_memory_row(conn, "e1", cross_validated=0, last_accessed_at=recent, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90)

        assert result["processed"] == 0
        row = conn.execute("SELECT importance FROM memories WHERE id = 'e1'").fetchone()
        assert row is not None
        assert abs(row[0] - 0.8) < 0.001

    def test_respects_batch_size_limit(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        for idx in range(10):
            _insert_memory_row(conn, f"e{idx}", cross_validated=1, last_accessed_at=old_date, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90, batch_size=3)

        assert result["processed"] == 3
        assert result["more"] is True

    def test_clamps_batch_size_to_1000(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        insert_sql = (
            "INSERT INTO memories ("
            "id, content, created_at, updated_at, last_accessed_at, cross_validated, importance"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        conn.executemany(
            insert_sql,
            [(f"e{idx}", "content", old_date, old_date, old_date, 1, 0.8) for idx in range(1_500)],
        )
        conn.commit()

        result = memory_decay_pass(conn, cutoff_days=90, batch_size=2_000)

        assert result["processed"] == 1_000
        assert result["more"] is True

    def test_rejects_non_positive_batch_size(self) -> None:
        conn = _make_conn()

        with pytest.raises(ValueError, match="batch_size must be positive"):
            memory_decay_pass(conn, cutoff_days=90, batch_size=0)

    def test_skips_recently_accessed_entries(self) -> None:
        conn = _make_conn()
        recent = datetime.now(timezone.utc).isoformat()
        _insert_memory_row(conn, "e1", cross_validated=1, last_accessed_at=recent, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90)

        assert result["processed"] == 0

    def test_skips_never_accessed_fresh_entries(self) -> None:
        conn = _make_conn()
        recent = datetime.now(timezone.utc).isoformat()
        _insert_memory_row(conn, "e1", cross_validated=1, created_at=recent, last_accessed_at=None, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90)

        assert result["processed"] == 0

    def test_decays_never_accessed_old_entries_by_created_at(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        _insert_memory_row(conn, "e1", cross_validated=1, created_at=old_date, last_accessed_at=None, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90)

        assert result["processed"] == 1
        row = conn.execute("SELECT importance FROM memories WHERE id = 'e1'").fetchone()
        assert row is not None
        assert abs(row[0] - 0.7) < 0.001

    def test_decay_floors_at_zero(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        _insert_memory_row(conn, "e1", cross_validated=1, last_accessed_at=old_date, importance=0.05)

        memory_decay_pass(conn, cutoff_days=90)

        row = conn.execute("SELECT importance FROM memories WHERE id = 'e1'").fetchone()
        assert row is not None
        assert row[0] == 0.0


class TestMemoryDecayPassBatch:
    def test_memory_decay_pass_batch(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"

        for idx in range(5):
            _insert_memory_row(
                conn,
                f"decay-{idx}",
                cross_validated=1,
                last_accessed_at=old_date,
                importance=0.8,
            )

        result = memory_decay_pass(conn, cutoff_days=90, batch_size=3)

        assert result["processed"] == 3
        assert result["more"] is True
        assert result["total_decayed"] == 3

        decayed_count = 0
        for idx in range(5):
            row = conn.execute("SELECT importance FROM memories WHERE id = ?", (f"decay-{idx}",)).fetchone()
            if row and abs(row[0] - 0.7) < 0.001:
                decayed_count += 1
        assert decayed_count == 3

    def test_memory_decay_pass_peak_memory_under_512mb_for_50000_entries(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"

        insert_sql = (
            "INSERT INTO memories ("
            "id, content, created_at, updated_at, last_accessed_at, cross_validated, importance"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        for start in range(0, 50_000, 5_000):
            rows = [
                (f"decay-{idx}", "content", old_date, old_date, old_date, 1, 0.8) for idx in range(start, start + 5_000)
            ]
            conn.executemany(insert_sql, rows)
        conn.commit()

        tracemalloc.start()
        try:
            result = memory_decay_pass(conn, cutoff_days=90, batch_size=1000)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert result["processed"] == 1000
        # PRD-CORE-331 FR10: no COUNT(*) is issued (see TestMemoryDecayPassNoCount below); "more"
        # is the whole continuation signal now, true whenever the pass didn't reach the end.
        assert result["more"] is True
        assert result["total_decayed"] == 1000
        assert peak < 512 * 1024 * 1024


class TestMemoryDecayPassCursor:
    """PRD-CORE-307 FR05: a persisted keyset cursor reaches every eligible row before any repeat.

    Fails on 574756d31: the pre-fix batch SELECT has no ``ORDER BY`` and no cursor, so 3 passes of
    a 1,000-row window over 2,500 eligible rows re-select the same top rows every time -- this test
    observed ``decayed_count`` stuck at 1,000 forever instead of reaching 2,500 within 3 passes.
    """

    def test_cursor_reaches_all_rows_before_repeat(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        for idx in range(2_500):
            _insert_memory_row(conn, f"e{idx:05d}", cross_validated=1, last_accessed_at=old_date, importance=0.8)

        decayed_ids: list[str] = []
        cursor: tuple[str, str] | None = None
        for _ in range(3):
            result = memory_decay_pass(conn, cutoff_days=90, batch_size=1_000, cursor=cursor)
            rows = conn.execute("SELECT id FROM memories WHERE importance < 0.8").fetchall()
            decayed_ids = [row[0] for row in rows]
            next_cursor = result["next"]
            cursor = (next_cursor[0], next_cursor[1]) if next_cursor is not None else None

        assert len(decayed_ids) == len(set(decayed_ids)) == 2_500, "every eligible row decayed exactly once"

    def test_cursor_wraps_to_start_when_it_reaches_the_end(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        for idx in range(5):
            _insert_memory_row(conn, f"e{idx}", cross_validated=1, last_accessed_at=old_date, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90, batch_size=1_000)

        assert result["next"] is None

    def test_cursor_advances_past_a_fully_ineligible_window(self) -> None:
        """The cursor advances by rows EXAMINED, not rows qualifying: a window of only fresh rows
        still returns a ``next`` past them, so the following pass does not re-scan them forever."""
        conn = _make_conn()
        recent = datetime.now(timezone.utc).isoformat()
        for idx in range(3):
            _insert_memory_row(conn, f"e{idx}", cross_validated=1, last_accessed_at=recent, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90, batch_size=2)

        assert result["processed"] == 0
        assert result["next"] == ["default", "e1"]


class TestMemoryDecayPassIndexUse:
    """PRD-CORE-307 NFR03: the batch SELECT is a seek over the primary key, not a full scan.

    ``tests/_test_graph_support._make_conn`` builds ``memories`` with only its primary key (no
    ``idx_memories_*`` composite indexes), so this is "a namespace with no other usable index".
    """

    def test_batch_select_reports_index_use(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "e1", cross_validated=1, last_accessed_at="2020-01-01T00:00:00+00:00")

        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        try:
            memory_decay_pass(conn, cutoff_days=90)
        finally:
            conn.set_trace_callback(None)

        select = next(sql for sql in statements if "ORDER BY namespace, id" in sql)
        plan = conn.execute(f"EXPLAIN QUERY PLAN {select}").fetchall()
        plan_text = " ".join(str(row[3]) for row in plan)
        assert "USING INDEX" in plan_text.upper(), plan_text


class TestMemoryDecayPassLockDiscipline:
    """Verify the batch SELECT runs inside the caller-supplied lock scope.

    The fix moves the batch SELECT inside the with lock-or-nullcontext block so it is
    serialised with respect to concurrent backend writes that hold the same lock. The
    COUNT query this class also covered was deleted by PRD-CORE-331 FR10 -- see
    ``TestMemoryDecayPassNoCount`` for its replacement coverage.
    """

    def test_more_is_consistent_with_processed(self) -> None:
        """The batch SELECT executes under the caller's lock so a concurrent write cannot
        interleave with it. PRD-CORE-331 FR10 removed the separate COUNT query this test
        used to also serialise against the lock -- ``more`` is now derived purely from the
        SELECT's own window, so there is nothing left to race.
        """
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        for i in range(5):
            _insert_memory_row(conn, f"e{i}", cross_validated=1, last_accessed_at=old_date, importance=0.6)

        lock = threading.Lock()
        result = memory_decay_pass(conn, cutoff_days=90, batch_size=3, lock=lock)

        assert result["processed"] == 3
        assert result["more"] is True

    def test_lock_passed_does_not_block_single_threaded_caller(self) -> None:
        """Passing a Lock should not cause a deadlock when called from a
        single thread (the lock is not already held by the caller). The
        function must acquire + release it correctly.
        """
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        _insert_memory_row(conn, "e1", cross_validated=1, last_accessed_at=old_date, importance=0.5)

        lock = threading.Lock()
        result = memory_decay_pass(conn, cutoff_days=90, lock=lock)

        assert result["processed"] == 1
        # Lock must be released after the call completes.
        acquired = lock.acquire(blocking=False)
        assert acquired, "Lock was not released after memory_decay_pass returned"
        lock.release()


class TestMemoryDecayPassNoCount:
    """PRD-CORE-331 FR10 (B71-135a): the pass no longer issues an O(namespace) ``COUNT(*)``.

    Fails on 79e84147d (the archive base): that revision's ``memory_decay_pass`` runs
    ``SELECT COUNT(*) FROM (SELECT 1 FROM memories WHERE ... LIMIT ?)`` every call, unconditionally
    -- present even on a store where only one row qualifies, which is exactly the point: the cost
    was never proportional to what one batch needed, only to how many qualifying rows the whole
    namespace held. Quoted from that revision:

        total = conn.execute(
            f"SELECT COUNT(*) FROM (SELECT 1 FROM memories WHERE {count_predicate} LIMIT ?)",
            (cutoff, *scope, DECAY_COUNT_MAX + 1),
        ).fetchone()
    """

    def test_no_count_star_is_issued_when_few_rows_qualify(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        # Only one qualifying row: a COUNT(*) here would be firing for information "more"
        # (derived from the SELECT's own window) makes redundant.
        _insert_memory_row(conn, "e1", cross_validated=1, last_accessed_at=old_date, importance=0.8)

        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        try:
            memory_decay_pass(conn, cutoff_days=90)
        finally:
            conn.set_trace_callback(None)

        count_statements = [sql for sql in statements if "COUNT(" in sql.upper()]
        assert count_statements == [], f"expected no COUNT(*) query, got: {count_statements}"


class TestMemoryDecayPassMoreFlag:
    """PRD-CORE-331 FR10: the reply carries ``more`` (true exactly when ``next`` is not None) and
    no ``remaining``/``remaining_capped`` key at all."""

    def test_more_is_true_when_the_pass_did_not_reach_the_end(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        for idx in range(5):
            _insert_memory_row(conn, f"e{idx}", cross_validated=1, last_accessed_at=old_date, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90, batch_size=2)

        assert result["more"] is True
        assert result["next"] is not None
        assert "remaining" not in result
        assert "remaining_capped" not in result

    def test_more_is_false_on_the_last_page(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        for idx in range(3):
            _insert_memory_row(conn, f"e{idx}", cross_validated=1, last_accessed_at=old_date, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90, batch_size=1_000)

        assert result["next"] is None
        assert result["more"] is False
        assert "remaining" not in result
        assert "remaining_capped" not in result
