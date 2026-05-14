"""Graph decay pass tests."""

from __future__ import annotations

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

    def test_skips_non_cross_validated_entries(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        _insert_memory_row(conn, "e1", cross_validated=0, last_accessed_at=old_date, importance=0.8)

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
        assert result["remaining"] == 7

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
        assert result["remaining"] == 500

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
        assert result["remaining"] == 2
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
        assert result["remaining"] == 49_000
        assert result["total_decayed"] == 1000
        assert peak < 512 * 1024 * 1024
