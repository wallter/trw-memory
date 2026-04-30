"""Conflict resolution tests for typed graph edges."""

from __future__ import annotations

from trw_memory.graph import filter_conflicts, get_conflicts

from ._test_graph_typed_edges_support import _insert_edge, _insert_memory_row, _make_conn


class TestConflictResolution:
    def test_get_conflicts(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "e1")
        _insert_memory_row(conn, "e2")
        _insert_edge(conn, "e1", "e2", "conflicts_with", 1.0, metadata={"reason": "contradicting"})

        conflicts = get_conflicts(conn, "e1")
        assert len(conflicts) == 1
        assert conflicts[0]["source_id"] == "e1"
        assert conflicts[0]["target_id"] == "e2"

    def test_get_conflicts_bidirectional(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "e1")
        _insert_memory_row(conn, "e2")
        _insert_edge(conn, "e2", "e1", "conflicts_with", 1.0)

        conflicts = get_conflicts(conn, "e1")
        assert len(conflicts) == 1
        assert conflicts[0]["source_id"] == "e2"
        assert conflicts[0]["target_id"] == "e1"

    def test_get_conflicts_empty(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "e1")

        assert get_conflicts(conn, "e1") == []

    def test_filter_conflicts_keeps_higher(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "e1", importance=0.9)
        _insert_memory_row(conn, "e2", importance=0.3)
        _insert_edge(conn, "e1", "e2", "conflicts_with", 1.0)

        result = filter_conflicts(
            [
                {"id": "e1", "importance": 0.9, "content": "high"},
                {"id": "e2", "importance": 0.3, "content": "low"},
            ],
            conn,
        )
        ids = [entry["id"] for entry in result]
        assert "e1" in ids
        assert "e2" not in ids

    def test_filter_conflicts_no_conflicts(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "e1")
        _insert_memory_row(conn, "e2")

        result = filter_conflicts(
            [
                {"id": "e1", "importance": 0.5, "content": "a"},
                {"id": "e2", "importance": 0.5, "content": "b"},
            ],
            conn,
        )
        assert len(result) == 2

    def test_filter_conflicts_equal_importance(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "e1", importance=0.5)
        _insert_memory_row(conn, "e2", importance=0.5)
        _insert_edge(conn, "e1", "e2", "conflicts_with", 1.0)

        result = filter_conflicts(
            [
                {"id": "e1", "importance": 0.5, "content": "a"},
                {"id": "e2", "importance": 0.5, "content": "b"},
            ],
            conn,
        )
        assert len(result) == 2
