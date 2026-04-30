"""Impact propagation tests for typed graph edges."""

from __future__ import annotations

from trw_memory.graph import propagate_impact

from ._test_graph_typed_edges_support import _insert_edge, _insert_memory_row, _make_conn


class TestImpactPropagation:
    def test_propagate_impact_evidence_for(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "neighbor", importance=0.5)
        _insert_edge(conn, "root", "neighbor", "evidence_for", 0.9)

        affected = propagate_impact(conn, "root", 0.20)
        neighbor_deltas = [item for item in affected if item[0] == "neighbor"]
        assert len(neighbor_deltas) == 1
        assert abs(neighbor_deltas[0][1] - 0.06) < 1e-6

    def test_propagate_impact_co_anchored(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "neighbor", importance=0.5)
        _insert_edge(conn, "root", "neighbor", "co_anchored", 0.8)

        affected = propagate_impact(conn, "root", 0.20)
        neighbor_deltas = [item for item in affected if item[0] == "neighbor"]
        assert len(neighbor_deltas) == 1
        assert abs(neighbor_deltas[0][1] - 0.04) < 1e-6

    def test_propagate_impact_max_depth(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "a", importance=0.5)
        _insert_memory_row(conn, "b", importance=0.5)
        _insert_memory_row(conn, "c", importance=0.5)
        _insert_memory_row(conn, "d", importance=0.5)

        _insert_edge(conn, "a", "b", "evidence_for", 0.9)
        _insert_edge(conn, "b", "c", "evidence_for", 0.9)
        _insert_edge(conn, "c", "d", "evidence_for", 0.9)

        affected = propagate_impact(conn, "a", 0.20, max_depth=2)
        affected_ids = {item[0] for item in affected}
        assert "b" in affected_ids
        assert "c" in affected_ids
        assert "d" not in affected_ids

    def test_propagate_impact_max_affected(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)

        for i in range(60):
            nid = f"n{i}"
            _insert_memory_row(conn, nid, importance=0.5)
            _insert_edge(conn, "root", nid, "related_to", 0.7)

        affected = propagate_impact(conn, "root", 0.10, max_affected=50)
        assert len(affected) <= 50

    def test_propagate_impact_negative_delta(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "neighbor", importance=0.5)
        _insert_edge(conn, "root", "neighbor", "evidence_for", 0.9)

        affected = propagate_impact(conn, "root", -0.20)
        neighbor_deltas = [item for item in affected if item[0] == "neighbor"]
        assert len(neighbor_deltas) == 1
        assert abs(neighbor_deltas[0][1] - (-0.03)) < 1e-6

    def test_propagate_impact_records_outcome_history(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "neighbor", importance=0.5)
        _insert_edge(conn, "root", "neighbor", "evidence_for", 0.9)

        propagate_impact(conn, "root", 0.20)

        row = conn.execute("SELECT outcome_history FROM memories WHERE id = ?", ("neighbor",)).fetchone()
        assert row is not None
        assert "impact_propagation" in str(row[0])

    def test_propagate_impact_updates_importance(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "neighbor", importance=0.5)
        _insert_edge(conn, "root", "neighbor", "evidence_for", 0.9)

        propagate_impact(conn, "root", 0.20)

        row = conn.execute("SELECT importance FROM memories WHERE id = ?", ("neighbor",)).fetchone()
        assert row is not None
        assert abs(row[0] - 0.56) < 1e-6

    def test_propagate_impact_clamps_to_bounds(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "neighbor", importance=0.98)
        _insert_edge(conn, "root", "neighbor", "evidence_for", 0.9)

        propagate_impact(conn, "root", 0.50)

        row = conn.execute("SELECT importance FROM memories WHERE id = ?", ("neighbor",)).fetchone()
        assert row is not None
        assert row[0] <= 1.0
