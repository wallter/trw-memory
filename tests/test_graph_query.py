"""Graph traversal and performance tests."""

from __future__ import annotations

import time

from trw_memory.graph import graph_query

from ._test_graph_support import _insert_edge, _insert_memory_row, _make_conn


class TestGraphQuery:
    def test_depth_1_returns_direct_neighbors_only(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "B", "C", "similarity", 0.8)

        results = graph_query(conn, ["A"], depth=1)

        ids = {result["id"] for result in results}
        assert ids == {"B"}
        assert all(result["depth"] == 1 for result in results)

    def test_depth_2_returns_neighbors_of_neighbors(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "B", "C", "similarity", 0.8)

        results = graph_query(conn, ["A"], depth=2)

        ids = {result["id"] for result in results}
        assert ids == {"B", "C"}
        depths = {result["id"]: result["depth"] for result in results}
        assert depths["B"] == 1
        assert depths["C"] == 2

    def test_excludes_root_nodes(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "B", "A", "similarity", 0.9)

        results = graph_query(conn, ["A"], depth=2)

        ids = {result["id"] for result in results}
        assert "A" not in ids
        assert "B" in ids

    def test_deduplicates_multipath_results(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "A", "C", "similarity", 0.8)
        _insert_edge(conn, "B", "D", "similarity", 0.7)
        _insert_edge(conn, "C", "D", "similarity", 0.7)

        results = graph_query(conn, ["A"], depth=2)

        id_list = [result["id"] for result in results]
        assert id_list.count("D") == 1

    def test_edge_types_filter(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "A", "C", "tag_cooccurrence", 0.5)

        results = graph_query(conn, ["A"], depth=1, edge_types=["similarity"])

        ids = {result["id"] for result in results}
        assert ids == {"B"}

    def test_clamps_depth_above_max(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "B", "C", "similarity", 0.8)
        _insert_edge(conn, "C", "D", "similarity", 0.7)
        _insert_edge(conn, "D", "E", "similarity", 0.6)

        results = graph_query(conn, ["A"], depth=10)

        ids = {result["id"] for result in results}
        assert "B" in ids
        assert "C" in ids
        assert "D" in ids
        assert "E" not in ids

    def test_empty_root_ids_returns_empty(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)

        results = graph_query(conn, [], depth=2)
        assert results == []

    def test_isolated_node_returns_empty(self) -> None:
        conn = _make_conn()

        results = graph_query(conn, ["X"], depth=2)
        assert results == []


class TestGraphQueryNamespaceIsolation:
    def test_namespace_filter_excludes_foreign_namespace_target(self) -> None:
        conn = _make_conn()
        # A and B(ns A) belong to namespace A; X belongs to namespace B.
        _insert_memory_row(conn, "A", namespace="project:a")
        _insert_memory_row(conn, "B", namespace="project:a")
        _insert_memory_row(conn, "X", namespace="project:b")
        # A -> B (same ns) and A -> X (cross-namespace leak edge)
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "A", "X", "similarity", 0.9)

        results = graph_query(conn, ["A"], depth=2, namespace="project:a")

        ids = {result["id"] for result in results}
        assert "B" in ids
        # The ns-B node must NOT be returned for a ns-A BFS.
        assert "X" not in ids

    def test_namespace_filter_blocks_recursion_through_foreign_node(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "A", namespace="project:a")
        _insert_memory_row(conn, "X", namespace="project:b")
        _insert_memory_row(conn, "Y", namespace="project:b")
        # A -> X (foreign) -> Y (foreign). Filtering X out must also prevent
        # the BFS from recursing into Y via X.
        _insert_edge(conn, "A", "X", "similarity", 0.9)
        _insert_edge(conn, "X", "Y", "similarity", 0.9)

        results = graph_query(conn, ["A"], depth=2, namespace="project:a")

        ids = {result["id"] for result in results}
        assert ids == set()

    def test_namespace_none_preserves_legacy_unscoped_behavior(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "A", namespace="project:a")
        _insert_memory_row(conn, "X", namespace="project:b")
        _insert_edge(conn, "A", "X", "similarity", 0.9)

        # Without a namespace argument the legacy cross-namespace behaviour
        # is retained (no JOIN to memories).
        results = graph_query(conn, ["A"], depth=1)

        ids = {result["id"] for result in results}
        assert ids == {"X"}


class TestGraphQueryEdgeCases:
    def test_graph_query_circular_reference(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "B", "C", "similarity", 0.8)
        _insert_edge(conn, "C", "A", "similarity", 0.7)

        results = graph_query(conn, ["A"], depth=3)
        ids = {result["id"] for result in results}
        assert "B" in ids
        assert "C" in ids
        assert "A" not in ids
        assert len(results) == 2

    def test_graph_query_empty_root_ids(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        results = graph_query(conn, [], depth=2)
        assert results == []

    def test_graph_query_depth_clamping(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "B", "C", "similarity", 0.8)
        _insert_edge(conn, "C", "D", "similarity", 0.7)
        _insert_edge(conn, "D", "E", "similarity", 0.6)

        results = graph_query(conn, ["A"], depth=10)
        ids = {result["id"] for result in results}
        assert "B" in ids
        assert "C" in ids
        assert "D" in ids
        assert "E" not in ids

    def test_graph_query_disconnected_node(self) -> None:
        conn = _make_conn()
        results = graph_query(conn, ["lonely"], depth=2)
        assert results == []


class TestGraphQueryPerformance:
    def test_graph_query_p95_under_100ms_for_1000_nodes_and_5000_edges(self) -> None:
        conn = _make_conn()
        for node in range(1000):
            for edge_idx in range(5):
                target = (node + edge_idx + 1) % 1000
                _insert_edge(conn, f"n{node}", f"n{target}", "similarity", 0.9)

        timings_ms: list[float] = []
        for _ in range(20):
            started = time.perf_counter()
            results = graph_query(conn, ["n0"], depth=3)
            timings_ms.append((time.perf_counter() - started) * 1000)
            assert results

        p95_index = max(int(len(timings_ms) * 0.95) - 1, 0)
        p95_ms = sorted(timings_ms)[p95_index]
        assert p95_ms < 100
