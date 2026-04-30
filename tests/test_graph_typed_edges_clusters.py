"""Cluster detection tests for typed graph edges."""

from __future__ import annotations

from trw_memory.graph import detect_clusters

from ._test_graph_typed_edges_support import _insert_edge, _insert_memory_row, _make_conn


class TestClusterDetection:
    def test_detect_clusters_dense(self) -> None:
        conn = _make_conn()
        node_ids = [f"n{i}" for i in range(6)]
        for node_id in node_ids:
            _insert_memory_row(conn, node_id)

        pairs = [(node_ids[i], node_ids[j]) for i in range(6) for j in range(i + 1, 6)]
        for src, tgt in pairs[:12]:
            _insert_edge(conn, src, tgt, "related_to", 0.7)
            _insert_edge(conn, tgt, src, "related_to", 0.7)
        conn.commit()

        clusters = detect_clusters(conn, min_size=5, min_connectivity=0.6)
        assert len(clusters) >= 1
        cluster = clusters[0]
        assert "domain_name" in cluster
        assert "member_ids" in cluster
        assert "connectivity" in cluster
        assert isinstance(cluster["member_ids"], list)
        assert len(cluster["member_ids"]) >= 5
        assert isinstance(cluster["connectivity"], float)
        assert cluster["connectivity"] >= 0.6

    def test_detect_clusters_sparse(self) -> None:
        conn = _make_conn()
        for i in range(4):
            _insert_memory_row(conn, f"s{i}")

        _insert_edge(conn, "s0", "s1", "related_to", 0.5)
        _insert_edge(conn, "s1", "s0", "related_to", 0.5)
        _insert_edge(conn, "s2", "s3", "related_to", 0.5)
        _insert_edge(conn, "s3", "s2", "related_to", 0.5)
        conn.commit()

        assert detect_clusters(conn, min_size=5, min_connectivity=0.6) == []

    def test_detect_clusters_proposes_domain_name(self) -> None:
        conn = _make_conn()
        node_ids = [f"c{i}" for i in range(5)]
        for node_id in node_ids:
            _insert_memory_row(conn, node_id)

        for i in range(5):
            for j in range(i + 1, 5):
                _insert_edge(conn, node_ids[i], node_ids[j], "related_to", 0.8)
                _insert_edge(conn, node_ids[j], node_ids[i], "related_to", 0.8)
        conn.commit()

        clusters = detect_clusters(conn, min_size=5, min_connectivity=0.6)
        assert len(clusters) >= 1
        assert isinstance(clusters[0]["domain_name"], str)
        assert len(clusters[0]["domain_name"]) > 0
