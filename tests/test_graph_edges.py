"""Graph edge creation and similarity helper tests."""

from __future__ import annotations

from trw_memory.graph import (
    _safe_cosine_similarity,
    create_consolidation_edges,
    create_similarity_edges,
    create_tag_cooccurrence_edges,
)

from ._test_graph_support import (
    _V1,
    _V2,
    _V3,
    _count_edges,
    _insert_memory_row,
    _make_conn,
    _make_entry,
)


class TestCreateSimilarityEdges:
    def test_creates_bidirectional_edges_above_threshold(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        candidates = [("e2", _V2)]

        count = create_similarity_edges(entry, conn, embedding=_V1, candidate_embeddings=candidates)

        assert count == 2
        edges = conn.execute("SELECT source_id, target_id, edge_type FROM memory_graph_edges").fetchall()
        assert len(edges) == 2
        sources = {(edge[0], edge[1]) for edge in edges}
        assert ("e1", "e2") in sources
        assert ("e2", "e1") in sources

    def test_no_edges_at_or_below_threshold(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        candidates = [("e2", _V3)]

        count = create_similarity_edges(entry, conn, embedding=_V1, candidate_embeddings=candidates)

        assert count == 0
        assert _count_edges(conn) == 0

    def test_no_op_when_embedding_is_none(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        candidates = [("e2", _V2)]

        count = create_similarity_edges(entry, conn, embedding=None, candidate_embeddings=candidates)
        assert count == 0
        assert _count_edges(conn) == 0

    def test_no_op_when_candidates_is_none(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")

        count = create_similarity_edges(entry, conn, embedding=_V1, candidate_embeddings=None)
        assert count == 0

    def test_skips_self_reference(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        candidates = [("e1", _V1)]

        count = create_similarity_edges(entry, conn, embedding=_V1, candidate_embeddings=candidates)
        assert count == 0


class TestCreateTagCooccurrenceEdges:
    def test_correct_jaccard_weight(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1", tags=["a", "b", "c"])
        candidates = [_make_entry("e2", tags=["b", "c", "d"])]

        count = create_tag_cooccurrence_edges(entry, conn, candidate_entries=candidates)

        assert count == 2
        row = conn.execute(
            "SELECT weight FROM memory_graph_edges WHERE source_id = 'e1' AND target_id = 'e2'"
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 0.5) < 0.001

    def test_updates_existing_edge_weight(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1", tags=["a", "b"])
        candidates = [_make_entry("e2", tags=["a", "b"])]
        create_tag_cooccurrence_edges(entry, conn, candidate_entries=candidates)

        entry2 = _make_entry("e1", tags=["a", "b", "c"])
        candidates2 = [_make_entry("e2", tags=["a", "b", "d"])]
        create_tag_cooccurrence_edges(entry2, conn, candidate_entries=candidates2)

        edges = conn.execute(
            "SELECT weight FROM memory_graph_edges WHERE source_id = 'e1' AND target_id = 'e2' AND edge_type = 'tag_cooccurrence'"
        ).fetchall()
        assert len(edges) == 1
        assert abs(edges[0][0] - 0.5) < 0.001

    def test_no_edge_for_less_than_2_shared_tags(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1", tags=["a", "b"])
        candidates = [_make_entry("e2", tags=["a", "c"])]

        count = create_tag_cooccurrence_edges(entry, conn, candidate_entries=candidates)
        assert count == 0

    def test_no_edge_for_empty_tags(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1", tags=[])
        candidates = [_make_entry("e2", tags=["a", "b"])]

        count = create_tag_cooccurrence_edges(entry, conn, candidate_entries=candidates)
        assert count == 0

    def test_no_edge_when_candidate_has_empty_tags(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1", tags=["a", "b"])
        candidates = [_make_entry("e2", tags=[])]

        count = create_tag_cooccurrence_edges(entry, conn, candidate_entries=candidates)
        assert count == 0


class TestCreateConsolidationEdges:
    def test_creates_edges_for_valid_source_ids(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "src1")
        _insert_memory_row(conn, "src2")

        entry = _make_entry("consolidated-1", consolidated_from=["src1", "src2"])
        count = create_consolidation_edges(entry, conn)

        assert count == 2
        edges = conn.execute(
            "SELECT target_id, edge_type, weight FROM memory_graph_edges WHERE source_id = 'consolidated-1'"
        ).fetchall()
        assert len(edges) == 2
        for edge in edges:
            assert edge[1] == "consolidation"
            assert edge[2] == 1.0

    def test_skips_missing_source_ids(self) -> None:
        conn = _make_conn()
        _insert_memory_row(conn, "src1")

        entry = _make_entry("consolidated-1", consolidated_from=["src1", "src2"])
        count = create_consolidation_edges(entry, conn)

        assert count == 1

    def test_no_op_when_no_consolidated_from(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1", consolidated_from=[])

        count = create_consolidation_edges(entry, conn)
        assert count == 0


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert abs(_safe_cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 0.001

    def test_orthogonal_vectors(self) -> None:
        assert abs(_safe_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 0.001

    def test_empty_vectors(self) -> None:
        assert _safe_cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self) -> None:
        assert _safe_cosine_similarity([1.0], [1.0, 0.0]) == 0.0

    def test_zero_vector(self) -> None:
        assert _safe_cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestCreateSimilarityEdgesEdgeCases:
    def test_create_similarity_edges_skip_self(self) -> None:
        conn = _make_conn()
        entry = _make_entry("self-ref")
        candidates = [("self-ref", _V1)]

        count = create_similarity_edges(entry, conn, embedding=_V1, candidate_embeddings=candidates)
        assert count == 0
        assert _count_edges(conn) == 0


class TestSafeCosineSimilarityEdgeCases:
    def test_safe_cosine_similarity_dimension_mismatch(self) -> None:
        result = _safe_cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
        assert result == 0.0

    def test_safe_cosine_similarity_zero_vectors(self) -> None:
        result = _safe_cosine_similarity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert result == 0.0
