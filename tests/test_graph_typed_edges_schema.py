"""Typed graph edge schema and validation tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trw_memory.graph import VALID_EDGE_TYPES, _upsert_edge

from ._test_graph_typed_edges_support import _count_edges, _get_edge_metadata, _make_conn


class TestValidEdgeTypes:
    def test_valid_edge_types_frozenset(self) -> None:
        expected = {
            "similarity",
            "tag_cooccurrence",
            "consolidation",
            "anchored_to",
            "related_to",
            "same_root_cause",
            "depends_on",
            "produced",
            "motivated_by",
            "co_anchored",
            "supersedes",
            "evidence_for",
            "conflicts_with",
        }

        assert isinstance(VALID_EDGE_TYPES, frozenset)
        assert VALID_EDGE_TYPES == expected
        assert len(VALID_EDGE_TYPES) == 13

    def test_upsert_edge_with_metadata(self) -> None:
        conn = _make_conn()
        now = datetime.now(timezone.utc).isoformat()
        metadata = {"anchor_file": "src/graph.py", "reason": "shared_anchor"}

        _upsert_edge(conn, "e1", "e2", "co_anchored", 0.8, now, metadata=metadata, namespace="default")
        conn.commit()

        parsed = _get_edge_metadata(conn, "e1", "e2", "co_anchored")
        assert parsed["anchor_file"] == "src/graph.py"
        assert parsed["reason"] == "shared_anchor"

    def test_upsert_edge_without_metadata(self) -> None:
        conn = _make_conn()
        now = datetime.now(timezone.utc).isoformat()

        _upsert_edge(conn, "e1", "e2", "similarity", 0.9, now, namespace="default")
        conn.commit()

        assert _get_edge_metadata(conn, "e1", "e2", "similarity") == {}

    def test_upsert_edge_invalid_type_raises(self) -> None:
        conn = _make_conn()
        now = datetime.now(timezone.utc).isoformat()

        with pytest.raises(ValueError, match="Invalid edge type"):
            _upsert_edge(conn, "e1", "e2", "nonexistent_type", 0.5, now, namespace="default")

    def test_upsert_edge_all_valid_types_accepted(self) -> None:
        conn = _make_conn()
        now = datetime.now(timezone.utc).isoformat()

        for i, edge_type in enumerate(sorted(VALID_EDGE_TYPES)):
            _upsert_edge(conn, f"src-{i}", f"tgt-{i}", edge_type, 0.5, now, namespace="default")
        conn.commit()

        assert _count_edges(conn) == 13

    def test_edge_metadata_migration(self) -> None:
        conn = _make_conn()
        cursor = conn.execute("PRAGMA table_info(memory_graph_edges)")
        columns = {row[1] for row in cursor.fetchall()}

        assert "edge_metadata" in columns

    def test_upsert_edge_metadata_size_limit(self) -> None:
        conn = _make_conn()
        now = datetime.now(timezone.utc).isoformat()
        large_metadata = {"key": "x" * 5000}

        with pytest.raises(ValueError, match="exceeds 4096 byte limit"):
            _upsert_edge(conn, "e1", "e2", "related_to", 0.5, now, metadata=large_metadata, namespace="default")

    def test_upsert_edge_metadata_within_limit(self) -> None:
        conn = _make_conn()
        now = datetime.now(timezone.utc).isoformat()
        ok_metadata = {"key": "x" * 100}

        _upsert_edge(conn, "e1", "e2", "related_to", 0.5, now, metadata=ok_metadata, namespace="default")
        conn.commit()

        assert _count_edges(conn) == 1
