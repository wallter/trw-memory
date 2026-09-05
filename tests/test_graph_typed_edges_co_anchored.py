"""Co-anchored typed edge creation tests."""

from __future__ import annotations

import json

from trw_memory.graph import create_co_anchored_edges

from ._test_graph_typed_edges_support import _count_edges, _get_edge_metadata, _insert_memory_row, _make_conn


class TestCoAnchoredEdges:
    def test_co_anchored_edge_creation(self) -> None:
        conn = _make_conn()
        anchor_data = [{"file": "src/graph.py", "symbol_name": "func_a", "symbol_type": "function"}]
        _insert_memory_row(conn, "e1", anchors_json=json.dumps(anchor_data))
        _insert_memory_row(conn, "e2", anchors_json=json.dumps(anchor_data))

        count = create_co_anchored_edges(conn, "e1", ["src/graph.py"], namespace="default")
        assert count >= 1
        assert _count_edges(conn, "co_anchored") >= 1
        assert _get_edge_metadata(conn, "e1", "e2", "co_anchored")["anchor_file"] == "src/graph.py"

    def test_co_anchored_no_match(self) -> None:
        conn = _make_conn()
        anchor_data = [{"file": "src/unique.py", "symbol_name": "func_a", "symbol_type": "function"}]
        _insert_memory_row(conn, "e1", anchors_json=json.dumps(anchor_data))

        assert create_co_anchored_edges(conn, "e1", ["src/unique.py"], namespace="default") == 0

    def test_co_anchored_cap_per_file(self) -> None:
        conn = _make_conn()
        anchor_data = [{"file": "src/big.py", "symbol_name": "func", "symbol_type": "function"}]

        for i in range(10):
            _insert_memory_row(conn, f"e{i}", anchors_json=json.dumps(anchor_data))

        assert create_co_anchored_edges(conn, "e0", ["src/big.py"], max_per_file=3, namespace="default") <= 3

    def test_co_anchored_multiple_files(self) -> None:
        conn = _make_conn()
        anchors_e1 = [
            {"file": "src/a.py", "symbol_name": "f", "symbol_type": "function"},
            {"file": "src/b.py", "symbol_name": "g", "symbol_type": "function"},
        ]
        anchor_a = [{"file": "src/a.py", "symbol_name": "f", "symbol_type": "function"}]
        anchor_b = [{"file": "src/b.py", "symbol_name": "g", "symbol_type": "function"}]
        _insert_memory_row(conn, "e1", anchors_json=json.dumps(anchors_e1))
        _insert_memory_row(conn, "e2", anchors_json=json.dumps(anchor_a))
        _insert_memory_row(conn, "e3", anchors_json=json.dumps(anchor_b))

        assert create_co_anchored_edges(conn, "e1", ["src/a.py", "src/b.py"], namespace="default") == 2

    def test_co_anchored_skips_self(self) -> None:
        conn = _make_conn()
        anchor_data = [{"file": "src/x.py", "symbol_name": "f", "symbol_type": "function"}]
        _insert_memory_row(conn, "e1", anchors_json=json.dumps(anchor_data))

        assert create_co_anchored_edges(conn, "e1", ["src/x.py"], namespace="default") == 0
