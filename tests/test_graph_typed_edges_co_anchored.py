"""Co-anchored typed edge creation tests."""

from __future__ import annotations

import json

from trw_memory.graph import create_co_anchored_edges
from trw_memory.storage._anchor_index import replace_anchor_postings

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

    def test_co_anchored_uses_postings_index_not_anchors_column(self) -> None:
        """A row whose ``anchors`` column is empty but whose postings exist is still found.

        Proves the query is driven by ``anchor_postings`` (PRD-CORE-332), not a
        ``json_each(memories.anchors)`` scan: the column here carries no anchors at all.
        """
        conn = _make_conn()
        _insert_memory_row(conn, "e1", anchors_json="[]")
        _insert_memory_row(conn, "e2", anchors_json="[]")
        replace_anchor_postings(conn, "default", "e2", [{"file": "src/ghost.py"}])
        conn.commit()

        count = create_co_anchored_edges(conn, "e1", ["src/ghost.py"], namespace="default")
        assert count == 1
        assert _get_edge_metadata(conn, "e1", "e2", "co_anchored")["anchor_file"] == "src/ghost.py"

    def test_co_anchored_rejects_absolute_and_dotdot_anchor_files(self) -> None:
        """Absolute or ``..``-bearing anchors have no posting key, so they yield no edge.

        The distill writer only emits repo-relative anchors, so this is a deliberate
        narrowing from routing onto ``anchor_postings``' normalized key, not a regression.
        """
        conn = _make_conn()
        anchor_data = [{"file": "src/real.py", "symbol_name": "f", "symbol_type": "function"}]
        _insert_memory_row(conn, "e1", anchors_json=json.dumps(anchor_data))
        _insert_memory_row(conn, "e2", anchors_json=json.dumps(anchor_data))

        assert create_co_anchored_edges(conn, "e1", ["/abs/src/real.py"], namespace="default") == 0
        assert create_co_anchored_edges(conn, "e1", ["../src/real.py"], namespace="default") == 0
        # Sanity: the normalized, valid form of the same file still matches.
        assert create_co_anchored_edges(conn, "e1", ["src/real.py"], namespace="default") == 1

    def test_co_anchored_refused_by_a_canary_leaves_no_open_write_transaction(self) -> None:
        """A canary-refused upsert writes nothing but still opened a write txn; it must be closed.

        Regression: ``create_co_anchored_edges`` committed only when an edge was
        created, so a pass whose every candidate was a system canary left the
        connection holding the write lock and a second connection blocked on it.
        """
        conn = _make_conn()
        anchor_data = [{"file": "src/canary.py", "symbol_name": "f", "symbol_type": "function"}]
        _insert_memory_row(conn, "e1", anchors_json=json.dumps(anchor_data))
        _insert_memory_row(conn, "e2", anchors_json=json.dumps(anchor_data))
        conn.execute("UPDATE memories SET metadata = ? WHERE id = 'e2'", (json.dumps({"system_canary": "true"}),))
        conn.commit()

        assert create_co_anchored_edges(conn, "e1", ["src/canary.py"], namespace="default") == 0
        assert _count_edges(conn, "co_anchored") == 0
        assert not conn.in_transaction
