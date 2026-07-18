"""Tests for Graph-Augmented Recall (GAR) — frontier-001.

NOTE: the knowledge-graph edge-insertion primitive is ``_upsert_edge``
(``trw_memory.graph._upsert_edge``), not a hypothetical
``update_entry_graph(conn, src, dst, ...)`` helper. ``update_entry_graph``
is a *whole-backend enrichment* pass with a different signature
(``update_entry_graph(entry, backend, *, embedding, config)``).  These
tests insert edges directly with ``_upsert_edge`` using a VALID edge
type from ``VALID_EDGE_TYPES`` (e.g. ``related_to`` — ``related`` is
not a valid type and ``_upsert_edge`` rejects it).
"""

from __future__ import annotations

from datetime import datetime, timezone

from trw_memory.graph import _upsert_edge
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result(memory_id: str, content: str, score: float, source: str = "bm25") -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "content": content,
        "score": score,
        "source": source,
        "detail": "",
        "tags": [],
        "importance": 0.5,
        "created_at": "",
        "updated_at": "",
        "namespace": "default",
    }


class TestGraphExpandResults:
    """Unit tests for graph_expand_results()."""

    def test_expands_with_graph_neighbours(self, tmp_path):
        """Graph neighbours not in initial results are added."""
        from trw_memory._client_recall_graph import graph_expand_results

        backend = SQLiteBackend(tmp_path / "test.db")
        backend.store(MemoryEntry(id="M-A", content="alpha topic"))
        backend.store(MemoryEntry(id="M-B", content="related to alpha"))
        # Add a graph edge A -> B (weight 0.8) using a valid edge type.
        _upsert_edge(backend._conn, "M-A", "M-B", "related_to", 0.8, _now())

        class FakeClient:
            _backend = backend
            _namespace = "default"

        initial = [_result("M-A", "alpha topic", 1.0)]
        result = graph_expand_results(FakeClient(), initial, depth=1)  # type: ignore[arg-type]

        ids = [r["memory_id"] for r in result]
        assert "M-B" in ids, "Graph neighbour M-B should be added"
        b = next(r for r in result if r["memory_id"] == "M-B")
        assert b["source"] == "graph"
        assert float(b["score"]) < 1.0  # discounted (1.0 * 0.5 * 0.8 = 0.4)

    def test_no_sqlite_backend_returns_unchanged(self, tmp_path):
        """Falls back gracefully when backend has no SQLite connection."""
        from trw_memory._client_recall_graph import graph_expand_results

        class FakeBackend:
            pass  # no _conn attribute

        class FakeClient:
            _backend = FakeBackend()
            _namespace = "default"

        initial = [_result("M-X", "x", 1.0)]
        result = graph_expand_results(FakeClient(), initial)  # type: ignore[arg-type]
        assert result == initial  # unchanged

    def test_none_backend_returns_unchanged(self, tmp_path):
        """Falls back gracefully when backend is None."""
        from trw_memory._client_recall_graph import graph_expand_results

        class FakeClient:
            _backend = None
            _namespace = "default"

        initial = [_result("M-X", "x", 1.0)]
        result = graph_expand_results(FakeClient(), initial)  # type: ignore[arg-type]
        assert result == initial  # unchanged

    def test_empty_results_returns_empty(self, tmp_path):
        """Empty input returns empty output."""
        from trw_memory._client_recall_graph import graph_expand_results

        class FakeClient:
            _backend = None
            _namespace = "default"

        result = graph_expand_results(FakeClient(), [])  # type: ignore[arg-type]
        assert result == []

    def test_no_graph_edges_returns_unchanged(self, tmp_path):
        """No graph edges -> result unchanged."""
        from trw_memory._client_recall_graph import graph_expand_results

        backend = SQLiteBackend(tmp_path / "test.db")
        backend.store(MemoryEntry(id="M-isolated", content="isolated"))

        class FakeClient:
            _backend = backend
            _namespace = "default"

        initial = [_result("M-isolated", "isolated", 0.9)]
        result = graph_expand_results(FakeClient(), initial, depth=1)  # type: ignore[arg-type]
        assert result == initial  # no neighbours -> unchanged

    def test_deduplicates_existing_results(self, tmp_path):
        """Graph neighbours already in results are not duplicated."""
        from trw_memory._client_recall_graph import graph_expand_results

        backend = SQLiteBackend(tmp_path / "test.db")
        backend.store(MemoryEntry(id="M-A2", content="a2"))
        backend.store(MemoryEntry(id="M-B2", content="b2"))
        _upsert_edge(backend._conn, "M-A2", "M-B2", "related_to", 1.0, _now())

        class FakeClient:
            _backend = backend
            _namespace = "default"

        # Both A and B already in results
        initial = [
            _result("M-A2", "a2", 1.0),
            _result("M-B2", "b2", 0.8),
        ]
        result = graph_expand_results(FakeClient(), initial, depth=1)  # type: ignore[arg-type]
        # Should be exactly 2, no duplicates
        assert len(result) == 2
        assert [r["memory_id"] for r in result] == ["M-A2", "M-B2"]

    def test_skips_non_active_neighbours(self, tmp_path):
        """Archived/obsolete graph neighbours are excluded."""
        from trw_memory._client_recall_graph import graph_expand_results

        backend = SQLiteBackend(tmp_path / "test.db")
        backend.store(MemoryEntry(id="M-src", content="source"))
        backend.store(MemoryEntry(id="M-archived", content="archived", status=MemoryStatus.ARCHIVED))
        _upsert_edge(backend._conn, "M-src", "M-archived", "related_to", 1.0, _now())

        class FakeClient:
            _backend = backend
            _namespace = "default"

        initial = [_result("M-src", "source", 1.0)]
        result = graph_expand_results(FakeClient(), initial, depth=1)  # type: ignore[arg-type]
        ids = [r["memory_id"] for r in result]
        assert "M-archived" not in ids  # excluded

    def test_score_uses_max_score_discount_and_weight(self, tmp_path):
        """Neighbour score = max(result scores) * 0.5 * edge_weight."""
        from trw_memory._client_recall_graph import graph_expand_results

        backend = SQLiteBackend(tmp_path / "test.db")
        backend.store(MemoryEntry(id="M-r1", content="root one"))
        backend.store(MemoryEntry(id="M-r2", content="root two"))
        backend.store(MemoryEntry(id="M-n", content="neighbour"))
        _upsert_edge(backend._conn, "M-r2", "M-n", "related_to", 0.5, _now())

        class FakeClient:
            _backend = backend
            _namespace = "default"

        # max score across roots is 0.8
        initial = [
            _result("M-r1", "root one", 0.4),
            _result("M-r2", "root two", 0.8),
        ]
        result = graph_expand_results(FakeClient(), initial, depth=1)  # type: ignore[arg-type]
        neighbour = next(r for r in result if r["memory_id"] == "M-n")
        # 0.8 * 0.5 (discount) * 0.5 (edge weight) = 0.2
        assert abs(float(neighbour["score"]) - 0.2) < 1e-9


class TestMemoryClientRecallGraphExpansion:
    """Integration tests for MemoryClient.recall(include_graph_expansion=True)."""

    async def test_recall_with_graph_expansion_wires_through(self, memory_client):
        """MemoryClient.recall(include_graph_expansion=True) does not error."""
        # Store two entries with no graph edges; expansion is a no-op
        await memory_client.store("primary topic about databases", tags=["db"])
        result = await memory_client.recall("databases", include_graph_expansion=True)
        assert isinstance(result, list)

    async def test_recall_without_graph_expansion_unchanged(self, memory_client):
        """Default recall (include_graph_expansion=False) is unchanged."""
        await memory_client.store("no graph expansion here", tags=["test"])
        result = await memory_client.recall("no graph expansion here")
        assert isinstance(result, list)

    async def test_recall_graph_expansion_surfaces_neighbour(self, memory_client):
        """An entry reachable only via a graph edge surfaces when expansion is on."""
        await memory_client.store("zzqqxx unique anchor token", tags=["anchor"])
        await memory_client.store("totally different vocabulary here", tags=["other"])

        backend = memory_client._backend
        ids = [e.id for e in backend.list_entries(namespace=memory_client._namespace)]
        anchor_id = next(
            e.id for e in backend.list_entries(namespace=memory_client._namespace) if "zzqqxx" in e.content
        )
        other_id = next(i for i in ids if i != anchor_id)
        _upsert_edge(backend._conn, anchor_id, other_id, "related_to", 1.0, _now())

        with_expansion = await memory_client.recall("zzqqxx", include_graph_expansion=True)
        ids_with = {r["memory_id"] for r in with_expansion}
        assert other_id in ids_with, "graph neighbour should surface with expansion on"
