"""PRD-CORE-143 active knowledge-graph wiring tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from trw_memory._client_recall_graph import filter_conflicting_results
from trw_memory.graph import _upsert_edge, update_entry_graph
from trw_memory.models.memory import Anchor, MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _result(memory_id: str, importance: float) -> dict[str, Any]:
    return {"memory_id": memory_id, "importance": importance, "score": importance}


def test_recall_conflict_filter_suppresses_lower_importance_result(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "memory.db")
    try:
        backend.store(MemoryEntry(id="M-high", content="current", importance=0.9))
        backend.store(MemoryEntry(id="M-low", content="obsolete", importance=0.4))
        _upsert_edge(
            backend._conn, "M-high", "M-low", "conflicts_with", 1.0, "2026-07-12T00:00:00+00:00", namespace="default"
        )
        backend._conn.commit()

        client = cast("Any", type("Client", (), {"_backend": backend})())
        filtered = filter_conflicting_results(
            client,
            cast("Any", [_result("M-low", 0.4), _result("M-high", 0.9)]),
        )

        assert [result["memory_id"] for result in filtered] == ["M-high"]
    finally:
        backend.close()


def test_store_graph_update_creates_co_anchored_edge(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "memory.db")
    shared_anchors = [
        Anchor(file="src/auth.py", symbol_name="authenticate"),
        Anchor(file="src/session.py", symbol_name="create_session"),
        Anchor(file="src/user.py", symbol_name="User"),
    ]
    try:
        backend.store(MemoryEntry(id="M-existing", content="first", anchors=shared_anchors))
        fresh = MemoryEntry(id="M-fresh", content="second", anchors=shared_anchors)
        backend.store(fresh)

        result = update_entry_graph(fresh, backend)

        assert result["co_anchored_edges"] == 1
        row = backend._conn.execute(
            "SELECT source_id, target_id, edge_type FROM memory_graph_edges WHERE edge_type = 'co_anchored'"
        ).fetchone()
        assert tuple(row) == ("M-fresh", "M-existing", "co_anchored")
    finally:
        backend.close()


def test_store_graph_update_requires_three_shared_anchors(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "memory.db")
    shared = Anchor(file="src/auth.py", symbol_name="authenticate")
    try:
        backend.store(MemoryEntry(id="M-existing", content="first", anchors=[shared]))
        fresh = MemoryEntry(id="M-fresh", content="second", anchors=[shared])
        backend.store(fresh)

        result = update_entry_graph(fresh, backend)

        assert result["co_anchored_edges"] == 0
    finally:
        backend.close()


def test_graph_wiring_degrades_without_sqlite_connection() -> None:
    client = cast("Any", type("Client", (), {"_backend": object()})())
    original = cast("Any", [_result("M-one", 0.5), _result("M-two", 0.5)])

    assert filter_conflicting_results(client, original) is original
