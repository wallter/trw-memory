"""memory_graph_related: a learning's active graph neighbours in its own namespace, bounded (PRD-CORE-143, CORE-280 FR01).

trw-mcp's ``trw_recall`` graph mode reads these over the daemon, so the
traversal's namespace scoping, active-only hydration and breadth bound are
pinned here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.graph import _upsert_edge
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.recall_support import memory_graph_related_impl


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "graph.db")
    yield store
    store.close()


def _edge(
    backend: SQLiteBackend, source: str, target: str, namespace: str = "project:a", edge_type: str = "related_to"
) -> None:
    _upsert_edge(backend._conn, source, target, edge_type, 0.8, "2026-07-12T00:00:00+00:00", namespace=namespace)
    backend._conn.commit()


def _related(backend: SQLiteBackend, limit: int = 50) -> dict[str, object]:
    return memory_graph_related_impl("project:a", "L-root", 1, None, limit, backend=backend)


def test_only_active_neighbours_in_the_roots_namespace_are_returned(backend: SQLiteBackend) -> None:
    backend.store(MemoryEntry(id="L-root", content="root", namespace="project:a"))
    backend.store(MemoryEntry(id="L-active", content="active", namespace="project:a", tags=["graph"]))
    backend.store(MemoryEntry(id="L-obsolete", content="old", namespace="project:a", status=MemoryStatus.OBSOLETE))
    backend.store(MemoryEntry(id="L-foreign", content="foreign", namespace="project:b"))
    for target in ("L-active", "L-obsolete", "L-foreign"):
        _edge(backend, "L-root", target)

    answer = _related(backend)

    assert [(row["id"], row["content"], row["edge_type"], row["depth"]) for row in answer["related"]] == [  # type: ignore[attr-defined]
        ("L-active", "active", "related_to", 1)
    ]
    assert answer["truncated"] is False


def test_a_dense_neighbourhood_is_capped_and_reported_truncated(backend: SQLiteBackend) -> None:
    backend.store(MemoryEntry(id="L-root", content="root", namespace="project:a"))
    for index in range(8):
        backend.store(MemoryEntry(id=f"L-{index}", content=f"L-{index}", namespace="project:a"))
        _edge(backend, "L-root", f"L-{index}")

    answer = _related(backend, limit=3)

    assert len(answer["related"]) == 3  # type: ignore[arg-type]
    assert answer["truncated"] is True


def test_a_target_reached_by_two_edge_types_takes_one_place_in_the_window(backend: SQLiteBackend) -> None:
    for entry_id in ("L-root", "L-twice", "L-second"):
        backend.store(MemoryEntry(id=entry_id, content=entry_id, namespace="project:a"))
    _edge(backend, "L-root", "L-twice")
    _edge(backend, "L-root", "L-twice", edge_type="similarity")
    _edge(backend, "L-root", "L-second")

    answer = _related(backend, limit=1)

    assert [row["id"] for row in answer["related"]] == ["L-twice"]  # type: ignore[attr-defined]
    assert answer["truncated"] is True  # L-second exists beyond the window


def test_another_namespaces_edge_between_the_same_ids_is_not_followed(backend: SQLiteBackend) -> None:
    for namespace in ("project:a", "project:b"):
        backend.store(MemoryEntry(id="L-root", content="root", namespace=namespace))
        backend.store(MemoryEntry(id="L-twin", content="twin", namespace=namespace))
    _edge(backend, "L-root", "L-twin", namespace="project:b")

    assert _related(backend)["related"] == []


def test_obsolete_neighbours_neither_fill_the_window_nor_claim_more(backend: SQLiteBackend) -> None:
    backend.store(MemoryEntry(id="L-root", content="root", namespace="project:a"))
    for index in range(3):
        backend.store(
            MemoryEntry(id=f"L-old{index}", content="old", namespace="project:a", status=MemoryStatus.OBSOLETE)
        )
        _edge(backend, "L-root", f"L-old{index}")
    backend.store(MemoryEntry(id="L-live", content="live", namespace="project:a"))
    _edge(backend, "L-root", "L-live")

    answer = _related(backend, limit=1)

    assert [row["id"] for row in answer["related"]] == ["L-live"]  # type: ignore[attr-defined]
    assert answer["truncated"] is False


@pytest.mark.parametrize(
    "arguments", [{"depth": 0}, {"depth": 4}, {"limit": 0}, {"limit": 1001}, {"edge_types": ["not-real"]}]
)
def test_an_unbounded_or_unknown_traversal_is_refused(arguments: dict[str, object]) -> None:
    from trw_memory.tools.recall_support import register_recall_support_tools

    tools: dict[str, object] = {}

    class _Captured:
        def tool(self) -> object:
            return lambda fn: tools.setdefault(fn.__name__, fn)

    register_recall_support_tools(_Captured())  # type: ignore[arg-type]
    answer = asyncio.run(tools["memory_graph_related"](namespace="project:a", learning_id="L-root", **arguments))  # type: ignore[operator]

    assert answer["status"] == "invalid", answer
