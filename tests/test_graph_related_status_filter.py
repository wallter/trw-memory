"""Status-filtering tests for the graph ``related`` recall path.

The main recall path lists only ``MemoryStatus.ACTIVE`` entries. The graph
traversal ``related`` path hydrates neighbour nodes via ``backend.get(namespace="default")``,
which returns an entry regardless of status. Without an explicit filter the
``related`` block re-introduces the obsolete-leak bug (obsolete / archived /
poisoned / resolved learnings surfaced to the caller).
"""

from __future__ import annotations

from datetime import datetime, timezone

from trw_memory.models.memory import MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools._recall_helpers import _graph_related

from .conftest import make_entry


def _link(backend: SQLiteBackend, source_id: str, target_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with backend._lock:
        backend._conn.execute(
            "INSERT INTO memory_graph_edges "
            "(source_id, target_id, edge_type, weight, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, target_id, "related_to", 0.9, now),
        )
        backend._conn.commit()


class TestGraphRelatedStatusFilter:
    def test_obsolete_neighbour_is_not_surfaced(self, sqlite_memory_backend: SQLiteBackend) -> None:
        backend = sqlite_memory_backend
        backend.store(make_entry(entry_id="seed", content="active seed", status=MemoryStatus.ACTIVE))
        backend.store(make_entry(entry_id="ghost", content="obsolete neighbour", status=MemoryStatus.OBSOLETE))
        _link(backend, "seed", "ghost")

        related = _graph_related([{"id": "seed"}], depth=2, backend=backend, conn=backend._conn, namespace="default")

        ids = {str(item["id"]) for item in related}
        assert "ghost" not in ids, "obsolete neighbour leaked into graph related results"

    def test_active_neighbour_is_surfaced(self, sqlite_memory_backend: SQLiteBackend) -> None:
        backend = sqlite_memory_backend
        backend.store(make_entry(entry_id="seed", content="active seed", status=MemoryStatus.ACTIVE))
        backend.store(make_entry(entry_id="kin", content="active neighbour", status=MemoryStatus.ACTIVE))
        _link(backend, "seed", "kin")

        related = _graph_related([{"id": "seed"}], depth=2, backend=backend, conn=backend._conn, namespace="default")

        ids = {str(item["id"]) for item in related}
        assert "kin" in ids, "active neighbour should be surfaced via graph traversal"

    def test_archived_and_poisoned_neighbours_filtered(self, sqlite_memory_backend: SQLiteBackend) -> None:
        backend = sqlite_memory_backend
        backend.store(make_entry(entry_id="seed", content="active seed", status=MemoryStatus.ACTIVE))
        backend.store(make_entry(entry_id="arch", content="archived", status=MemoryStatus.ARCHIVED))
        backend.store(make_entry(entry_id="poison", content="poisoned", status=MemoryStatus.OBSOLETE_POISONED))
        backend.store(make_entry(entry_id="kin", content="active", status=MemoryStatus.ACTIVE))
        _link(backend, "seed", "arch")
        _link(backend, "seed", "poison")
        _link(backend, "seed", "kin")

        related = _graph_related([{"id": "seed"}], depth=2, backend=backend, conn=backend._conn, namespace="default")

        ids = {str(item["id"]) for item in related}
        assert ids == {"kin"}
