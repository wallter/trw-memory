"""Namespace-scoping behaviour for vector ops (data-isolation fixes).

Covers two adversarially-verified bugs in storage/_vector_ops.py:

- ``existing_vector_ids`` did a full-table scan of ``vec_index`` with no
  namespace predicate, returning every tenant's vector ids.
- ``search_vectors`` ran an unscoped KNN, so ids whose canonical memory row
  lives in another namespace could surface (cross-namespace leak).

``vec_index`` carries no namespace column, so scoping joins to ``memories``.
These are integration tests against a real sqlite-vec backend.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend

_DIM = 4


def _backend(tmp_path: Path) -> SQLiteBackend:
    backend = SQLiteBackend(tmp_path / "vec_ns.db", dim=_DIM)
    if not backend.vec_available:
        pytest.skip("sqlite-vec virtual table unavailable")
    return backend


def _store(backend: SQLiteBackend, entry_id: str, namespace: str, embedding: list[float]) -> None:
    now = datetime.now(timezone.utc)
    backend.store(
        MemoryEntry(
            id=entry_id,
            content=f"content {entry_id}",
            detail="",
            tags=[],
            namespace=namespace,
            status=MemoryStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    backend.upsert_vector(entry_id, embedding)


class TestExistingVectorIdsNamespaceScope:
    def test_namespace_scoped_excludes_other_namespace_ids(self, tmp_path: Path) -> None:
        backend = _backend(tmp_path)
        _store(backend, "A-1", "ns-a", [1.0, 0.0, 0.0, 0.0])
        _store(backend, "A-2", "ns-a", [0.0, 1.0, 0.0, 0.0])
        _store(backend, "B-1", "ns-b", [0.0, 0.0, 1.0, 0.0])

        scoped = backend.existing_vector_ids(namespace="ns-a")
        assert scoped == {"A-1", "A-2"}
        assert "B-1" not in scoped

    def test_no_namespace_returns_all_ids(self, tmp_path: Path) -> None:
        backend = _backend(tmp_path)
        _store(backend, "A-1", "ns-a", [1.0, 0.0, 0.0, 0.0])
        _store(backend, "B-1", "ns-b", [0.0, 0.0, 1.0, 0.0])

        assert backend.existing_vector_ids() == {"A-1", "B-1"}


class TestSearchVectorsNamespaceScope:
    def test_search_does_not_return_other_namespace_vector(self, tmp_path: Path) -> None:
        backend = _backend(tmp_path)
        # The CLOSEST vector to the query lives in ns-b — without a namespace
        # predicate it would be the top KNN hit and leak across the boundary.
        query = [1.0, 0.0, 0.0, 0.0]
        _store(backend, "B-CLOSE", "ns-b", [1.0, 0.0, 0.0, 0.0])  # identical to query
        _store(backend, "A-FAR", "ns-a", [0.0, 1.0, 0.0, 0.0])  # orthogonal but in-namespace

        results = backend.search_vectors(query, top_k=5, namespace="ns-a")
        ids = {eid for eid, _ in results}
        assert "B-CLOSE" not in ids  # cross-namespace leak closed
        assert "A-FAR" in ids  # in-namespace hit still returned

    def test_search_returns_in_namespace_hits_when_other_ns_dominates(self, tmp_path: Path) -> None:
        backend = _backend(tmp_path)
        query = [1.0, 0.0, 0.0, 0.0]
        # Several near neighbours in ns-b crowd the global KNN front; the
        # over-fetch + post-filter must still surface the ns-a hit.
        for i in range(8):
            _store(backend, f"B-{i}", "ns-b", [1.0 - i * 0.01, 0.0, 0.0, 0.0])
        _store(backend, "A-1", "ns-a", [0.9, 0.1, 0.0, 0.0])

        results = backend.search_vectors(query, top_k=3, namespace="ns-a")
        ids = {eid for eid, _ in results}
        assert ids == {"A-1"}

    def test_unscoped_search_can_return_any_namespace(self, tmp_path: Path) -> None:
        backend = _backend(tmp_path)
        query = [1.0, 0.0, 0.0, 0.0]
        _store(backend, "B-CLOSE", "ns-b", [1.0, 0.0, 0.0, 0.0])
        _store(backend, "A-FAR", "ns-a", [0.0, 1.0, 0.0, 0.0])

        results = backend.search_vectors(query, top_k=5)  # no namespace = legacy
        ids = {eid for eid, _ in results}
        assert "B-CLOSE" in ids  # legacy unscoped behaviour preserved
