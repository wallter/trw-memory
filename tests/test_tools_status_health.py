"""PRD-CORE-280: ``memory_status`` measures one namespace's health from the store itself.

trw-mcp's pipeline-health probes and store inventory read this block instead of
opening a checkout's ``memory.db``, which the daemon never writes.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests._timing import assert_budget
from tests.conftest import make_entry
from trw_memory._graph_primitives import _upsert_edge
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.status import memory_status_impl

_NS = "project:acme-1a2b3c4d"
_OTHER = "project:other-00000000"


def test_the_health_block_counts_one_namespace_without_canaries(tmp_path: Path) -> None:
    store = SQLiteBackend(tmp_path / "memory.db", dim=4)
    try:
        store.store(make_entry(entry_id="L-1", namespace=_NS, access_count=0))
        store.store(make_entry(entry_id="L-2", namespace=_NS, source="team_sync"))
        # A canary counts toward nothing, its recall count included; "false" is an ordinary row.
        store.store(make_entry(entry_id="C-1", namespace=_NS, metadata={"system_canary": "true"}))
        store.store(make_entry(entry_id="L-3", namespace=_NS, metadata={"system_canary": "false"}))
        # The other namespace's row, vector and edge are not this namespace's.
        store.store(make_entry(entry_id="L-9", namespace=_OTHER))
        store.upsert_vector("L-1", [1.0, 0.0, 0.0, 0.0], namespace=_NS)
        store.upsert_vector("C-1", [0.0, 1.0, 0.0, 0.0], namespace=_NS)
        store.upsert_vector("L-9", [0.0, 0.0, 1.0, 0.0], namespace=_OTHER)
        with store._lock:
            _upsert_edge(store._conn, "L-1", "L-2", "related_to", 0.5, "2026-09-23T00:00:00+00:00", namespace=_NS)
            _upsert_edge(store._conn, "L-9", "L-8", "related_to", 0.5, "2026-09-23T00:00:00+00:00", namespace=_OTHER)
            store._conn.commit()
        store.update("L-1", namespace=_NS, recall_count=4)
        store.update("C-1", namespace=_NS, recall_count=40)

        health = memory_status_impl(_NS, backend=store)["health"]
    finally:
        store.close()

    embedded = 1 if store.vec_available else None
    assert health == {
        "entries": 3,
        "synced": 1,
        "edges": 1,
        "has_relations": True,
        "embedded": embedded,
        "max_recall_count": 4,
    }


def test_another_namespaces_edge_is_not_a_relation_here(tmp_path: Path) -> None:
    store = SQLiteBackend(tmp_path / "memory.db", dim=4)
    try:
        store.store(make_entry(entry_id="L-1", namespace=_NS, tags=["alpha"]))
        store.store(make_entry(entry_id="L-2", namespace=_NS, tags=["beta"]))
        with store._lock:
            _upsert_edge(store._conn, "L-8", "L-9", "related_to", 0.5, "2026-09-23T00:00:00+00:00", namespace=_OTHER)
            store._conn.commit()
        health = memory_status_impl(_NS, backend=store)["health"]
    finally:
        store.close()

    assert (health["edges"], health["has_relations"]) == (0, False)


def test_an_unreadable_vector_index_is_an_error_not_zero_vectors(tmp_path: Path) -> None:
    store = SQLiteBackend(tmp_path / "memory.db", dim=4)
    try:
        if not store.supports_vectors():
            pytest.skip("sqlite-vec is not loadable here, so there is no vector index to break")
        store.store(make_entry(entry_id="L-1", namespace=_NS))
        with store._lock:
            store._conn.execute("DROP TABLE vec_index")
        answer = memory_status_impl(_NS, backend=store)
    finally:
        store.close()

    assert answer["status"] == "error"
    assert "health" not in answer


def _seed_twenty_thousand_rows(store: SQLiteBackend) -> None:
    with store._lock:
        store._conn.executemany(
            "INSERT INTO memories (id, namespace, content, metadata, created_at, updated_at) "
            "VALUES (?, ?, 'x', '{}', '2026-09-23T00:00:00+00:00', '2026-09-23T00:00:00+00:00')",
            [(f"L-{i}", _NS) for i in range(20_000)],
        )
        if store.vec_available:
            store._conn.executemany(
                "INSERT INTO vec_index (entry_id, namespace) VALUES (?, ?)",
                [(f"L-{i}", _NS) for i in range(20_000)],
            )
        store._conn.commit()


def test_the_health_block_counts_twenty_thousand_rows_correctly(tmp_path: Path) -> None:
    """Counts are SQL aggregates over the full table, not a sampled or truncated subset."""
    store = SQLiteBackend(tmp_path / "memory.db", dim=4)
    try:
        _seed_twenty_thousand_rows(store)
        health = memory_status_impl(_NS, backend=store)["health"]
    finally:
        store.close()

    assert health["entries"] == 20_000
    assert health["embedded"] == (20_000 if store.vec_available else None)


@pytest.mark.requires_local_timing
def test_the_health_block_stays_cheap_at_twenty_thousand_rows(tmp_path: Path) -> None:
    """Counts are SQL aggregates, not materialised ids: one status call on 20k rows stays well under a second."""
    store = SQLiteBackend(tmp_path / "memory.db", dim=4)
    try:
        _seed_twenty_thousand_rows(store)
        started = time.perf_counter()
        memory_status_impl(_NS, backend=store)
        elapsed = time.perf_counter() - started
    finally:
        store.close()

    assert_budget("memory_status_20k_rows", elapsed, 1.0, "s")


def test_an_empty_namespace_is_measured_as_empty(tmp_path: Path) -> None:
    store = SQLiteBackend(tmp_path / "memory.db", dim=4)
    try:
        health = memory_status_impl(_NS, backend=store)["health"]
    finally:
        store.close()

    assert (health["entries"], health["edges"], health["has_relations"]) == (0, 0, False)


def test_a_namespace_related_only_by_shared_tags_has_relations(tmp_path: Path) -> None:
    """Tag co-occurrence materialises no edge (PRD-CORE-245 FR07); an edge count alone would call it dead."""
    store = SQLiteBackend(tmp_path / "memory.db", dim=4)
    try:
        for entry_id in ("L-1", "L-2", "L-3"):
            store.store(make_entry(entry_id=entry_id, namespace=_NS, tags=["alpha", "beta"]))
        health = memory_status_impl(_NS, backend=store)["health"]
    finally:
        store.close()

    assert (health["edges"], health["has_relations"]) == (0, True)


def test_a_store_it_cannot_read_answers_an_error_not_an_empty_graph(tmp_path: Path) -> None:
    """Could-not-read is never "no relations": both health consumers would call the graph dead (W06)."""
    store = SQLiteBackend(tmp_path / "memory.db", dim=4)
    try:
        store.store(make_entry(entry_id="L-1", namespace=_NS))
        with store._lock:
            store._conn.execute("DROP TABLE memory_graph_edges")
        answer = memory_status_impl(_NS, backend=store)
    finally:
        store.close()

    assert answer["status"] == "error"
    assert "health" not in answer


def test_status_reports_the_embedder_without_loading_it_and_coverage_by_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD-CORE-302 C3: trw-mcp reads embedder state and coverage here; a status read never loads a model."""
    pytest.importorskip("sqlite_vec")
    from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance

    def _no_load(**_kw: object) -> None:
        raise AssertionError("memory_status must not load a model")

    monkeypatch.setattr("trw_memory.tools._embedder.get_local_embedder", _no_load, raising=False)
    store = SQLiteBackend(tmp_path / "memory.db", dim=3)
    try:
        space = EmbeddingSpace("d" * 64, "test-encoder:d", 3)
        for entry_id in ("L-proven", "L-unknown", "L-plain"):
            store.store(make_entry(entry_id=entry_id, namespace=_NS))
        proof = VectorProvenance.for_vector(space, "x", [1.0, 0.0, 0.0])
        store.upsert_vector("L-proven", [1.0, 0.0, 0.0], namespace=_NS, provenance=proof)
        store.upsert_vector("L-unknown", [0.0, 1.0, 0.0], namespace=_NS)

        answer = memory_status_impl(_NS, backend=store)
    finally:
        store.close()

    embedder = answer["embedder"]
    assert isinstance(embedder, dict) and embedder["loaded"] is False and embedder["space"] is None
    # Nothing is loaded, so which stored space is the active one is not known yet.
    assert answer["coverage"] == {
        "active_space": None,
        "other_space": None,
        "unknown_provenance": 1,
        "outside_active_space": None,
        "no_vector": 1,
    }


def test_coverage_splits_active_and_other_space_once_the_model_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sqlite_vec")
    from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance

    active, other = EmbeddingSpace("e" * 64, "test-encoder:e", 3), EmbeddingSpace("f" * 64, "test-encoder:f", 3)

    class _Loaded:
        def embedding_space(self) -> EmbeddingSpace:
            return active

    monkeypatch.setattr("trw_memory.tools._embedder.loaded_local_embedder", lambda _key: _Loaded())
    store = SQLiteBackend(tmp_path / "memory.db", dim=3)
    try:
        for entry_id, space in (("L-a1", active), ("L-a2", active), ("L-o", other)):
            store.store(make_entry(entry_id=entry_id, namespace=_NS))
            store.upsert_vector(
                entry_id,
                [1.0, 0.0, 0.0],
                namespace=_NS,
                provenance=VectorProvenance.for_vector(space, "x", [1.0, 0.0, 0.0]),
            )

        answer = memory_status_impl(_NS, backend=store)
    finally:
        store.close()

    assert answer["coverage"] == {
        "active_space": 2,
        "other_space": 1,
        "unknown_provenance": 0,
        "outside_active_space": 1,
        "no_vector": 0,
    }
    assert answer["embedder"]["loaded"] is True  # type: ignore[index]
