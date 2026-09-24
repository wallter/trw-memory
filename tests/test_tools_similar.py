"""memory_similar: the trw_learn dedup KNN, answered by the store's owner (PRD-CORE-280 FR01 slice c)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.similar import memory_similar_impl

SPACE_A = EmbeddingSpace("a" * 64, "test-encoder:a", 3)
SPACE_B = EmbeddingSpace("b" * 64, "test-encoder:b", 3)
NEAR = [1.0, 0.0, 0.0]


def _put(
    backend: SQLiteBackend, entry_id: str, vector: list[float], space: EmbeddingSpace | None, **fields: object
) -> None:
    entry = MemoryEntry(id=entry_id, content=f"content {entry_id}", namespace="default", **fields)  # type: ignore[arg-type]
    backend.store(entry)
    proof = VectorProvenance.for_vector(space, f"{entry.content} {entry.detail}", vector) if space else None
    backend.upsert_vector(entry_id, vector, namespace="default", provenance=proof)


def _similar(backend: SQLiteBackend, space: EmbeddingSpace) -> dict[str, object]:
    return memory_similar_impl("default", NEAR, dataclasses.asdict(space), 10, backend=backend)


def test_neighbours_in_one_space_are_a_complete_verdict_with_status(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "m.db", dim=3)
    _put(backend, "L-live", NEAR, SPACE_A)
    _put(backend, "L-gone", [0.0, 1.0, 0.0], SPACE_A, status=MemoryStatus.OBSOLETE)

    result = _similar(backend, SPACE_A)

    assert result["status"] == "ok" and result["complete"] is True
    hits = {hit["id"]: hit for hit in result["hits"]}  # type: ignore[union-attr]
    assert hits["L-live"]["similarity"] > 0.99 and hits["L-live"]["active"] is True
    assert hits["L-gone"]["active"] is False


def test_a_neighbour_from_another_space_makes_the_verdict_incomplete(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "m.db", dim=3)
    _put(backend, "L-a", NEAR, SPACE_A)
    _put(backend, "L-b", NEAR, SPACE_B)

    result = _similar(backend, SPACE_A)
    assert (result["complete"], result["hits"]) == (False, [])


def test_without_a_space_only_the_window_size_is_reported(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "m.db", dim=3)
    _put(backend, "L-a", NEAR, SPACE_A)

    result = memory_similar_impl("default", NEAR, None, 10, backend=backend)

    assert result == {"status": "ok", "window": 1, "complete": True, "hits": []}


def test_an_invalid_space_is_refused(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "m.db", dim=3)

    assert memory_similar_impl("default", NEAR, {"encoding": "x"}, 10, backend=backend)["status"] == "invalid"
