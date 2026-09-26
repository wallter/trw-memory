"""When a learn-time dedup KNN window supports a dense verdict (batch 2026-09-19 X3-4).

``memory_similar`` runs this rule in the daemon; trw-mcp decides exhaustively
(its YAML scan) whenever it answers ``None``. A window of other-space vectors
compares nothing, and a single-space window is trusted only when the store's
provenance census shows the WHOLE namespace in the loaded space: an unsupported
census, an unknown-provenance row or an other-space row past the window defers.
"""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from trw_memory.embeddings._space_gate import comparable_neighbours
from trw_memory.embeddings.provenance import EmbeddingSpace, StoredVector, VectorProvenance

OLD_SPACE = EmbeddingSpace("a" * 64, "trw-declared-encoder-v1:all-MiniLM-L6-v2", 2)
NEW_SPACE = EmbeddingSpace("b" * 64, "trw-declared-encoder-v1:BAAI/bge-small-en-v1.5", 2)


def _stored(vector: Sequence[float], space: EmbeddingSpace | None) -> StoredVector:
    proof = VectorProvenance.for_vector(space, "text", list(vector)) if space is not None else None
    return StoredVector(tuple(vector), proof)


def _backend(
    hits: list[tuple[str, float]], records: dict[str, StoredVector], census: object, rows: int | None = None
) -> MagicMock:
    backend = MagicMock()
    backend.search_vectors.return_value = hits
    backend.get_vector_records.return_value = records
    backend.vector_space_census.return_value = census
    backend.count.return_value = len(hits) if rows is None else rows
    return backend


def _window(backend: MagicMock) -> list[tuple[str, float]] | None:
    return comparable_neighbours(backend, [1.0, 0.0], NEW_SPACE, namespace="project:a", top_k=10, surface="test")


def test_an_all_other_space_window_is_incomplete() -> None:
    hits = [(f"L-old{i}", 0.01 * i) for i in range(10)]
    records = {entry_id: _stored((1.0, 0.0), OLD_SPACE) for entry_id, _ in hits}

    assert _window(_backend(hits, records, {OLD_SPACE: 10})) is None


def test_a_mixed_window_is_incomplete_even_with_an_admitted_hit() -> None:
    hits = [("L-unmigrated", 0.0), ("L-unrelated", 1.4)]
    records = {"L-unmigrated": _stored((1.0, 0.0), OLD_SPACE), "L-unrelated": _stored((0.0, 1.0), NEW_SPACE)}

    assert _window(_backend(hits, records, {NEW_SPACE: 1, OLD_SPACE: 1})) is None


def test_a_single_space_window_over_a_proven_namespace_is_the_verdict() -> None:
    backend = _backend([("L-new", 0.0)], {"L-new": _stored((1.0, 0.0), NEW_SPACE)}, {NEW_SPACE: 1})

    assert _window(backend) == [("L-new", 0.0)]


@pytest.mark.parametrize(
    "census",
    [
        None,
        {NEW_SPACE: 1, OLD_SPACE: 1},
        {NEW_SPACE: 1, None: 1},
        MagicMock(),
        {},
        {NEW_SPACE: 0},
        {NEW_SPACE: True},
        {NEW_SPACE: "1"},
    ],
    ids=[
        "unsupported",
        "other-space-row-past-the-window",
        "unknown-provenance-row",
        "not-a-mapping",
        "empty",
        "zero-count",
        "bool-count",
        "str-count",
    ],
)
def test_an_unproven_namespace_makes_a_single_space_window_incomplete(census: object) -> None:
    backend = _backend([("L-new", 0.5)], {"L-new": _stored((1.0, 0.0), NEW_SPACE)}, census)

    assert _window(backend) is None


def test_a_census_smaller_than_the_window_proves_nothing() -> None:
    hits = [("L-a", 0.5), ("L-b", 0.6)]
    records = {"L-a": _stored((1.0, 0.0), NEW_SPACE), "L-b": _stored((1.0, 0.0), NEW_SPACE)}

    assert _window(_backend(hits, records, {NEW_SPACE: 1})) is None


def test_a_census_that_misses_a_vectorless_row_proves_nothing() -> None:
    """C12 rc4: one in-space vector proved a window although a vectorless near-duplicate was never compared."""
    backend = _backend([("L-unrelated", 1.4)], {"L-unrelated": _stored((0.0, 1.0), NEW_SPACE)}, {NEW_SPACE: 1}, rows=2)

    assert _window(backend) is None


def test_an_empty_window_is_a_complete_empty_verdict() -> None:
    assert _window(_backend([], {}, None)) == []
