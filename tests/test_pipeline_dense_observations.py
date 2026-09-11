"""Request-owned raw dense evidence must not mutate the legacy fusion policy."""

from __future__ import annotations

import math

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.pipeline import hybrid_search
from trw_memory.security.namespace_scope import NamespaceScopeError

from ._test_scope_support import DEFAULT_SCOPE

pytestmark = pytest.mark.unit


def _entries() -> list[MemoryEntry]:
    return [
        MemoryEntry(id="semantic", content="restore connectivity", importance=0.1),
        MemoryEntry(id="literal", content="network repair", importance=1.0),
    ]


@pytest.mark.parametrize("importance_alpha", [0.0, 0.3, 1.0])
def test_raw_semantic_scores_precede_caps_and_importance(importance_alpha: float) -> None:
    entries = _entries()
    before = [entry.model_dump() for entry in entries]
    batches: list[tuple[tuple[str, float], ...]] = []
    kwargs = {
        "scope": DEFAULT_SCOPE,
        "query_embedding": [1.0, 0.0],
        "stored_embeddings": {"semantic": [1.0, 0.0], "literal": [0.6, 0.8]},
        "vector_candidates": 1,
        "top_k": 1,
        "importance_alpha": importance_alpha,
    }
    baseline = hybrid_search("network repair", entries, **kwargs)
    observed = hybrid_search("network repair", entries, dense_observer=batches.append, **kwargs)
    assert observed == baseline
    assert len(batches) == 1
    assert dict(batches[0]) == pytest.approx({"semantic": 1.0, "literal": 0.6})
    assert [entry.model_dump() for entry in entries] == before


def test_hype_observations_collapse_before_exposure_without_changing_cap() -> None:
    entries = _entries()
    batches: list[tuple[tuple[str, float], ...]] = []
    kwargs = {
        "scope": DEFAULT_SCOPE,
        "query_embedding": [1.0, 0.0],
        "stored_embeddings": {
            "semantic": [0.0, 1.0],
            "semantic#hype0": [1.0, 0.0],
            "semantic#hype1": [0.8, 0.6],
            "literal": [0.6, 0.8],
            "outside#hype0": [1.0, 0.0],
        },
        "collapse_hype": True,
        "vector_candidates": 2,
        "top_k": 5,
    }
    baseline = hybrid_search("unrelated", entries, **kwargs)
    observed = hybrid_search("unrelated", entries, dense_observer=batches.append, **kwargs)
    assert observed == baseline
    assert dict(batches[0]) == pytest.approx({"semantic": 1.0, "literal": 0.6})
    assert all("#hype" not in eid for eid, _ in batches[0])


def test_scope_violation_never_calls_observer() -> None:
    batches: list[tuple[tuple[str, float], ...]] = []
    entries = [MemoryEntry(id="secret", content="network", namespace="project:other")]
    with pytest.raises(NamespaceScopeError):
        hybrid_search(
            "network",
            entries,
            scope=DEFAULT_SCOPE,
            query_embedding=[1.0],
            stored_embeddings={"secret": [1.0]},
            dense_observer=batches.append,
        )
    assert batches == []


def test_nonfinite_scores_are_not_evidence() -> None:
    batches: list[tuple[tuple[str, float], ...]] = []
    hybrid_search(
        "network",
        _entries(),
        scope=DEFAULT_SCOPE,
        query_embedding=[1.0, 0.0],
        stored_embeddings={"semantic": [math.nan, 0.0], "literal": [0.6, 0.8]},
        dense_observer=batches.append,
    )
    assert dict(batches[0]) == pytest.approx({"literal": 0.6})


def test_callback_failure_propagates_and_snapshot_cannot_mutate_ranking() -> None:
    def fail(snapshot: tuple[tuple[str, float], ...]) -> None:
        assert isinstance(snapshot, tuple)
        assert isinstance(snapshot[0], tuple)
        raise RuntimeError("collector failed")

    with pytest.raises(RuntimeError, match="collector failed"):
        hybrid_search(
            "network",
            _entries(),
            scope=DEFAULT_SCOPE,
            query_embedding=[1.0],
            stored_embeddings={"semantic": [1.0]},
            dense_observer=fail,
        )


def test_unavailable_dense_reports_empty_batch_without_fabricating_scores() -> None:
    batches: list[tuple[tuple[str, float], ...]] = []
    hybrid_search(
        "network",
        _entries(),
        scope=DEFAULT_SCOPE,
        stored_embeddings={"semantic": [1.0]},
        dense_observer=batches.append,
    )
    assert batches == [()]


def test_observation_does_not_request_a_second_query_embedding() -> None:
    class CountingEmbedder:
        calls = 0

        def available(self) -> bool:
            return True

        def embed(self, text: str) -> list[float]:
            self.calls += 1
            return [1.0, 0.0]

    embedder = CountingEmbedder()
    batches: list[tuple[tuple[str, float], ...]] = []
    hybrid_search(
        "network",
        _entries(),
        scope=DEFAULT_SCOPE,
        embedder=embedder,  # type: ignore[arg-type] -- deterministic protocol subset used by dense_search
        stored_embeddings={"semantic": [1.0, 0.0], "literal": [0.6, 0.8]},
        vector_candidates=1,
        dense_observer=batches.append,
    )
    assert embedder.calls == 1
    assert len(batches[0]) == 2
