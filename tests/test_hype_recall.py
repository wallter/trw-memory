"""PRD-CORE-272 FR03: canonical membership precedes dense caps and observation."""

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.pipeline import hybrid_search

from ._test_scope_support import DEFAULT_SCOPE


@pytest.mark.parametrize("observe", [False, True])
def test_legacy_siblings_never_consume_canonical_candidate_limit(observe):
    entries = [MemoryEntry(id="P", content="parent"), MemoryEntry(id="P#hype0", content="canonical")]
    batches = []
    result = hybrid_search(
        "unrelated",
        entries,
        scope=DEFAULT_SCOPE,
        query_embedding=[1.0, 0.0],
        stored_embeddings={"P": [0.0, 1.0], "P#hype0": [0.8, 0.6], **{f"P#hype{i}": [1.0, 0.0] for i in range(1, 100)}},
        bm25_candidates=0,
        vector_candidates=1,
        top_k=1,
        dense_observer=batches.append if observe else None,
    )
    assert [entry.id for entry in result] == ["P#hype0"]
    if observe:
        assert dict(batches[0]) == pytest.approx({"P": 0.0, "P#hype0": 0.8})


@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_collapse_activation_rejects_even_empty_input(value):
    with pytest.raises(TypeError, match="retired"):
        hybrid_search("q", [], scope=DEFAULT_SCOPE, collapse_hype=value)
