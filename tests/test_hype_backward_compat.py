"""Disabled-path ordered baseline captured from f80062857a before retirement."""

from datetime import datetime, timezone

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.pipeline import hybrid_search

from ._test_scope_support import DEFAULT_SCOPE


def test_default_and_neutral_tombstone_preserve_frozen_baseline():
    entries = [
        MemoryEntry(id=i, content=c, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        for i, c in [("a", "alpha beta gamma"), ("b", "beta gamma delta"), ("c", "gamma delta epsilon")]
    ]
    baseline = hybrid_search("beta gamma", entries, top_k=10, scope=DEFAULT_SCOPE)
    with pytest.warns(UserWarning, match="retired"):
        neutral = hybrid_search("beta gamma", entries, top_k=10, collapse_hype=False, scope=DEFAULT_SCOPE)
    assert [e.id for e in baseline] == [e.id for e in neutral] == ["a", "b", "c"]


def test_dense_hybrid_frozen_baseline_preserves_canonical_suffix_ids():
    entries = [
        MemoryEntry(id=i, content=c, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        for i, c in [("a", "alpha beta gamma"), ("a#hype0", "beta gamma delta"), ("c", "gamma delta epsilon")]
    ]
    result = hybrid_search(
        "beta gamma",
        entries,
        top_k=3,
        scope=DEFAULT_SCOPE,
        query_embedding=[1.0, 0.0],
        stored_embeddings={"a": [0.6, 0.8], "a#hype0": [0.8, 0.6], "c": [0.0, 1.0], "a#hype1": [1.0, 0.0]},
    )
    assert [e.id for e in result] == ["a", "a#hype0", "c"]
