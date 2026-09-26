"""Tests for the BM25 corpus invalidation cache.

The cache is a small LRU of ``BM25Okapi`` models keyed on the *set* of entries
(ids plus lexical fields) that produced each one.  Two ``bm25_search`` calls
presenting the same corpus reuse the model (and the tokenized corpus); any
change to the id set misses and rebuilds.  Several corpora (one per namespace)
stay cached at once, bounded by model count and total indexed rows.  This eliminates the O(N) per-call corpus rebuild that
costs 7-15GB RAM and seconds at 1M+ entries.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import trw_memory.retrieval.bm25 as bm25_mod
from trw_memory.retrieval.bm25 import bm25_search

from ._test_retrieval_support import make_entry


def _clear_bm25_cache() -> None:
    """Test seam mirroring the removed ``bm25.clear_bm25_cache``."""
    with bm25_mod._bm25_cache_lock:
        bm25_mod._bm25_cache.clear()


@pytest.fixture(autouse=True)
def _reset_bm25_cache() -> Iterator[None]:
    """Reset the module-level cache before and after each test for isolation."""
    _clear_bm25_cache()
    yield
    _clear_bm25_cache()


class _CountingBM25:
    """Drop-in BM25Okapi replacement that counts constructions.

    Delegates scoring to the real ``BM25Okapi`` so search semantics (and the
    Jaccard fallback) are unaffected by the instrumentation.
    """

    construction_count = 0

    def __init__(self, corpus: list[list[str]]) -> None:
        type(self).construction_count += 1
        from rank_bm25 import BM25Okapi as _RealBM25

        self._delegate = _RealBM25(corpus)

    def get_scores(self, query: list[str]) -> list[float]:
        return list(self._delegate.get_scores(query))


@pytest.fixture
def counting_bm25(monkeypatch: pytest.MonkeyPatch) -> type[_CountingBM25]:
    """Patch the BM25Okapi symbol used inside bm25.py with a counting double."""
    _CountingBM25.construction_count = 0
    monkeypatch.setattr(bm25_mod, "BM25Okapi", _CountingBM25)
    return _CountingBM25


class TestBm25Cache:
    def test_bm25_model_reused_on_same_corpus(self, counting_bm25: type[_CountingBM25]) -> None:
        """BM25Okapi is not rebuilt when the entry id set is unchanged."""
        entries = [
            make_entry("e1", "pydantic validation error handling"),
            make_entry("e2", "fastmcp middleware pattern"),
            make_entry("e3", "structlog event keyword reserved"),
        ]

        first = bm25_search("pydantic validation", entries)
        second = bm25_search("fastmcp middleware", entries)

        # Built exactly once across two calls with the identical id set.
        assert counting_bm25.construction_count == 1
        # The second call still returns correct, query-specific results.
        assert "e1" in [eid for eid, _ in first]
        assert "e2" in [eid for eid, _ in second]

    def test_bm25_model_rebuilt_on_corpus_change(self, counting_bm25: type[_CountingBM25]) -> None:
        """BM25Okapi is rebuilt when the entry id set changes (add/remove/swap)."""
        base = [
            make_entry("e1", "pydantic validation error handling"),
            make_entry("e2", "fastmcp middleware pattern"),
        ]
        bm25_search("pydantic", base)
        assert counting_bm25.construction_count == 1

        # Add an entry → new id set → rebuild.
        added = [*base, make_entry("e3", "structlog reserved keyword")]
        bm25_search("structlog", added)
        assert counting_bm25.construction_count == 2

        # Remove an entry → different id set → rebuild.
        removed = [base[0]]
        bm25_search("pydantic", removed)
        assert counting_bm25.construction_count == 3

        # Swap an id (same count, different members) → rebuild, not a hit.
        swapped = [
            base[0],
            make_entry("e9", "fastmcp middleware pattern"),
        ]
        bm25_search("fastmcp", swapped)
        assert counting_bm25.construction_count == 4

    def test_cache_invalidates_on_id_swap_same_count(self, counting_bm25: type[_CountingBM25]) -> None:
        """Same entry count but different id membership must NOT be a cache hit.

        Guards against a count-only invalidation bug — the brief explicitly
        requires comparing the id *set*, not just ``len(entries)``.
        """
        a = [make_entry("a1", "alpha"), make_entry("a2", "beta")]
        b = [make_entry("b1", "alpha"), make_entry("b2", "beta")]

        bm25_search("alpha", a)
        bm25_search("alpha", b)

        assert counting_bm25.construction_count == 2

    def test_cache_hit_independent_of_entry_order(self, counting_bm25: type[_CountingBM25]) -> None:
        """Reordering the same id set is still a cache hit and stays correct."""
        entries = [
            make_entry("x", "machine learning training data"),
            make_entry("y", "neural network weights gradient"),
            make_entry("z", "pydantic schema validation"),
        ]
        bm25_search("machine learning", entries)
        reordered = [entries[2], entries[0], entries[1]]
        results = bm25_search("machine learning", reordered)

        assert counting_bm25.construction_count == 1
        # Correct entry still wins despite the reordered corpus alignment.
        assert results[0][0] == "x"

    def test_reused_model_matches_uncached_results(self, counting_bm25: type[_CountingBM25]) -> None:
        """A cache-hit query returns identical results to a fresh build."""
        entries = [
            make_entry("alpha", "machine learning training data pipeline"),
            make_entry("beta", "neural network weights gradient descent"),
            make_entry("gamma", "pydantic model validation schema"),
        ]
        # Prime the cache.
        bm25_search("machine learning", entries)
        cached_results = bm25_search("neural gradient", entries)

        # Force a fresh build by clearing the cache, then compare.
        _clear_bm25_cache()
        fresh_results = bm25_search("neural gradient", entries)

        assert cached_results == fresh_results

    def test_duplicate_ids_not_cached(self, counting_bm25: type[_CountingBM25]) -> None:
        """Entries with duplicate ids are never cached (ambiguous reorder key).

        With duplicate ids ``len(id_set) != len(entries)``, so the by-id reorder
        on a hit would be unsafe — each call must rebuild.
        """
        dupes = [
            make_entry("dup", "foo bar baz"),
            make_entry("dup", "foo bar qux"),
        ]
        bm25_search("foo", dupes)
        bm25_search("bar", dupes)

        assert counting_bm25.construction_count == 2
        assert len(bm25_mod._bm25_cache) == 0


class TestBm25CacheLru:
    """Several corpora stay cached at once, within a model-count and row bound."""

    def test_alternating_namespaces_both_hit(self, counting_bm25: type[_CountingBM25]) -> None:
        """Two corpora recalled alternately each build once, then always hit.

        A single-slot cache evicted one namespace's model on every switch, so
        concurrent recalls over two namespaces rebuilt O(N) on every call.
        """
        alpha = [make_entry("a1", "pottery class schedule"), make_entry("a2", "adoption agency research")]
        beta = [make_entry("b1", "camping trip gear"), make_entry("b2", "marathon training plan")]

        for _ in range(3):
            assert bm25_search("pottery", alpha)[0][0] == "a1"
            assert bm25_search("marathon", beta)[0][0] == "b2"

        assert counting_bm25.construction_count == 2

    def test_least_recently_used_model_is_evicted_at_capacity(
        self, counting_bm25: type[_CountingBM25], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bm25_mod, "_BM25_CACHE_MAX_MODELS", 2)
        corpora = [[make_entry(f"{name}1", f"{name} alpha"), make_entry(f"{name}2", "beta")] for name in "xyz"]

        bm25_search("alpha", corpora[0])
        bm25_search("alpha", corpora[1])
        bm25_search("alpha", corpora[0])  # hit: x becomes most recently used
        bm25_search("alpha", corpora[2])  # evicts y, the least recently used
        assert counting_bm25.construction_count == 3
        assert len(bm25_mod._bm25_cache) == 2

        bm25_search("alpha", corpora[0])  # still cached
        assert counting_bm25.construction_count == 3
        bm25_search("alpha", corpora[1])  # was evicted: rebuilds
        assert counting_bm25.construction_count == 4

    def test_row_bound_evicts_but_keeps_the_newest_model(
        self, counting_bm25: type[_CountingBM25], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bm25_mod, "_BM25_CACHE_MAX_ROWS", 3)
        small = [make_entry("s1", "alpha"), make_entry("s2", "beta")]
        large = [make_entry(f"l{i}", f"alpha {i}") for i in range(5)]

        bm25_search("alpha", small)
        bm25_search("alpha", large)  # 7 rows > 3: small is evicted, large alone is kept

        assert [len(ids) for _, ids, _ in bm25_mod._bm25_cache.values()] == [5]
        bm25_search("beta", large)
        assert counting_bm25.construction_count == 2
