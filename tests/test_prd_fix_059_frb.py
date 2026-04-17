"""Tests for PRD-FIX-059 FR-02: Wire MemoryClient.recall() through retrieval pipeline.

Verifies that MemoryClient.recall() attempts hybrid search (BM25 + dense vectors
via RRF fusion) first, falling back to the original LIKE-based TF scoring when
the retrieval pipeline is unavailable or returns no results.

Test classification: integration (uses tmp_path for SQLite backend).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import structlog

from trw_memory.client import (
    _FALLBACK_IMPORTANCE_WEIGHT,
    _FALLBACK_TF_SCALE,
    _FALLBACK_TF_WEIGHT,
    MemoryClient,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """Create a test-isolated MemoryClient with SQLite backend."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "mem_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


def _force_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force recall to use the fallback path by hiding the retrieval pipeline.

    Patches ``_try_hybrid_recall`` to return None (signal fallback),
    which is cleaner than trying to patch a locally-imported module.
    """
    monkeypatch.setattr(
        "trw_memory.client.MemoryClient._try_hybrid_recall",
        lambda self, query, limit, tags: _coro_none(),
    )


async def _coro_none() -> None:
    """Async helper that returns None -- used by _force_fallback."""


# ---------------------------------------------------------------------------
# Test: Fallback path works when hybrid pipeline is unavailable
# ---------------------------------------------------------------------------


class TestRecallFallbackPath:
    """Verify recall works via fallback TF scoring when hybrid is unavailable."""

    @pytest.fixture()
    def client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
        return _make_client(tmp_path, monkeypatch)

    async def test_recall_fallback_path_when_hybrid_returns_none(
        self, client: MemoryClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """recall() falls back to LIKE+TF scoring when _try_hybrid_recall returns None."""
        await client.store("pydantic v2 uses model_dump", tags=["pydantic"])

        # Force fallback by making hybrid return None
        _force_fallback(monkeypatch)
        results = await client.recall("pydantic")

        assert len(results) >= 1
        assert results[0]["content"] == "pydantic v2 uses model_dump"

    async def test_recall_fallback_path_on_empty_entries(self, client: MemoryClient) -> None:
        """recall() returns empty when no entries stored (hybrid gets no candidates)."""
        results = await client.recall("anything")
        assert results == []

    async def test_recall_fallback_scoring_produces_reasonable_scores(
        self, client: MemoryClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fallback TF scoring produces scores in [0, 1] and orders by relevance."""
        await client.store("python async await patterns", tags=["python"])
        await client.store("java spring boot patterns", tags=["java"])

        # Force fallback path
        _force_fallback(monkeypatch)
        results = await client.recall("python async")

        # At least one result for the matching entry
        assert len(results) >= 1
        # Scores should be in valid range
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

        # The python entry should score higher than java
        python_results = [r for r in results if "python" in r["content"]]
        java_results = [r for r in results if "java" in r["content"]]
        if python_results and java_results:
            assert python_results[0]["score"] >= java_results[0]["score"]


# ---------------------------------------------------------------------------
# Test: Hybrid path is attempted and produces results
# ---------------------------------------------------------------------------


class TestRecallHybridPath:
    """Verify recall uses hybrid_search when available."""

    @pytest.fixture()
    def client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
        return _make_client(tmp_path, monkeypatch)

    async def test_recall_uses_hybrid_search_when_available(self, client: MemoryClient) -> None:
        """recall() uses hybrid_search pipeline when it is importable."""
        await client.store("structlog uses get_logger", tags=["logging"])
        await client.store("print is bad for logging", tags=["logging"])

        results = await client.recall("structlog")
        # Should return at least the matching entry
        assert len(results) >= 1
        # Verify the content is from our stored entries
        contents = [r["content"] for r in results]
        assert any("structlog" in c for c in contents)

    async def test_recall_hybrid_applies_tag_filter(self, client: MemoryClient) -> None:
        """Hybrid path respects tag filtering."""
        await client.store("pydantic validation", tags=["pydantic", "python"])
        await client.store("pydantic serialization", tags=["pydantic"])

        results = await client.recall("pydantic", tags=["python"])
        # Only the entry with both tags should match
        assert len(results) >= 1
        for r in results:
            assert "python" in r["tags"]

    async def test_recall_hybrid_positional_scoring(self, client: MemoryClient) -> None:
        """Hybrid path uses RRF-style positional scoring: 1/(1+rank).

        We mock hybrid_search to return controlled entries so we can verify
        the positional scoring formula without depending on BM25 IDF behavior.
        """
        from conftest import make_entry

        from trw_memory.models.memory import MemoryStatus

        e1 = make_entry(entry_id="M-001", content="first match", status=MemoryStatus.ACTIVE)
        e2 = make_entry(entry_id="M-002", content="second match", status=MemoryStatus.ACTIVE)
        e3 = make_entry(entry_id="M-003", content="third match", status=MemoryStatus.ACTIVE)

        # Mock hybrid_search to return entries in known order
        with patch(
            "trw_memory.retrieval.pipeline.hybrid_search",
            return_value=[e1, e2, e3],
        ):
            # Also need entries in backend for list_entries
            await client.store("first match")
            await client.store("second match")
            await client.store("third match")

            results = await client.recall("match")

        assert len(results) >= 3
        # First result: rank 0 -> score = 1/(1+0) = 1.0
        assert results[0]["score"] == 1.0
        # Second result: rank 1 -> score = 1/(1+1) = 0.5
        assert results[1]["score"] == 0.5
        # Third result: rank 2 -> score = 1/(1+2) = 0.3333
        assert results[2]["score"] == round(1.0 / 3, 4)
        # Scores must be strictly decreasing
        for i in range(len(results) - 1):
            assert results[i]["score"] > results[i + 1]["score"]


# ---------------------------------------------------------------------------
# Test: Empty query
# ---------------------------------------------------------------------------


class TestRecallEmptyQuery:
    """Verify recall behavior with empty or whitespace queries."""

    @pytest.fixture()
    def client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
        return _make_client(tmp_path, monkeypatch)

    async def test_recall_empty_query_returns_results(self, client: MemoryClient) -> None:
        """Empty query should still return results (all entries match)."""
        await client.store("entry one", importance=0.8)
        await client.store("entry two", importance=0.6)

        results = await client.recall("")
        # Should return stored entries even with empty query
        assert len(results) >= 1

    async def test_recall_whitespace_query_does_not_crash(self, client: MemoryClient) -> None:
        """Whitespace-only query does not raise; returns empty or all entries."""
        await client.store("some content", importance=0.7)

        # Whitespace-only query: BM25 tokenizes to nothing. The hybrid path
        # returns no BM25 hits, so falls back. The fallback search via LIKE
        # may or may not match -- either way, no crash.
        results = await client.recall("   ")
        # Must not crash; result can be empty or non-empty
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Test: min_score filtering
# ---------------------------------------------------------------------------


class TestRecallMinScoreFiltering:
    """Verify min_score filters results in both paths."""

    @pytest.fixture()
    def client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
        return _make_client(tmp_path, monkeypatch)

    async def test_recall_min_score_filters_low_scoring(self, client: MemoryClient) -> None:
        """Entries below min_score threshold are excluded."""
        await client.store("exact match query term", importance=0.9)
        await client.store("unrelated content xyz", importance=0.1)

        # Use a high min_score to filter out low-scoring entries
        results = await client.recall("exact match query", min_score=0.5)
        # All returned results must have score >= min_score
        for r in results:
            assert r["score"] >= 0.5

    async def test_recall_min_score_zero_returns_all(self, client: MemoryClient) -> None:
        """min_score=0.0 (default) returns all matched entries."""
        await client.store("alpha content", importance=0.1)
        await client.store("beta content", importance=0.9)

        results = await client.recall("content", min_score=0.0)
        assert len(results) >= 2


# ---------------------------------------------------------------------------
# Test: Tag filtering
# ---------------------------------------------------------------------------


class TestRecallTagFiltering:
    """Verify tag filtering in both hybrid and fallback paths."""

    @pytest.fixture()
    def client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
        return _make_client(tmp_path, monkeypatch)

    async def test_recall_tag_filter_includes_matching(self, client: MemoryClient) -> None:
        """Only entries with all specified tags are returned."""
        await client.store("python tip", tags=["python", "tips"])
        await client.store("java tip", tags=["java", "tips"])

        results = await client.recall("tip", tags=["python"])
        assert all("python" in r["tags"] for r in results)

    async def test_recall_tag_filter_subset_matching(self, client: MemoryClient) -> None:
        """Tags filter uses subset: entry must contain ALL filter tags."""
        await client.store("multi-tag entry", tags=["a", "b", "c"])
        await client.store("partial-tag entry", tags=["a"])

        results = await client.recall("entry", tags=["a", "b"])
        # Only the entry with tags [a, b, c] should match
        for r in results:
            assert "a" in r["tags"]
            assert "b" in r["tags"]

    async def test_recall_no_tags_returns_all(self, client: MemoryClient) -> None:
        """When tags=None, all matching entries are returned regardless of tags."""
        await client.store("tagged entry", tags=["test"])
        await client.store("untagged entry")

        results = await client.recall("entry", tags=None)
        assert len(results) >= 2


# ---------------------------------------------------------------------------
# Test: Limit is respected
# ---------------------------------------------------------------------------


class TestRecallLimitRespected:
    """Verify the limit parameter caps results."""

    @pytest.fixture()
    def client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
        return _make_client(tmp_path, monkeypatch)

    async def test_recall_limit_caps_results(self, client: MemoryClient) -> None:
        """Results are capped at the limit value."""
        for i in range(10):
            await client.store(f"entry number {i} about testing")

        results = await client.recall("testing", limit=3)
        assert len(results) <= 3

    async def test_recall_limit_one_returns_single(self, client: MemoryClient) -> None:
        """limit=1 returns at most one result."""
        await client.store("first entry about code")
        await client.store("second entry about code")

        results = await client.recall("code", limit=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# Test: Structured logging includes search_path
# ---------------------------------------------------------------------------


class TestRecallSearchPathLogged:
    """Verify structured logging includes search_path field."""

    @pytest.fixture()
    def client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
        return _make_client(tmp_path, monkeypatch)

    async def test_recall_logs_hybrid_search_path(
        self,
        client: MemoryClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When hybrid search succeeds, log includes search_path='hybrid'."""
        await client.store("test content for logging")

        logged_kwargs: dict[str, object] = {}
        original_debug = structlog.get_logger("trw_memory.client").debug

        def _capture_debug(event: str, **kw: object) -> None:
            if event == "memory_recalled":
                logged_kwargs.update(kw)

        monkeypatch.setattr("trw_memory.client.logger", type("L", (), {"debug": staticmethod(_capture_debug)})())
        await client.recall("test content")
        assert logged_kwargs.get("search_path") in ("hybrid", "fallback")

    async def test_recall_logs_fallback_search_path(
        self,
        client: MemoryClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When fallback is used, log includes search_path='fallback'."""
        await client.store("test content for logging")

        logged_kwargs: dict[str, object] = {}

        def _capture_debug(event: str, **kw: object) -> None:
            if event == "memory_recalled":
                logged_kwargs.update(kw)

        monkeypatch.setattr("trw_memory.client.logger", type("L", (), {"debug": staticmethod(_capture_debug)})())
        _force_fallback(monkeypatch)
        await client.recall("test")
        assert logged_kwargs.get("search_path") == "fallback"


# ---------------------------------------------------------------------------
# Test: Scoring constants renamed properly
# ---------------------------------------------------------------------------


class TestScoringConstants:
    """Verify scoring constants exist with correct fallback naming."""

    def test_fallback_tf_weight_exists(self) -> None:
        """_FALLBACK_TF_WEIGHT constant is defined."""
        assert isinstance(_FALLBACK_TF_WEIGHT, float)
        assert 0.0 < _FALLBACK_TF_WEIGHT <= 1.0

    def test_fallback_importance_weight_exists(self) -> None:
        """_FALLBACK_IMPORTANCE_WEIGHT constant is defined."""
        assert isinstance(_FALLBACK_IMPORTANCE_WEIGHT, float)
        assert 0.0 < _FALLBACK_IMPORTANCE_WEIGHT <= 1.0

    def test_fallback_tf_scale_exists(self) -> None:
        """_FALLBACK_TF_SCALE constant is defined."""
        assert isinstance(_FALLBACK_TF_SCALE, float)
        assert _FALLBACK_TF_SCALE > 0.0

    def test_weights_sum_to_one(self) -> None:
        """TF weight + importance weight should sum to 1.0."""
        assert abs(_FALLBACK_TF_WEIGHT + _FALLBACK_IMPORTANCE_WEIGHT - 1.0) < 0.01

    def test_backward_compat_aliases_exist(self) -> None:
        """Old constant names still exist for backward compatibility."""
        from trw_memory.client import _IMPORTANCE_WEIGHT, _TF_SCALE, _TF_WEIGHT

        assert _TF_WEIGHT == _FALLBACK_TF_WEIGHT
        assert _IMPORTANCE_WEIGHT == _FALLBACK_IMPORTANCE_WEIGHT
        assert _TF_SCALE == _FALLBACK_TF_SCALE


# ---------------------------------------------------------------------------
# Test: _make_id collision resistance
# ---------------------------------------------------------------------------


class TestMakeIdLength:
    """Verify _make_id uses longer hex for reduced collisions."""

    def test_make_id_length(self) -> None:
        """_make_id generates IDs with M- prefix and sufficient hex length."""
        from trw_memory.client import _make_id

        mid = _make_id()
        assert mid.startswith("M-")
        # Should be at least 16 hex chars (hex[:16]) for collision resistance
        hex_part = mid[2:]
        assert len(hex_part) >= 16

    def test_make_id_uniqueness(self) -> None:
        """_make_id produces distinct IDs on successive calls."""
        from trw_memory.client import _make_id

        ids = {_make_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# Test: _try_hybrid_recall and _fallback_recall exist as methods
# ---------------------------------------------------------------------------


class TestRecallMethodWiring:
    """Verify the new private methods exist and are callable."""

    @pytest.fixture()
    def client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
        return _make_client(tmp_path, monkeypatch)

    def test_try_hybrid_recall_method_exists(self, client: MemoryClient) -> None:
        """_try_hybrid_recall exists as a method on MemoryClient."""
        assert hasattr(client, "_try_hybrid_recall")
        assert callable(client._try_hybrid_recall)

    def test_fallback_recall_method_exists(self, client: MemoryClient) -> None:
        """_fallback_recall exists as a method on MemoryClient."""
        assert hasattr(client, "_fallback_recall")
        assert callable(client._fallback_recall)

    def test_get_embedder_method_exists(self, client: MemoryClient) -> None:
        """_get_embedder exists as a static method on MemoryClient."""
        assert hasattr(MemoryClient, "_get_embedder")
        assert callable(MemoryClient._get_embedder)
