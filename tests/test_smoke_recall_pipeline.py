"""Smoke tests for the recall pipeline — end-to-end from store through budget-filtered recall.

Verifies the complete recall flow: store entries → hybrid search → utility ranking →
token budget filtering → metadata computation. Uses real SQLiteBackend and MemoryClient
(no mocks) to catch integration issues at the storage/retrieval/budgeting boundaries.

These tests form the foundation for cross-package integration tests that verify
trw-mcp's trw_recall() calls trw-memory's recall pipeline correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.client import MemoryClient
from trw_memory.retrieval.token_budget import estimate_entry_tokens, estimate_tokens

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """Isolated MemoryClient backed by SQLite in tmp_path."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "smoke_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


@pytest.fixture()
async def client_with_varied_entries(client: MemoryClient) -> MemoryClient:
    """Client pre-loaded with entries of varying sizes for budget testing."""
    entries = [
        {
            "content": "Short learning about imports",
            "tags": ["python"],
            "importance": 0.9,
        },
        {
            "content": (
                "Medium-length learning about database connection pooling. "
                "SQLAlchemy provides connection pooling via the Engine. The default "
                "pool implementation uses QueuePool which maintains a fixed set of "
                "connections and recycles them. For testing, use StaticPool which "
                "reuses a single connection across all sessions."
            ),
            "tags": ["database", "sqlalchemy", "testing"],
            "importance": 0.8,
        },
        {
            "content": (
                "Long detailed learning about the TRW ceremony model. "
                "The ceremony system enforces a 6-phase execution model: "
                "RESEARCH → PLAN → IMPLEMENT → VALIDATE → REVIEW → DELIVER. "
                "Each phase has exit criteria that must be met before advancing. "
                "Phase reversion is allowed when new information invalidates "
                "assumptions made in earlier phases. The ceremony nudge system "
                "monitors tool usage and injects reminders when agents skip "
                "required ceremony steps. Nudge frequency is configurable per "
                "step and per tier (MINIMAL, STANDARD, COMPREHENSIVE). "
                "The system tracks nudge responsiveness and recall pull rate "
                "for measuring ceremony effectiveness."
            ),
            "tags": ["ceremony", "framework", "nudges"],
            "importance": 0.7,
        },
        {
            "content": "Always use structlog.get_logger(__name__) for logging",
            "tags": ["logging", "convention"],
            "importance": 0.6,
        },
        {
            "content": (
                "Pydantic v2 migration notes: use_enum_values=True is required on "
                "all model configs for YAML round-trip serialization. Without it "
                "enum fields serialize as member objects rather than their string "
                "values which causes the YAML writer to produce invalid output. "
                "Additionally populate_by_name=True is required when using Field "
                "with alias parameters."
            ),
            "tags": ["pydantic", "migration", "gotcha"],
            "importance": 0.75,
        },
    ]
    for entry in entries:
        await client.store(**entry)
    return client


# ===================================================================
# 1. Recall Pipeline Smoke Tests
# ===================================================================


class TestRecallPipelineBasics:
    """Verify the recall pipeline returns correctly ranked results."""

    async def test_recall_returns_results_sorted_by_score(self, client_with_varied_entries: MemoryClient) -> None:
        """Results come back sorted by relevance score descending."""
        results = await client_with_varied_entries.recall(query="database pooling")
        assert len(results) >= 1
        scores = [float(r.get("score", 0)) for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_recall_broad_query_returns_multiple(self, client_with_varied_entries: MemoryClient) -> None:
        """A broad query matching multiple entries returns several results."""
        results = await client_with_varied_entries.recall(query="learning about", limit=100)
        assert len(results) >= 2

    async def test_recall_tag_filter_narrows_results(self, client_with_varied_entries: MemoryClient) -> None:
        """Tag filter restricts results to matching entries."""
        results = await client_with_varied_entries.recall(query="learning", tags=["pydantic"])
        for r in results:
            assert "pydantic" in r["tags"]


# ===================================================================
# 2. Token Budget Integration
# ===================================================================


class TestTokenBudgetIntegration:
    """Verify token budgeting works end-to-end through the recall pipeline."""

    async def test_token_budget_reduces_result_count(self, client_with_varied_entries: MemoryClient) -> None:
        """A tight token budget returns fewer results than no budget."""
        all_results = await client_with_varied_entries.recall(query="learning about", limit=100)
        budgeted_results = await client_with_varied_entries.recall(query="learning about", limit=100, token_budget=100)
        assert len(budgeted_results) < len(all_results)

    async def test_token_budget_large_returns_all_matches(self, client_with_varied_entries: MemoryClient) -> None:
        """A large budget returns all matching results (no truncation)."""
        # First get baseline without budget
        baseline = await client_with_varied_entries.recall(query="learning about", limit=100)
        # Same query with very large budget should return same count
        budgeted = await client_with_varied_entries.recall(query="learning about", limit=100, token_budget=100_000)
        assert len(budgeted) == len(baseline)

    async def test_token_budget_minimum_one_guarantee(self, client_with_varied_entries: MemoryClient) -> None:
        """Even a tiny budget returns at least 1 result (minimum-one guarantee)."""
        results = await client_with_varied_entries.recall(query="learning about", limit=100, token_budget=1)
        assert len(results) >= 1

    async def test_token_budget_invalid_raises_value_error(self, client_with_varied_entries: MemoryClient) -> None:
        """Zero or negative token_budget raises ValueError."""
        with pytest.raises(ValueError, match="token_budget must be positive"):
            await client_with_varied_entries.recall(query="test", token_budget=0)

        with pytest.raises(ValueError, match="token_budget must be positive"):
            await client_with_varied_entries.recall(query="test", token_budget=-10)

    async def test_token_budget_none_unchanged_behavior(self, client_with_varied_entries: MemoryClient) -> None:
        """token_budget=None produces same results as no budget parameter."""
        results_default = await client_with_varied_entries.recall(query="learning about", limit=100)
        results_none = await client_with_varied_entries.recall(query="learning about", limit=100, token_budget=None)
        # Same count and same IDs
        ids_default = {r["memory_id"] for r in results_default}
        ids_none = {r["memory_id"] for r in results_none}
        assert ids_default == ids_none


# ===================================================================
# 3. Token Estimation Consistency
# ===================================================================


class TestTokenEstimationConsistency:
    """Verify token estimation is consistent across different text types."""

    def test_estimate_tokens_monotonic_with_length(self) -> None:
        """Longer text produces higher token estimates."""
        short = "hello world"
        medium = "the quick brown fox jumps over the lazy dog"
        long_text = " ".join(f"word{i}" for i in range(100))
        assert estimate_tokens(short) < estimate_tokens(medium) < estimate_tokens(long_text)

    def test_estimate_entry_tokens_includes_all_fields(self) -> None:
        """Entry estimation combines content + detail + tags + overhead."""
        entry_minimal: dict[str, object] = {"content": "hello", "detail": "", "tags": []}
        entry_rich: dict[str, object] = {
            "content": "hello world",
            "detail": "additional context here",
            "tags": ["tag1", "tag2", "tag3"],
        }
        assert estimate_entry_tokens(entry_minimal) < estimate_entry_tokens(entry_rich)

    def test_estimate_entry_tokens_with_real_learning(self) -> None:
        """Estimation on a realistic learning entry produces reasonable result."""
        entry: dict[str, object] = {
            "content": (
                "Pydantic v2 requires use_enum_values=True on all model configs "
                "for YAML round-trip serialization to work correctly"
            ),
            "detail": (
                "Without it enum fields serialize as enum member objects rather "
                "than their string values which causes the YAML writer to produce "
                "invalid output that cannot be loaded back"
            ),
            "tags": ["pydantic", "gotcha", "yaml", "serialization"],
        }
        tokens = estimate_entry_tokens(entry)
        # Realistic range: 20 (overhead) + ~50 tokens for content/detail/tags
        assert 40 < tokens < 100


# ===================================================================
# 4. Store → Recall → Token Budget Round-Trip
# ===================================================================


class TestStoreRecallBudgetRoundTrip:
    """Verify the complete store → recall → budget pipeline works atomically."""

    async def test_store_then_recall_with_budget_returns_stored_entry(self, client: MemoryClient) -> None:
        """A single stored entry is retrievable with any token budget."""
        await client.store(
            content="Test entry for round-trip verification",
            tags=["test"],
            importance=0.5,
        )
        results = await client.recall(query="round-trip", token_budget=50)
        assert len(results) >= 1
        assert "round-trip" in results[0]["content"]

    async def test_store_many_recall_budget_preserves_ranking(self, client: MemoryClient) -> None:
        """Budget filtering preserves the score-based ranking order."""
        for i in range(10):
            await client.store(
                content=f"Entry number {i} about testing patterns",
                tags=["test"],
                importance=0.5 + (i * 0.05),
            )
        results = await client.recall(query="testing patterns", token_budget=500)
        scores = [float(r.get("score", 0)) for r in results]
        assert scores == sorted(scores, reverse=True)
