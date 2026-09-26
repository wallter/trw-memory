"""C12 rc7: caller text costs the recall legs no more than the FTS leg allows.

``memory_recall``'s ``limit`` sized the scored candidate pool with no ceiling, and every leg but FTS
(BM25, lexical, the reranker, tiers, the SDK fallback) read the whole query, so both factors of
their per-row cost were the caller's.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from rank_bm25 import BM25Okapi

from trw_memory.client import MemoryClient
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.bm25 import bm25_search
from trw_memory.retrieval.lexical import MAX_QUERY_CHARS, MAX_QUERY_TERMS, bounded_query, tokenize_query
from trw_memory.retrieval.recall_policy import MAX_RECALL_LIMIT
from trw_memory.tools.recall import memory_recall_impl

_MANY_WORDS = " ".join(f"word{i}" for i in range(4 * MAX_QUERY_TERMS))


def test_recall_refuses_a_limit_past_the_ceiling_before_reading_a_row() -> None:
    backend = MagicMock()

    result = memory_recall_impl("q", "default", backend=backend, limit=MAX_RECALL_LIMIT + 1)

    assert result["status"] == "invalid"
    backend.list_entries.assert_not_called()
    backend.search.assert_not_called()


def test_recall_serves_trw_mcp_s_deepest_daemon_page() -> None:
    """trw-mcp's daemon store pages recall up to its DEFAULT_LIST_LIMIT (10,000); a refusal fails trw_recall."""
    backend = MagicMock(**{"list_entries.return_value": [], "search.return_value": []})

    assert "memories" in memory_recall_impl("q", "default", backend=backend, limit=MAX_RECALL_LIMIT)


def test_recall_reads_at_most_the_bounded_query() -> None:
    backend = MagicMock(**{"list_entries.return_value": [], "search.return_value": []})

    result = memory_recall_impl(_MANY_WORDS, "default", backend=backend)

    assert result["query"] == bounded_query(_MANY_WORDS) != _MANY_WORDS
    assert bounded_query("keep  these\nexact") == "keep  these\nexact"


async def test_sdk_recall_bounds_the_limit_and_the_query(client: MemoryClient, monkeypatch) -> None:
    seen: list[str] = []

    async def _hybrid(query: str, *_args: object, **_kwargs: object) -> list[object]:
        seen.append(query)
        return []

    monkeypatch.setattr(client, "_try_hybrid_recall", _hybrid)
    with pytest.raises(ValueError, match="limit must be >= 1"):
        await client.recall("q", limit=MAX_RECALL_LIMIT + 1)
    await client.recall(_MANY_WORDS)

    assert seen == [bounded_query(_MANY_WORDS)]


def test_the_lexical_leg_tokenizes_at_most_the_fts_bounds() -> None:
    assert len(tokenize_query(_MANY_WORDS)) == MAX_QUERY_TERMS
    assert sum(map(len, tokenize_query("a" * (50 * MAX_QUERY_CHARS)))) == MAX_QUERY_CHARS


def test_the_bm25_leg_scores_at_most_the_fts_bounds(monkeypatch) -> None:
    scored: list[list[str]] = []
    get_scores = BM25Okapi.get_scores
    monkeypatch.setattr(BM25Okapi, "get_scores", lambda self, query: scored.append(query) or get_scores(self, query))
    entries = [MemoryEntry(id="e1", content="word1 lives here"), MemoryEntry(id="e2", content="nothing")]

    bm25_search(_MANY_WORDS, entries)
    bm25_search("b" * (50 * MAX_QUERY_CHARS), entries)

    assert len(scored[0]) == MAX_QUERY_TERMS
    assert sum(map(len, scored[1])) <= MAX_QUERY_CHARS
