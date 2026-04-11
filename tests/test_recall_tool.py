"""Tests for memory_recall_impl token budget wiring (PRD-CORE-123 FR05).

Verifies that the trw-memory MCP tool layer correctly:
- Passes token_budget through to budget-fitting logic
- Returns tokens_used, tokens_budget, tokens_truncated metadata
- Validates invalid token_budget values
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trw_memory.retrieval.token_budget import estimate_entry_tokens


def _make_backend_with_entries(entries: list[dict[str, object]]) -> MagicMock:
    """Create a mock StorageBackend that returns the given entries."""
    backend = MagicMock()
    backend.list_entries.return_value = []
    backend.get_stored_embeddings.return_value = {}
    backend.search.return_value = [
        MagicMock(
            id=e.get("id", f"M-{i}"),
            content=str(e.get("content", "")),
            detail=str(e.get("detail", "")),
            tags=e.get("tags", []),
            impact=float(e.get("impact", 0.5)),
            to_dict=MagicMock(return_value=e),
        )
        for i, e in enumerate(entries)
    ]
    return backend


def _make_sized_entry(entry_id: str, word_count: int) -> dict[str, object]:
    """Create an entry dict with a known content size."""
    content = " ".join(f"word{i}" for i in range(word_count))
    return {
        "id": entry_id,
        "summary": f"Learning {entry_id}",
        "content": content,
        "detail": "",
        "tags": [],
        "impact": 0.5,
    }


class TestMemoryRecallImplTokenBudget:
    """FR05: memory_recall_impl accepts token_budget and returns metadata."""

    def test_token_budget_metadata_present(self, tmp_path: Path) -> None:
        """Response includes tokens_used, tokens_budget, tokens_truncated."""
        from trw_memory.tools.recall import memory_recall_impl

        backend = MagicMock()
        backend.list_entries.return_value = []
        backend.get_stored_embeddings.return_value = {}
        backend.search.return_value = []

        result = memory_recall_impl(
            query="test",
            namespace="default",
            backend=backend,
            token_budget=4000,
        )

        assert "tokens_used" in result
        assert "tokens_budget" in result
        assert "tokens_truncated" in result
        assert result["tokens_budget"] == 4000
        assert isinstance(result["tokens_used"], int)
        assert isinstance(result["tokens_truncated"], bool)

    def test_token_budget_none_returns_informational(self, tmp_path: Path) -> None:
        """token_budget=None still computes tokens_used for info."""
        from trw_memory.tools.recall import memory_recall_impl

        backend = MagicMock()
        backend.list_entries.return_value = []
        backend.get_stored_embeddings.return_value = {}
        backend.search.return_value = []

        result = memory_recall_impl(
            query="test",
            namespace="default",
            backend=backend,
            token_budget=None,
        )

        assert result["tokens_budget"] is None
        assert result["tokens_truncated"] is False
        assert result["tokens_used"] == 0  # No entries

    def test_token_budget_invalid_raises(self) -> None:
        """token_budget <= 0 raises ValueError (NFR03)."""
        from trw_memory.tools.recall import memory_recall_impl

        backend = MagicMock()
        backend.list_entries.return_value = []
        backend.get_stored_embeddings.return_value = {}
        backend.search.return_value = []

        with pytest.raises(ValueError, match="token_budget must be positive"):
            memory_recall_impl(
                query="test",
                namespace="default",
                backend=backend,
                token_budget=0,
            )

        with pytest.raises(ValueError, match="token_budget must be positive"):
            memory_recall_impl(
                query="test",
                namespace="default",
                backend=backend,
                token_budget=-5,
            )

    def test_token_budget_metadata_values_correct(self, tmp_path: Path) -> None:
        """tokens_used, tokens_truncated values are consistent with budget."""
        from trw_memory.tools.recall import memory_recall_impl

        # Create entries with known sizes
        entries = [_make_sized_entry(f"M-{i}", 50) for i in range(5)]

        # Compute expected cost of first entry
        first_cost = estimate_entry_tokens(entries[0])

        backend = MagicMock()
        backend.get_stored_embeddings.return_value = {}
        # Mock search to return entry-like objects
        mock_entries = []
        for e in entries:
            mock_entry = MagicMock()
            mock_entry.id = e["id"]
            mock_entry.content = e["content"]
            mock_entry.detail = e["detail"]
            mock_entry.tags = e["tags"]
            mock_entry.impact = e["impact"]
            mock_entry.to_dict.return_value = e
            mock_entries.append(mock_entry)
        backend.list_entries.return_value = []
        backend.search.return_value = mock_entries

        # Budget just fits first entry
        result = memory_recall_impl(
            query="test",
            namespace="default",
            backend=backend,
            token_budget=first_cost,
        )

        assert result["tokens_used"] <= first_cost or len(result["memories"]) == 1
        assert result["tokens_budget"] == first_cost
