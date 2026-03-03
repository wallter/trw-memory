"""Tests for auto_recall decorator (FR06).

Covers:
- Happy path — query extracted, memories injected
- No query_from arg — empty list injected (fail-open)
- Backend error — empty list injected (fail-open)
- min_score filtering
- Positional recalled_memories arg — TypeError at decoration time
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from trw_memory.client import MemoryClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """Create a MemoryClient with local backend in tmp_path."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "deco_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestAutoRecallHappy:
    async def test_query_extracted_and_memories_injected(
        self, client: MemoryClient
    ) -> None:
        """Memories matching the query are injected as recalled_memories."""
        await client.store("pydantic strict mode required", tags=["pydantic"])

        @client.auto_recall(query_from="prompt", limit=5)
        async def handler(
            prompt: str, *, recalled_memories: list[dict[str, Any]] | None = None
        ) -> list[dict[str, Any]] | None:
            return recalled_memories

        result = await handler(prompt="pydantic strict")
        assert result is not None
        assert len(result) >= 1
        assert result[0]["content"] == "pydantic strict mode required"

    async def test_limit_respected(self, client: MemoryClient) -> None:
        """Only up to `limit` memories should be injected."""
        for i in range(5):
            await client.store(f"recall test entry {i}", importance=0.5)

        @client.auto_recall(query_from="q", limit=2)
        async def handler(
            q: str, *, recalled_memories: list[dict[str, Any]] | None = None
        ) -> list[dict[str, Any]] | None:
            return recalled_memories

        result = await handler(q="recall test entry")
        assert result is not None
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# Fail-open behaviour
# ---------------------------------------------------------------------------


class TestAutoRecallFailOpen:
    async def test_missing_query_from_injects_empty(
        self, client: MemoryClient
    ) -> None:
        """If the query_from kwarg is absent, inject empty list."""

        @client.auto_recall(query_from="missing_key")
        async def handler(
            x: int, *, recalled_memories: list[dict[str, Any]] | None = None
        ) -> list[dict[str, Any]] | None:
            return recalled_memories

        result = await handler(x=42)
        assert result == []

    async def test_backend_error_injects_empty(
        self, client: MemoryClient
    ) -> None:
        """If the backend raises, inject empty list (fail-open)."""

        @client.auto_recall(query_from="prompt")
        async def handler(
            prompt: str, *, recalled_memories: list[dict[str, Any]] | None = None
        ) -> list[dict[str, Any]] | None:
            return recalled_memories

        # Patch recall to raise
        with patch.object(
            client, "recall", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ):
            result = await handler(prompt="anything")
        assert result == []

    async def test_empty_query_injects_empty(self, client: MemoryClient) -> None:
        """If query_from value is empty string, inject empty list."""

        @client.auto_recall(query_from="prompt")
        async def handler(
            prompt: str = "", *, recalled_memories: list[dict[str, Any]] | None = None
        ) -> list[dict[str, Any]] | None:
            return recalled_memories

        result = await handler(prompt="")
        assert result == []


# ---------------------------------------------------------------------------
# min_score filtering
# ---------------------------------------------------------------------------


class TestAutoRecallMinScore:
    async def test_min_score_filters_low_results(
        self, client: MemoryClient
    ) -> None:
        """Results below min_score should be filtered out."""
        await client.store("low importance entry", importance=0.1)
        await client.store("high importance entry", importance=0.9)

        @client.auto_recall(query_from="q", min_score=0.5)
        async def handler(
            q: str, *, recalled_memories: list[dict[str, Any]] | None = None
        ) -> list[dict[str, Any]] | None:
            return recalled_memories

        result = await handler(q="importance entry")
        assert result is not None
        for r in result:
            assert r["score"] >= 0.5


# ---------------------------------------------------------------------------
# Positional recalled_memories arg — TypeError
# ---------------------------------------------------------------------------


class TestAutoRecallPositionalArg:
    def test_positional_recalled_memories_raises(
        self, client: MemoryClient
    ) -> None:
        """If the decorated function has recalled_memories as a required
        positional parameter, raise TypeError at decoration time."""
        with pytest.raises(TypeError, match="recalled_memories"):

            @client.auto_recall(query_from="q")
            async def handler(
                q: str, recalled_memories: list[dict[str, Any]]
            ) -> None:
                pass

    def test_keyword_only_recalled_memories_ok(
        self, client: MemoryClient
    ) -> None:
        """keyword-only recalled_memories should not raise."""

        @client.auto_recall(query_from="q")
        async def handler(
            q: str, *, recalled_memories: list[dict[str, Any]] | None = None
        ) -> None:
            pass

        # Should not raise
        assert handler is not None

    def test_optional_positional_recalled_memories_ok(
        self, client: MemoryClient
    ) -> None:
        """Positional recalled_memories with a default value should be allowed."""

        @client.auto_recall(query_from="q")
        async def handler(
            q: str, recalled_memories: list[dict[str, Any]] | None = None
        ) -> None:
            pass

        assert handler is not None


# ---------------------------------------------------------------------------
# Decorator preserves function metadata
# ---------------------------------------------------------------------------


class TestAutoRecallMetadata:
    def test_preserves_function_name(self, client: MemoryClient) -> None:
        @client.auto_recall(query_from="q")
        async def my_handler(
            q: str, *, recalled_memories: list[dict[str, Any]] | None = None
        ) -> None:
            """My docstring."""

        assert my_handler.__name__ == "my_handler"
        assert my_handler.__doc__ == "My docstring."
