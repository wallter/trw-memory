"""Tests for HyDE (Hypothetical Document Embeddings) via query_expansion parameter.

Frontier-004: MemoryClient.recall(query, *, query_expansion=None) accepts an
optional hypothetical document. When provided, the dense embedding is computed
from query_expansion (the hypothetical answer) while BM25 still runs on the
original query. This improves recall on abstract or under-specified queries.

The public API is parameter-only: no LLM is embedded in the memory substrate.
Callers with LLM access generate the expansion; callers without just omit it.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestQueryExpansionParameter:
    """query_expansion is threaded through to the dense embedding step."""

    @pytest.mark.asyncio
    async def test_recall_accepts_query_expansion(self) -> None:
        """recall() must accept query_expansion without raising."""
        from trw_memory.client import MemoryClient

        client = MemoryClient(namespace="default", mode="local")
        await client.store(content="Paris is the capital of France", importance=0.8)
        # Should not raise regardless of whether embedder is available
        results = await client.recall(
            "What is the capital of France?",
            limit=5,
            query_expansion="The capital of France is Paris, a major European city.",
        )
        assert isinstance(results, list)
        await client.close()

    @pytest.mark.asyncio
    async def test_query_expansion_uses_expansion_for_dense_embedding(self) -> None:
        """Dense embedding must use query_expansion text, not the raw query."""
        from trw_memory._client_recall import recall_impl
        from trw_memory.client import MemoryClient

        expansion = "This is the hypothetical answer about memory retrieval."
        embedded_texts: list[str] = []

        class _CapturingEmbedder:
            def embed(self, text: str) -> list[float]:
                embedded_texts.append(text)
                return [0.1] * 384

        client = MemoryClient(namespace="default", mode="local")
        await client.store(content="some memory", importance=0.5)

        with patch.object(client, "_get_embedder", return_value=_CapturingEmbedder()):
            try:
                await recall_impl(
                    client,
                    "original query",
                    limit=5,
                    query_expansion=expansion,
                )
            except Exception:
                pass

        # The dense embedding must have been called with the expansion text
        assert expansion in embedded_texts, (
            f"Expected expansion text to be embedded, got: {embedded_texts}"
        )
        await client.close()

    @pytest.mark.asyncio
    async def test_no_query_expansion_uses_original_query(self) -> None:
        """When query_expansion is None, dense embedding uses the original query."""
        from trw_memory._client_recall import recall_impl
        from trw_memory.client import MemoryClient

        embedded_texts: list[str] = []

        class _CapturingEmbedder:
            def embed(self, text: str) -> list[float]:
                embedded_texts.append(text)
                return [0.1] * 384

        client = MemoryClient(namespace="default", mode="local")
        await client.store(content="some memory", importance=0.5)

        with patch.object(client, "_get_embedder", return_value=_CapturingEmbedder()):
            try:
                await recall_impl(client, "original query", limit=5)
            except Exception:
                pass

        assert "original query" in embedded_texts
        await client.close()

    @pytest.mark.asyncio
    async def test_empty_query_expansion_falls_back_to_original(self) -> None:
        """Empty or whitespace-only query_expansion must not override the query."""
        from trw_memory._client_recall import recall_impl
        from trw_memory.client import MemoryClient

        embedded_texts: list[str] = []

        class _CapturingEmbedder:
            def embed(self, text: str) -> list[float]:
                embedded_texts.append(text)
                return [0.1] * 384

        client = MemoryClient(namespace="default", mode="local")
        await client.store(content="some memory", importance=0.5)

        with patch.object(client, "_get_embedder", return_value=_CapturingEmbedder()):
            try:
                await recall_impl(client, "original query", limit=5, query_expansion="  ")
            except Exception:
                pass

        assert "original query" in embedded_texts
        assert "  " not in embedded_texts
        await client.close()

    @pytest.mark.asyncio
    async def test_query_expansion_in_client_recall_signature(self) -> None:
        """MemoryClient.recall() must expose query_expansion as a kwarg."""
        import inspect
        from trw_memory.client import MemoryClient

        sig = inspect.signature(MemoryClient.recall)
        assert "query_expansion" in sig.parameters

    def test_recall_impl_has_query_expansion_param(self) -> None:
        """recall_impl() must expose query_expansion as a kwarg."""
        import inspect
        from trw_memory._client_recall import recall_impl

        sig = inspect.signature(recall_impl)
        assert "query_expansion" in sig.parameters
        param = sig.parameters["query_expansion"]
        assert param.default is None
