"""Regression coverage for partial HyDE embedding failures."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from trw_memory._client_recall import recall_impl
from trw_memory.client import MemoryClient


@pytest.mark.asyncio
async def test_query_expansion_falls_back_to_expansion_vector() -> None:
    """A failed raw-query embedding must not discard a valid expansion."""

    class PartialEmbedder:
        def embed(self, text: str) -> list[float] | None:
            return None if text == "raw query" else [0.2, 0.4]

    client = MemoryClient(namespace="default", mode="local")
    captured: dict[str, object] = {}

    async def recall(*_args: object, **kwargs: object) -> list[object]:
        captured.update(kwargs)
        return []

    with (
        patch.object(client, "_get_embedder", return_value=PartialEmbedder()),
        patch.object(client, "_try_hybrid_recall", side_effect=recall),
    ):
        await recall_impl(client, "raw query", query_expansion="expanded answer")

    assert captured["query_embedding"] == [0.2, 0.4]
    await client.close()
