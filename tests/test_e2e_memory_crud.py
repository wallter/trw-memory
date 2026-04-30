"""E2E CRUD and validation tests for trw-memory."""

from __future__ import annotations

import pytest

from trw_memory.client import MemoryClient
from trw_memory.exceptions import MemoryNotFoundError

from ._test_e2e_memory_support import client, client_ns_a, client_ns_b


class TestStoreCRUD:
    """Section 1 of E2E plan: core CRUD operations."""

    async def test_store_basic_entry(self, client: MemoryClient) -> None:
        """1.1 — Store a basic entry and verify the result dict shape."""
        result = await client.store(
            content="Always validate JWT tokens before accessing protected routes",
            tags=["security", "auth"],
            importance=0.8,
        )
        assert result["memory_id"].startswith("M-")
        assert result["status"] == "stored"
        assert result["namespace"] == "default"
        assert result["timestamp"]

    async def test_recall_basic_search(self, client: MemoryClient) -> None:
        """1.5 — Recall by keyword query returns relevant results."""
        await client.store(
            content="JWT tokens expire after 30 minutes",
            tags=["auth"],
            importance=0.8,
        )
        await client.store(
            content="Database indices improve query speed",
            tags=["perf"],
            importance=0.6,
        )
        results = await client.recall(query="authentication tokens")
        assert len(results) >= 1
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_recall_tag_filtering(self, client: MemoryClient) -> None:
        """1.6 — Recall with tag filter returns only matching entries."""
        await client.store(content="Auth entry about JWT", tags=["auth"], importance=0.7)
        await client.store(content="Perf entry about caching", tags=["perf"], importance=0.6)
        results = await client.recall(query="entry", tags=["auth"])
        assert len(results) >= 1
        for r in results:
            assert "auth" in r["tags"]

    async def test_forget_valid_entry(self, client: MemoryClient) -> None:
        """1.9 — Forget an existing entry successfully."""
        result = await client.store(content="temporary note", importance=0.3)
        memory_id = result["memory_id"]
        deleted = await client.forget(memory_id=memory_id)
        assert deleted["status"] == "deleted"
        assert deleted["memory_id"] == memory_id
        results = await client.recall(query="temporary note")
        found_ids = [entry["memory_id"] for entry in results]
        assert memory_id not in found_ids

    async def test_forget_wrong_namespace_raises(
        self,
        client_ns_a: MemoryClient,
        client_ns_b: MemoryClient,
    ) -> None:
        """1.11 — Forgetting an entry from the wrong namespace raises."""
        result = await client_ns_a.store(content="namespace-a data", importance=0.5)
        with pytest.raises(MemoryNotFoundError):
            await client_ns_b.forget(memory_id=result["memory_id"])


class TestValidationEdgeCases:
    """Section 1.4 + 10 of E2E plan: input validation."""

    async def test_empty_content_raises_value_error(self, client: MemoryClient) -> None:
        """1.4 — Empty content string is rejected by schema validation."""
        from trw_memory.exceptions import SchemaValidationError

        with pytest.raises((ValueError, SchemaValidationError)):
            await client.store(content="", importance=0.5)

    async def test_importance_out_of_range_raises(self, client: MemoryClient) -> None:
        """1.4 — Importance outside [0,1] is rejected by schema validation."""
        from trw_memory.exceptions import SchemaValidationError

        with pytest.raises((ValueError, SchemaValidationError)):
            await client.store(content="test", importance=1.5)

        with pytest.raises((ValueError, SchemaValidationError)):
            await client.store(content="test", importance=-0.1)

    async def test_recall_limit_below_one_raises(self, client: MemoryClient) -> None:
        """Recall with limit < 1 raises ValueError."""
        with pytest.raises(ValueError, match="limit must be >= 1"):
            await client.recall(query="test", limit=0)


class TestRecallEmptyResults:
    """Verify recall handles queries with no matches gracefully."""

    async def test_recall_no_matches_returns_empty(self, client: MemoryClient) -> None:
        """1.8 — Recall with no matching entries returns empty list."""
        results = await client.recall(query="quantum_computing_patterns_xyz_nonexistent")
        assert results == []


class TestForgetNonExistent:
    """Verify forget raises MemoryNotFoundError for missing entries."""

    async def test_forget_nonexistent_raises(self, client: MemoryClient) -> None:
        """1.10 — Forgetting a non-existent entry raises MemoryNotFoundError."""
        with pytest.raises(MemoryNotFoundError):
            await client.forget(memory_id="nonexistent-id-xyz")
