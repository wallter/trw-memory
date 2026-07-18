"""Integration tests for MemoryClient.search_fts() and MemoryClient.store_many().

search_fts() returns a list of MemoryResultDict (TypedDict), so results are
accessed via dict keys (r["content"]), not attribute access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.client import MemoryClient
from trw_memory.exceptions import AuthorizationError
from trw_memory.models.memory import Assertion, AssertionType


@pytest.fixture()
async def client(tmp_path: Path) -> MemoryClient:
    return MemoryClient("default", db_path=tmp_path / "test.db")


class TestClientSearchFts:
    async def test_search_fts_finds_stored_entry(self, client: MemoryClient) -> None:
        await client.store("quantum entanglement physics")
        results = await client.search_fts("quantum")
        assert len(results) >= 1
        assert any("quantum" in r["content"].lower() for r in results)

    async def test_search_fts_namespace_isolated(self, tmp_path: Path) -> None:
        c1 = MemoryClient("project:alpha", db_path=tmp_path / "ns.db")
        c2 = MemoryClient("project:beta", db_path=tmp_path / "ns.db")
        await c1.store("alpha only content here")
        await c2.store("beta only content here")
        r1 = await c1.search_fts("alpha")
        r2 = await c1.search_fts("beta")
        assert len(r1) >= 1
        assert len(r2) == 0  # beta content not in alpha namespace

    async def test_search_fts_empty_query_returns_empty(self, client: MemoryClient) -> None:
        await client.store("some content")
        results = await client.search_fts("")
        assert results == []

    async def test_search_fts_no_match_returns_empty(self, client: MemoryClient) -> None:
        await client.store("hello world")
        results = await client.search_fts("zqxvmpw")
        assert results == []

    async def test_search_fts_requires_read_permission(self, client: MemoryClient) -> None:
        client._config.rbac_enabled = True
        client._config.namespace_roles = {"default": "writer"}

        with pytest.raises(AuthorizationError, match="search_fts permission"):
            await client.search_fts("blocked")


class TestClientStoreMany:
    async def test_store_many_returns_count(self, client: MemoryClient) -> None:
        entries: list[dict[str, object]] = [{"content": f"bulk entry {i}"} for i in range(10)]
        count = await client.store_many(entries)
        assert count == 10

    async def test_store_many_empty_returns_zero(self, client: MemoryClient) -> None:
        assert await client.store_many([]) == 0

    async def test_store_many_entries_searchable_via_fts(self, client: MemoryClient) -> None:
        entries: list[dict[str, object]] = [{"content": f"enterprise memory item {i}"} for i in range(5)]
        await client.store_many(entries)
        results = await client.search_fts("enterprise")
        assert len(results) >= 1

    async def test_store_many_honors_optional_fields(self, client: MemoryClient) -> None:
        assertion = Assertion(type=AssertionType.GLOB_EXISTS, target="src/**/*.py")
        entries: list[dict[str, object]] = [
            {
                "content": "tagged bulk record",
                "tags": ["alpha"],
                "importance": 0.9,
                "detail": "extra detail",
                "evidence": ["src/example.py:10-20"],
                "expires": "when migration ships",
                "assertions": [assertion],
                "entry_id": "M-store-many-fields",
            }
        ]
        count = await client.store_many(entries)
        assert count == 1
        results = await client.search_fts("tagged")
        assert len(results) >= 1
        assert results[0]["importance"] == pytest.approx(0.9)
        stored = client._get_backend().get("M-store-many-fields")
        assert stored is not None
        assert stored.evidence == ["src/example.py:10-20"]
        assert stored.expires == "when migration ships"
        assert stored.assertions == [assertion]
