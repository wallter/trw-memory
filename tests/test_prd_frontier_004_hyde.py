"""PRD-CORE-272 FR04: HyDE tombstone rejects before all recall side effects."""

import pytest

from tests.test_hype_store import _FakeEmbedder
from trw_memory.client import MemoryClient


@pytest.mark.parametrize("value", ["hypothetical answer", 0, False, [], {}])
async def test_expansion_rejects_before_permission_retry_or_embedding(value):
    # An uninitialized instance proves validation requires no client state.
    client = object.__new__(MemoryClient)
    with pytest.raises(TypeError, match="HyDE is retired"):
        await client.recall("query", query_expansion=value)


@pytest.mark.parametrize("value", [None, "", "  "])
async def test_neutral_expansion_warns_and_embeds_original_once(tmp_path, monkeypatch, value):
    client = MemoryClient("default", mode="local", db_path=tmp_path / "hyde.db")
    embedder = _FakeEmbedder()
    monkeypatch.setattr(client, "_get_embedder", lambda: embedder)
    try:
        with pytest.warns(UserWarning, match="retired"):
            assert await client.recall("original query", query_expansion=value) == []
        assert embedder.calls == ["original query"]
    finally:
        await client.close()


def test_internal_expansion_parameter_removed():
    import inspect

    from trw_memory._client_recall import recall_impl

    assert "query_expansion" not in inspect.signature(recall_impl).parameters
