"""Release-verify P1-B: a fresh ``[embeddings]`` install must not break the SDK.

Runtime loads never download (PLAN W40), so on an empty model cache
``get_local_embedder`` raises ``ModelNotCachedError``. The daemon tools already
degrade to keyword-only; ``MemoryClient`` must do the same instead of failing
every ``store()`` and ``recall()``.
"""

from __future__ import annotations

import pytest
import structlog.testing

import trw_memory._client_store as client_store
import trw_memory.embeddings as embeddings_pkg
from trw_memory.client import MemoryClient
from trw_memory.exceptions import ModelNotCachedError, RemoteCodeNotPermittedError


def _refusing(error: Exception, calls: list[int]):
    def _load(**_kw: object) -> None:
        calls.append(1)
        raise error

    return _load


async def test_an_uncached_model_degrades_store_and_recall_to_keyword_only(
    memory_client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.embeddings.local import FETCH_COMMAND

    calls: list[int] = []
    monkeypatch.setattr(
        embeddings_pkg,
        "get_local_embedder",
        _refusing(ModelNotCachedError(f"not cached. Fetch it: {FETCH_COMMAND}"), calls),
    )
    monkeypatch.setattr(client_store, "embedding_has_consumer", lambda *_a: True)

    with structlog.testing.capture_logs() as logs:
        stored = await memory_client.store("the reranker floor is adaptive", detail="keeps at least five rows")
        results = await memory_client.recall("reranker floor")

    assert stored["status"] == "stored"
    assert any(r["memory_id"] == stored["memory_id"] for r in results)
    assert len(calls) == 1, "the refusal is cached for the client lifetime"
    refusals = [log for log in logs if log["event"] == "embedder_refused_keyword_only"]
    assert refusals and refusals[0]["surface"] == "sdk"
    assert "fetch_models()" in refusals[0]["reason"]


async def test_a_model_that_needs_remote_code_still_raises(
    memory_client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        embeddings_pkg, "get_local_embedder", _refusing(RemoteCodeNotPermittedError("needs remote code"), [])
    )

    with pytest.raises(RemoteCodeNotPermittedError):
        await memory_client.recall("anything")
