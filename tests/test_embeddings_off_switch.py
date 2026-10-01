"""EMBEDDER-COLD-START: ``MEMORY_EMBEDDINGS_ENABLED=false`` never loads the local model.

The first store write used to load the embedding model (15.8 s observed in a
seed store). With the switch off a store write and a recall run keyword-only
and the model is never constructed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import trw_memory._client_store as client_store
import trw_memory.embeddings as embeddings_pkg
from trw_memory.client import MemoryClient


def _recording_provider(built: list[str]) -> type:
    class _Recorder:
        def __init__(self, *, model_name: str, dim: int) -> None:
            built.append(model_name)

        def available(self) -> bool:
            return False  # the load is what is being counted, not the vectors

    return _Recorder


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, built: list[str]) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "mem_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setattr(embeddings_pkg, "LocalEmbeddingProvider", _recording_provider(built))
    monkeypatch.setattr(client_store, "embedding_has_consumer", lambda *_a: True)
    return MemoryClient(namespace="default", mode="local")


async def test_store_and_recall_with_embeddings_off_never_load_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEMORY_EMBEDDINGS_ENABLED", "false")
    built: list[str] = []
    client = _client(tmp_path, monkeypatch, built)

    stored = await client.store("the reranker floor is adaptive", detail="keeps at least five rows")
    results = await client.recall("reranker floor")

    assert built == []
    assert stored["status"] == "stored"
    assert any(r["memory_id"] == stored["memory_id"] for r in results)


async def test_store_with_embeddings_on_does_load_the_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: the same store write with the default switch reaches the loader."""
    monkeypatch.delenv("MEMORY_EMBEDDINGS_ENABLED", raising=False)
    built: list[str] = []
    client = _client(tmp_path, monkeypatch, built)

    await client.store("the reranker floor is adaptive", detail="keeps at least five rows")

    assert len(built) == 1


def test_resolve_embedder_reports_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.models.config import MemoryConfig
    from trw_memory.tools._embedder import resolve_embedder

    built: list[str] = []
    monkeypatch.setattr(embeddings_pkg, "LocalEmbeddingProvider", _recording_provider(built))

    answer = resolve_embedder(MemoryConfig(embeddings_enabled=False), surface="test")

    assert answer == {"status": "unavailable", "reason": "embeddings_disabled"}
    assert built == []


def test_embedder_status_reports_disabled() -> None:
    from trw_memory.models.config import MemoryConfig
    from trw_memory.tools._embedder import embedder_status

    block = embedder_status(MemoryConfig(embeddings_enabled=False))

    assert block["available"] is False
    assert block["reason"] == "embeddings_disabled"


def test_an_embedder_that_cannot_load_names_the_command_that_repairs_it(monkeypatch) -> None:
    """E2E-INC-140: ``embedder_error`` carried no ``fix``, so the warning told the operator nothing to run."""
    from trw_memory.models.config import MemoryConfig
    from trw_memory.tools import _embedder

    monkeypatch.setattr(_embedder, "get_local_embedder", lambda **_kw: None)
    monkeypatch.setattr(_embedder, "find_spec", lambda _name: None)
    config = MemoryConfig(embeddings_enabled=True)

    answer = _embedder.resolve_embedder(config, surface="test")
    block = _embedder.embedder_status(config)

    assert isinstance(answer, dict) and answer["reason"] == "embedder_error"
    assert "trw-memory[embeddings]" in str(answer["fix"]) and "--no-embeddings" in str(answer["fix"])
    assert block["reason"] == "embedder_error"
    assert block["fix"] == answer["fix"]
