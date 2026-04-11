"""Runtime wiring tests for vector persistence and hybrid recall."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import MemoryClient
from trw_memory.exceptions import StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.tools.store import memory_store_impl


class _StubEmbedder:
    def embed(self, text: str) -> list[float] | None:
        return [float(len(text)), 1.0, 0.5]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    def available(self) -> bool:
        return True

    def dim(self) -> int:
        return 3


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


async def test_client_store_upserts_vector_when_embedder_available(client: MemoryClient) -> None:
    """MemoryClient.store persists a vector alongside the entry when possible."""
    real_backend = client._backend
    assert real_backend is not None
    real_backend.close()

    backend = MagicMock()
    client._backend = backend

    with patch.object(MemoryClient, "_get_embedder", return_value=_StubEmbedder()):
        await client.store("vectorized content", detail="with detail")

    assert backend.store.called
    backend.upsert_vector.assert_called_once()
    args = backend.upsert_vector.call_args.args
    assert str(args[0]).startswith("M-")
    assert args[1] == [30.0, 1.0, 0.5]


async def test_client_store_rolls_back_when_vector_upsert_fails(client: MemoryClient) -> None:
    """A vector write failure must not leave the primary row committed."""
    real_backend = client._backend
    assert real_backend is not None
    real_backend.close()

    backend = MagicMock()
    backend.upsert_vector.side_effect = RuntimeError("dimension mismatch")
    client._backend = backend

    with (
        patch.object(MemoryClient, "_get_embedder", return_value=_StubEmbedder()),
        pytest.raises(StorageError, match="rolled back"),
    ):
        await client.store("vectorized content", detail="with detail")

    backend.store.assert_called_once()
    backend.delete.assert_called_once()


async def test_client_recall_passes_stored_embeddings_to_hybrid_search(client: MemoryClient) -> None:
    """MemoryClient.recall feeds backend vectors into the hybrid pipeline."""
    real_backend = client._backend
    assert real_backend is not None
    real_backend.close()

    entry = MemoryEntry(id="M-001", content="pydantic model", namespace="default")
    backend = MagicMock()
    backend.list_entries.return_value = [entry]
    backend.get_stored_embeddings.return_value = {"M-001": [0.9, 0.1, 0.0]}
    client._backend = backend

    with (
        patch.object(MemoryClient, "_get_embedder", return_value=_StubEmbedder()),
        patch("trw_memory.retrieval.pipeline.hybrid_search", return_value=[entry]) as hybrid_search_mock,
    ):
        await client.recall("pydantic")

    assert hybrid_search_mock.called
    assert hybrid_search_mock.call_args.kwargs["stored_embeddings"] == {"M-001": [0.9, 0.1, 0.0]}


async def test_client_recall_updates_access_metadata_only_for_returned_entries(client: MemoryClient) -> None:
    """Client recall should mark surfaced entries without touching other rows."""
    stored_target = await client.store("target context", importance=0.9)
    stored_other = await client.store("unrelated background", importance=0.2)

    results = await client.recall("target", limit=1)

    backend = client._backend
    assert backend is not None
    touched = backend.get(stored_target["memory_id"])
    untouched = backend.get(stored_other["memory_id"])

    assert results[0]["memory_id"] == stored_target["memory_id"]
    assert touched is not None
    assert touched.access_count == 1
    assert touched.last_accessed_at is not None
    assert untouched is not None
    assert untouched.access_count == 0
    assert untouched.last_accessed_at is None


def test_memory_store_impl_upserts_vector_when_embedder_available() -> None:
    """memory_store_impl persists vectors through the backend when available."""
    backend = MagicMock()
    with patch("trw_memory.tools.store.get_local_embedder", return_value=_StubEmbedder()):
        result = memory_store_impl("stored through tool", "project:default", backend=backend)

    assert result["status"] == "stored"
    backend.store.assert_called_once()
    backend.upsert_vector.assert_called_once()


def test_memory_store_impl_rolls_back_when_vector_upsert_fails() -> None:
    """Tool store reports an error and deletes the row on vector failure."""
    backend = MagicMock()
    backend.upsert_vector.side_effect = RuntimeError("dimension mismatch")

    with patch("trw_memory.tools.store.get_local_embedder", return_value=_StubEmbedder()):
        result = memory_store_impl("stored through tool", "project:default", backend=backend)

    assert result["status"] == "error"
    backend.store.assert_called_once()
    backend.delete.assert_called_once()


def test_memory_store_impl_uses_configured_embedder_settings() -> None:
    """Tool store must use configured embedding model/dimension, not defaults."""
    backend = MagicMock()
    config = MemoryConfig(embedding_model="custom-model", embedding_dim=768)

    with patch("trw_memory.tools.store.get_local_embedder", return_value=None) as embedder_mock:
        memory_store_impl("stored through tool", "project:default", backend=backend, config=config)

    embedder_mock.assert_called_once_with(model_name="custom-model", dim=768)


def test_memory_recall_impl_passes_stored_embeddings_to_hybrid_search() -> None:
    """memory_recall_impl forwards backend vectors into hybrid_search."""
    entry = MemoryEntry(id="M-001", content="pydantic", namespace="project:default")
    backend = MagicMock()
    backend.list_entries.return_value = [entry]
    backend.get_stored_embeddings.return_value = {"M-001": [0.8, 0.2, 0.0]}

    with (
        patch("trw_memory.tools.recall.get_local_embedder", return_value=_StubEmbedder()),
        patch("trw_memory.tools.recall.hybrid_search", return_value=[entry]) as hybrid_search_mock,
    ):
        result = memory_recall_impl("pydantic", "project:default", backend=backend)

    assert "memories" in result
    assert hybrid_search_mock.called
    assert hybrid_search_mock.call_args.kwargs["stored_embeddings"] == {"M-001": [0.8, 0.2, 0.0]}


def test_memory_recall_impl_uses_configured_embedder_settings() -> None:
    """Tool recall must honor configured embedding settings."""
    entry = MemoryEntry(id="M-001", content="pydantic", namespace="project:default")
    backend = MagicMock()
    backend.list_entries.return_value = [entry]
    backend.get_stored_embeddings.return_value = {}
    config = MemoryConfig(embedding_model="custom-model", embedding_dim=768)

    with (
        patch("trw_memory.tools.recall.get_local_embedder", return_value=None) as embedder_mock,
        patch("trw_memory.tools.recall.hybrid_search", return_value=[entry]),
    ):
        memory_recall_impl("pydantic", "project:default", backend=backend, config=config)

    embedder_mock.assert_called_once_with(model_name="custom-model", dim=768)


def test_memory_recall_impl_updates_access_metadata_only_for_returned_entries(tmp_path: Path) -> None:
    """Tool recall should update surfaced rows after filtering and limit capping."""
    from trw_memory.integrations._backend import create_backend_from_config, make_entry

    config = MemoryConfig(storage_path=str(tmp_path / "storage"))
    target_entry = make_entry("highest utility", namespace="project:default", importance=0.9)
    other_entry = make_entry("lower utility", namespace="project:default", importance=0.1)

    with create_backend_from_config(config, "project:default") as backend:
        backend.store(target_entry)
        backend.store(other_entry)

        result = memory_recall_impl("", "project:default", backend=backend, limit=1, config=config)

        touched = backend.get(target_entry.id)
        untouched = backend.get(other_entry.id)

    memories = cast(list[dict[str, object]], result["memories"])
    assert result["total_matches"] == 1
    assert memories[0]["id"] == target_entry.id
    assert touched is not None
    assert touched.access_count == 1
    assert touched.last_accessed_at is not None
    assert untouched is not None
    assert untouched.access_count == 0
    assert untouched.last_accessed_at is None


def test_client_get_embedder_uses_configured_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MemoryClient._get_embedder forwards configured model/dim."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("MEMORY_EMBEDDING_DIM", "768")

    client = MemoryClient(namespace="default", mode="local")
    try:
        with patch("trw_memory.embeddings.get_local_embedder", return_value=None) as embedder_mock:
            client._get_embedder()
    finally:
        backend = client._backend
        if backend is not None:
            backend.close()

    embedder_mock.assert_called_once_with(model_name="custom-model", dim=768)
