"""Vector-capability seam: ``supports_vectors`` + embedding-skip on write paths.

Covers the explicit vector-capability contract and the store-path optimisation
that skips the embedding-model call when no vector sink can consume the result.

The friction this guards against: ``StorageBackend``'s vector methods silently
no-op on non-vector backends, so a store path could pay the (expensive)
embedding cost only to discard it through a no-op ``upsert_vector``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory._client_bulk_store import BulkStoreRequest
from trw_memory.client import MemoryClient
from trw_memory.lifecycle.tiers import _runtime as tier_runtime
from trw_memory.lifecycle.tiers._runtime import embedding_has_consumer
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend


class _MinimalBackend(StorageBackend):
    """Smallest concrete backend — exercises the ABC's safe defaults."""

    def store(self, entry: MemoryEntry) -> None:  # pragma: no cover - trivial
        return None

    def get(self, entry_id: str) -> MemoryEntry | None:  # pragma: no cover
        return None

    def update(self, entry_id: str, **fields: object) -> MemoryEntry | None:  # pragma: no cover
        return None

    def delete(self, entry_id: str) -> bool:  # pragma: no cover
        return False

    def search(self, query: str, **kwargs: object) -> list[MemoryEntry]:  # pragma: no cover
        return []

    def count(self, namespace: str | None = None) -> int:  # pragma: no cover
        return 0

    def list_entries(self, **kwargs: object) -> list[MemoryEntry]:  # pragma: no cover
        return []

    def close(self) -> None:  # pragma: no cover - trivial
        return None


class _RecordingVectorBackend(_MinimalBackend):
    """A vector-capable fake that records the vectors it is handed."""

    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[float]]] = []

    def supports_vectors(self) -> bool:
        return True

    def upsert_vector(self, entry_id: str, embedding: list[float]) -> None:
        self.upserts.append((entry_id, embedding))


# ---------------------------------------------------------------------------
# (3) Capability reflects backend reality
# ---------------------------------------------------------------------------


def test_default_backend_reports_no_vector_support() -> None:
    assert _MinimalBackend().supports_vectors() is False


def test_yaml_backend_reports_no_vector_support(tmp_path: Path) -> None:
    backend = YAMLBackend(tmp_path / "entries")
    try:
        assert backend.supports_vectors() is False
    finally:
        backend.close()


def test_sqlite_capability_matches_vec_available() -> None:
    backend = SQLiteBackend(Path(":memory:"))
    try:
        # supports_vectors() is the public capability signal; vec_available is
        # the underlying sqlite-vec load state — they must never disagree.
        assert backend.supports_vectors() == backend.vec_available
    finally:
        backend.close()


def test_sqlite_capability_false_when_vec_unavailable() -> None:
    backend = SQLiteBackend(Path(":memory:"))
    try:
        backend._vec_available = False  # simulate sqlite-vec missing
        assert backend.supports_vectors() is False
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# embedding_has_consumer predicate
# ---------------------------------------------------------------------------


def _config(
    *,
    encryption_enabled: bool = False,
    local_only: bool = True,
    sync_enabled: bool = False,
    platform_url: str = "",
) -> MemoryConfig:
    return MemoryConfig(
        encryption_enabled=encryption_enabled,
        local_only=local_only,
        sync_enabled=sync_enabled,
        platform_url=platform_url,
    )


def test_consumer_true_when_backend_supports_vectors() -> None:
    # Even with every other sink off, a vector-capable backend is a consumer.
    cfg = _config(encryption_enabled=True)  # tiers off
    assert embedding_has_consumer(cfg, _RecordingVectorBackend()) is True


def test_consumer_true_when_tier_runtime_enabled() -> None:
    # Non-vector backend, but the warm tier keeps its own vector sidecar.
    cfg = _config(encryption_enabled=False)  # tiers on
    assert embedding_has_consumer(cfg, _MinimalBackend()) is True


def test_consumer_true_when_remote_publish_configured() -> None:
    cfg = _config(
        encryption_enabled=True,  # tiers off
        local_only=False,
        sync_enabled=True,
        platform_url="https://example.invalid",
    )
    assert embedding_has_consumer(cfg, _MinimalBackend()) is True


def test_consumer_false_when_no_sink_is_live() -> None:
    # Non-vector backend, tiers off (encryption), no publish → pure waste.
    cfg = _config(encryption_enabled=True)
    assert embedding_has_consumer(cfg, _MinimalBackend()) is False


# ---------------------------------------------------------------------------
# (1)/(2) Store path honours the capability seam
# ---------------------------------------------------------------------------


class _SpyEmbedder:
    """Records every embed call and returns a fixed-dim vector."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self.embed_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return [0.1] * self._dim

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        return [[0.1] * self._dim for _ in texts]

    def available(self) -> bool:
        return True

    def dim(self) -> int:
        return self._dim


async def test_store_skips_embedder_when_no_vector_sink(
    client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(1) A non-vector write path must not call the embedder just to no-op."""
    backend = client._get_backend()
    monkeypatch.setattr(backend, "supports_vectors", lambda: False)
    # Disable the warm tier and remote publish so no sink remains.
    monkeypatch.setattr(tier_runtime, "tier_runtime_enabled", lambda _cfg: False)
    client._config.sync_enabled = False
    client._config.local_only = True

    spy = _SpyEmbedder(client._config.embedding_dim)
    monkeypatch.setattr(client, "_get_embedder", lambda: spy)

    await client.store("no vector sink here", tags=["seam"])

    assert spy.embed_calls == []


async def test_store_embeds_and_upserts_when_backend_supports_vectors(
    client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(2) A vector-capable backend still embeds and persists the vector."""
    backend = client._get_backend()
    assert backend.supports_vectors() is True  # sqlite-vec present in test env

    upserts: list[tuple[str, list[float]]] = []
    real_upsert = backend.upsert_vector

    def _record(entry_id: str, embedding: list[float]) -> None:
        upserts.append((entry_id, embedding))
        real_upsert(entry_id, embedding)

    monkeypatch.setattr(backend, "upsert_vector", _record)

    spy = _SpyEmbedder(client._config.embedding_dim)
    monkeypatch.setattr(client, "_get_embedder", lambda: spy)

    result = await client.store("vector sink present", tags=["seam"])

    assert spy.embed_calls, "embedder should run when backend supports vectors"
    assert [eid for eid, _ in upserts] == [result["memory_id"]]
    assert backend.vector_exists(result["memory_id"]) is True


async def test_bulk_store_skips_batch_embed_when_no_vector_sink(
    client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bulk path mirrors the single-store skip on the batch embed."""
    backend = client._get_backend()
    monkeypatch.setattr(backend, "supports_vectors", lambda: False)
    monkeypatch.setattr(tier_runtime, "tier_runtime_enabled", lambda _cfg: False)
    client._config.sync_enabled = False
    client._config.local_only = True

    spy = _SpyEmbedder(client._config.embedding_dim)
    monkeypatch.setattr(client, "_get_embedder", lambda: spy)

    summary = await client.bulk_store(
        [
            BulkStoreRequest(content="alpha bulk entry", tags=["seam"]),
            BulkStoreRequest(content="beta bulk entry", tags=["seam"]),
        ]
    )

    assert all(item.status == "stored" for item in summary.items)
    assert spy.batch_calls == []
    assert spy.embed_calls == []
