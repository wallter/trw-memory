"""Default embedder change: role-aware encoding, space-gated dense scoring, re-embed.

No test here loads a real model: embedders are deterministic doubles that
report an ``EmbeddingSpace`` the way ``LocalEmbeddingProvider`` does, and the
provider-level tests substitute ``sentence_transformers`` in ``sys.modules``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import structlog

from trw_memory.client import MemoryClient
from trw_memory.embeddings import get_local_embedder
from trw_memory.embeddings._declared_space import DECLARED_ENCODING_PREFIX, declared_embedding_space
from trw_memory.embeddings._hf_cache import CacheProbe, CacheState
from trw_memory.embeddings._query_prompts import embed_query, query_prefix
from trw_memory.embeddings.local import LocalEmbeddingProvider
from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.exceptions import EmbeddingUnavailableError, LocalOnlyViolationError
from trw_memory.lifecycle.tiers._warm import WARM_TIER_NAMESPACE, WarmTierStore
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.recall import memory_recall_impl

pytest.importorskip("sqlite_vec")

BGE = "BAAI/bge-small-en-v1.5"
BGE_QUERY = "Represent this sentence for searching relevant passages: "
SPACE_A = EmbeddingSpace("a" * 64, "test-encoder:a", 3)
SPACE_B = EmbeddingSpace("b" * 64, "test-encoder:b", 3)
TARGET = [1.0, 0.0, 0.0]
AWAY = [0.0, 1.0, 0.0]
EXCLUDED_EVENT = "dense_vectors_excluded_embedding_space"


class SpacedEmbedder:
    """Deterministic role-aware provider with a declared embedding space."""

    def __init__(self, space: EmbeddingSpace | None, vectors: dict[str, list[float]] | None = None) -> None:
        self.space = space
        self.vectors = vectors or {}
        self.queries: list[str] = []
        self.documents: list[str] = []

    def _vector(self, text: str) -> list[float]:
        return self.vectors.get(text.strip(), AWAY)

    def embed(self, text: str) -> list[float] | None:
        self.documents.append(text)
        return self._vector(text)

    def embed_query(self, text: str) -> list[float] | None:
        self.queries.append(text)
        return self._vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    def available(self) -> bool:
        return True

    def dim(self) -> int:
        return 3

    def embedding_space(self) -> EmbeddingSpace | None:
        return self.space


def _entry(entry_id: str, content: str, namespace: str = "default") -> MemoryEntry:
    return MemoryEntry(id=entry_id, content=content, namespace=namespace)


def _put(backend: SQLiteBackend, entry: MemoryEntry, vector: list[float] | None, space: EmbeddingSpace | None) -> None:
    backend.store(entry)
    if vector is None:
        return
    proof = VectorProvenance.for_vector(space, f"{entry.content} {entry.detail}", vector) if space else None
    backend.upsert_vector(entry.id, vector, namespace=entry.namespace, provenance=proof)


def _mixed_store(backend: SQLiteBackend) -> None:
    """Three rows whose vectors all equal the query vector, from three spaces."""
    _put(backend, _entry("row-a", "orange kiwi"), TARGET, SPACE_A)
    _put(backend, _entry("row-b", "purple mango"), TARGET, SPACE_B)
    _put(backend, _entry("row-legacy", "legacy lemon"), TARGET, None)


def _excluded_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["event"] == EXCLUDED_EVENT]


@pytest.fixture()
def cfg_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    storage = tmp_path / "storage"
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(storage))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_EMBEDDING_DIM", "3")
    # Rerank is unconditional (PRD-CORE-284); an unavailable model keeps fusion order.
    monkeypatch.setattr("trw_memory.retrieval.reranker.cross_encode_scores", lambda *a, **k: None)
    return storage


# --------------------------------------------------------------------------- defaults


def test_default_embedding_model_is_bge_small() -> None:
    config = MemoryConfig()
    assert config.embedding_model == BGE
    assert config.embedding_dim == 384
    assert LocalEmbeddingProvider()._model_name == BGE


def test_get_local_embedder_defaults_to_bge(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[str, int]] = []

    def fake_provider(*, model_name: str, dim: int) -> SimpleNamespace:
        built.append((model_name, dim))
        return SimpleNamespace(available=lambda: True)

    monkeypatch.setattr("trw_memory.embeddings.LocalEmbeddingProvider", fake_provider)
    assert get_local_embedder() is not None
    assert built == [(BGE, 384)]


# --------------------------------------------------------------------------- query role


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (BGE, BGE_QUERY),
        ("bge-small-en-v1.5", BGE_QUERY),
        ("baai/BGE-BASE-EN-V1.5", BGE_QUERY),
        ("all-MiniLM-L6-v2", ""),
        ("sentence-transformers/all-MiniLM-L6-v2", ""),
    ],
)
def test_query_prefix_table(model: str, expected: str) -> None:
    assert query_prefix(model) == expected


class _RecordingModel:
    def __init__(self) -> None:
        self.encoded: list[object] = []

    def encode(self, text: object, **kwargs: object) -> object:
        self.encoded.append(text)
        if isinstance(text, list):
            return [[1.0, 0.0] for _ in text]
        return [1.0, 0.0]


def _provider_with(model_name: str, monkeypatch: pytest.MonkeyPatch) -> tuple[LocalEmbeddingProvider, _RecordingModel]:
    model = _RecordingModel()
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *a, **k: model)
    )
    monkeypatch.setattr("trw_memory.embeddings.local.dependency_versions", lambda: None)
    provider = LocalEmbeddingProvider(model_name, dim=2)
    monkeypatch.setattr(provider, "_probe_cache", lambda: CacheProbe(CacheState.UNKNOWN))
    return provider, model


def test_bge_prefixes_queries_but_not_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, model = _provider_with(BGE, monkeypatch)
    assert provider.embed_query("when did we ship?") == [1.0, 0.0]
    assert provider.embed("the release shipped on friday") == [1.0, 0.0]
    assert provider.embed_batch(["doc one", "doc two"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert model.encoded == [
        BGE_QUERY + "when did we ship?",
        "the release shipped on friday",
        ["doc one", "doc two"],
    ]


def test_minilm_query_is_encoded_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, model = _provider_with("all-MiniLM-L6-v2", monkeypatch)
    provider.embed_query("when did we ship?")
    assert model.encoded == ["when did we ship?"]


def test_blank_query_is_not_encoded(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, model = _provider_with(BGE, monkeypatch)
    assert provider.embed_query("   ") is None
    assert model.encoded == []


def test_embed_query_helper_falls_back_to_embed_for_role_blind_providers() -> None:
    class RoleBlind:
        def embed(self, text: str) -> list[float] | None:
            return [float(len(text))]

    assert embed_query(RoleBlind(), "four") == [4.0]  # type: ignore[arg-type]
    role_aware = SpacedEmbedder(SPACE_A)
    embed_query(role_aware, "question")
    assert role_aware.queries == ["question"]
    assert role_aware.documents == []


def test_unmeasurable_encoder_reports_a_declared_space_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    bge, _ = _provider_with(BGE, monkeypatch)
    minilm, _ = _provider_with("all-MiniLM-L6-v2", monkeypatch)
    assert bge.embedding_space() is None  # nothing loaded yet: no identity
    assert bge.available() and minilm.available()
    bge_space, minilm_space = bge.embedding_space(), minilm.embedding_space()
    assert bge_space is not None and minilm_space is not None
    assert bge_space.encoding == DECLARED_ENCODING_PREFIX + BGE
    assert bge_space != minilm_space
    assert bge_space == declared_embedding_space(BGE, "", 2)


def test_declared_space_binds_the_snapshot_revision() -> None:
    assert declared_embedding_space(BGE, "rev1", 384) != declared_embedding_space(BGE, "rev2", 384)
    assert declared_embedding_space(BGE, "rev1", 384) != declared_embedding_space(BGE, "rev1", 768)


# --------------------------------------------------------------------------- dense gating


async def test_hybrid_recall_scores_only_the_active_space(cfg_env: Path) -> None:
    client = MemoryClient(namespace="default", mode="local")
    try:
        backend = client._get_backend()
        assert isinstance(backend, SQLiteBackend)
        _mixed_store(backend)
        embedder = SpacedEmbedder(SPACE_A, {"zzqx": TARGET})
        client._embedder, client._embedder_initialized = embedder, True

        with structlog.testing.capture_logs() as events:
            results = await client.recall("zzqx", limit=10)

        ids = [row["memory_id"] for row in results]
        assert "row-a" in ids
        assert "row-b" not in ids and "row-legacy" not in ids
        assert embedder.queries  # the query was encoded in the query role
        excluded = _excluded_events(events)
        assert len(excluded) == 1
        assert excluded[0]["log_level"] == "warning"
        assert excluded[0]["surface"] == "hybrid_recall"
        assert (excluded[0]["excluded"], excluded[0]["mismatched_space"], excluded[0]["no_provenance"]) == (2, 1, 1)
        assert excluded[0]["reembed_required"] is True

        # BM25 still ranks the rows whose vectors were excluded.
        lexical = await client.recall("legacy lemon", limit=10)
        assert "row-legacy" in [row["memory_id"] for row in lexical]
    finally:
        await client.close()


async def test_hybrid_recall_without_embedder_identity_fails_closed(cfg_env: Path) -> None:
    client = MemoryClient(namespace="default", mode="local")
    try:
        backend = client._get_backend()
        assert isinstance(backend, SQLiteBackend)
        _mixed_store(backend)
        client._embedder, client._embedder_initialized = SpacedEmbedder(None, {"zzqx": TARGET}), True
        with structlog.testing.capture_logs() as events:
            results = await client.recall("zzqx", limit=10)
        assert not {"row-a", "row-b", "row-legacy"} & {row["memory_id"] for row in results}
        assert _excluded_events(events)[0]["excluded"] == 3
    finally:
        await client.close()


def test_tool_recall_scores_only_the_active_space(
    tmp_path: Path, cfg_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = SQLiteBackend(tmp_path / "tool.db", dim=3)
    try:
        _mixed_store(backend)
        embedder = SpacedEmbedder(SPACE_A, {"zzqx": TARGET})
        monkeypatch.setattr("trw_memory.tools.recall.get_local_embedder", lambda **kwargs: embedder)
        with structlog.testing.capture_logs() as events:
            result = memory_recall_impl("zzqx", "default", backend=backend, config=MemoryConfig())
        ids = [row["id"] for row in result["memories"]]
        assert "row-a" in ids
        assert "row-b" not in ids and "row-legacy" not in ids
        excluded = _excluded_events(events)
        assert [(event["surface"], event["excluded"]) for event in excluded] == [("memory_recall_tool", 2)]
    finally:
        backend.close()


def _warm_with_mixed_vectors(tmp_path: Path) -> WarmTierStore:
    store = WarmTierStore(tmp_path / "warm")
    for entry_id, content, space in (("w-a", "orange", SPACE_A), ("w-b", "mango", SPACE_B), ("w-l", "lemon", None)):
        data = {"id": entry_id, "content": content, "detail": ""}
        proof = VectorProvenance.for_vector(space, f"{content} ", TARGET) if space else None
        store.warm_add(entry_id, data, TARGET, provenance=proof)
    return store


def test_warm_vector_search_scores_only_the_active_space(tmp_path: Path) -> None:
    store = _warm_with_mixed_vectors(tmp_path)
    try:
        with structlog.testing.capture_logs() as events:
            hits = store.warm_search(["zzqx"], TARGET, top_k=10, query_space=SPACE_A)
        assert [hit["id"] for hit in hits] == ["w-a"]
        assert _excluded_events(events)[0]["surface"] == "warm_search"

        with structlog.testing.capture_logs() as events:
            rows = store.discovery_entries(TARGET, query_space=SPACE_A)
        scored = {row["id"] for row in rows if "_tier_relevance" in row}
        assert scored == {"w-a"}
        assert {row["id"] for row in rows} == {"w-a", "w-b", "w-l"}  # every row stays a keyword candidate
        assert _excluded_events(events)[0]["excluded"] == 2
    finally:
        store.close()


def test_warm_vector_search_without_query_space_fails_closed(tmp_path: Path) -> None:
    store = _warm_with_mixed_vectors(tmp_path)
    try:
        assert store.warm_search(["zzqx"], TARGET, top_k=10) == []
        rows = store.discovery_entries(TARGET)
        assert not any("_tier_relevance" in row for row in rows)
    finally:
        store.close()


# --------------------------------------------------------------------------- re-embed


def _space_of(backend: SQLiteBackend, ids: list[str]) -> dict[str, EmbeddingSpace | None]:
    records = backend.get_vector_records(ids, namespace="default")
    return {entry_id: (record.provenance.space if record.provenance else None) for entry_id, record in records.items()}


async def test_reembed_converts_a_mixed_store_and_is_idempotent(cfg_env: Path) -> None:
    client = MemoryClient(namespace="default", mode="local")
    try:
        backend = client._get_backend()
        assert isinstance(backend, SQLiteBackend)
        _mixed_store(backend)
        backend.store(_entry("row-none", "no vector yet"))
        embedder = SpacedEmbedder(SPACE_A)
        client._embedder, client._embedder_initialized = embedder, True

        first = await client.reembed(batch_size=2)
        assert (first["examined"], first["reembedded"], first["already_current"], first["skipped"]) == (4, 3, 1, 0)
        assert first["embedding_space"] == SPACE_A.encoding
        ids = ["row-a", "row-b", "row-legacy", "row-none"]
        assert _space_of(backend, ids) == dict.fromkeys(ids, SPACE_A)
        assert sorted(embedder.documents) == ["legacy lemon ", "no vector yet ", "purple mango "]

        second = await client.reembed(batch_size=2)
        assert (second["reembedded"], second["already_current"]) == (0, 4)

        with structlog.testing.capture_logs() as events:
            await client.recall("zzqx", limit=10)
        assert _excluded_events(events) == []
    finally:
        await client.close()


async def test_reembed_migrates_warm_tier_vectors(cfg_env: Path) -> None:
    client = MemoryClient(namespace="default", mode="local")
    try:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager

        warm = get_tier_manager(client._config, "default")._warm_store
        warm.warm_add("w-b", {"id": "w-b", "content": "mango", "detail": ""}, TARGET)
        warm.warm_add("w-none", {"id": "w-none", "content": "no vector"}, None)
        client._embedder, client._embedder_initialized = SpacedEmbedder(SPACE_A), True

        result = await client.reembed()
        assert (result["warm_examined"], result["warm_reembedded"]) == (1, 1)
        backend = warm._get_warm_backend(dim=3)
        assert backend is not None
        record = backend.get_vector_records(["w-b"], namespace=WARM_TIER_NAMESPACE)["w-b"]
        assert record.provenance is not None and record.provenance.space == SPACE_A
        assert backend.get_vector_records(["w-none"], namespace=WARM_TIER_NAMESPACE) == {}
        assert (await client.reembed())["warm_reembedded"] == 0
    finally:
        await client.close()


async def test_reembed_refuses_without_an_identifiable_embedder(cfg_env: Path) -> None:
    client = MemoryClient(namespace="default", mode="local")
    try:
        client._embedder, client._embedder_initialized = None, True
        with pytest.raises(EmbeddingUnavailableError, match="no embedding model"):
            await client.reembed()
        client._embedder = SpacedEmbedder(None)
        with pytest.raises(EmbeddingUnavailableError, match="embedding space"):
            await client.reembed()
        with pytest.raises(ValueError, match="batch_size"):
            await client.reembed(batch_size=0)
    finally:
        await client.close()


async def test_reembed_offline_with_uncached_model_raises_instead_of_downloading(
    cfg_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRW_OFFLINE", "1")
    loads: list[dict[str, object]] = []

    def refuse(model_ref: str, **kwargs: object) -> None:
        loads.append({"model": model_ref, **kwargs})
        raise OSError("not in the local cache")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=refuse))
    monkeypatch.setattr(
        "trw_memory.embeddings.local.probe_model_cache", lambda model_name: CacheProbe(CacheState.ABSENT)
    )
    client = MemoryClient(namespace="default", mode="local")
    try:
        with pytest.raises(LocalOnlyViolationError, match="TRW_OFFLINE/HF_HUB_OFFLINE"):
            await client.reembed()
        assert loads and all(load["local_files_only"] is True for load in loads)
        assert loads[0]["model"] == BGE
    finally:
        await client.close()


def test_cli_reembed_reports_counts(cfg_env: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from trw_memory.cli import main

    monkeypatch.setattr(MemoryClient, "_get_embedder", lambda self: SpacedEmbedder(SPACE_A))
    client = MemoryClient(namespace="default", mode="local")
    backend = client._get_backend()
    assert isinstance(backend, SQLiteBackend)
    _put(backend, _entry("row-b", "purple mango"), TARGET, SPACE_B)
    backend.close()

    assert main(["reembed", "--namespace", "default", "--batch-size", "8", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reembedded"] == 1 and payload["namespace"] == "default"

    assert main(["reembed", "--namespace", "default"]) == 0
    assert "0 re-embedded" in capsys.readouterr().out


def test_graph_similarity_edges_only_join_same_space_vectors(tmp_path: Path) -> None:
    from trw_memory.graph import update_entry_graph

    backend = SQLiteBackend(tmp_path / "graph.db", dim=3)
    try:
        for entry_id, space in (("new", SPACE_A), ("same", SPACE_A), ("other", SPACE_B), ("legacy", None)):
            _put(backend, _entry(entry_id, f"{entry_id} text"), TARGET, space)
        entry = backend.get("new", namespace="default")
        assert entry is not None
        update_entry_graph(entry, backend, embedding=TARGET)
        rows = backend._conn.execute(
            "SELECT source_id, target_id FROM memory_graph_edges WHERE edge_type = 'similarity'"
        ).fetchall()
        assert {entry_id for row in rows for entry_id in row} == {"new", "same"}
    finally:
        backend.close()
