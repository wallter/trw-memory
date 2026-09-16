"""PRD-CORE-272: writes embed canonical text once and retire only owned siblings."""

import pytest

from trw_memory.client import MemoryClient


class _FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [0.1] * 384

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]

    def available(self):
        return True

    def dim(self):
        return 384


async def test_store_update_forget_preserve_canonical_suffix_and_other_namespace(tmp_path, monkeypatch):
    pytest.importorskip("sqlite_vec")
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "tiers"))
    client = MemoryClient("default", mode="local", db_path=tmp_path / "hype.db")
    embedder = _FakeEmbedder()
    monkeypatch.setattr(client, "_get_embedder", lambda: embedder)
    try:
        await client.store("canonical collision", entry_id="P#hype0")
        await client.store("parent content", entry_id="P")
        backend = client._get_backend()
        primary = backend.get_stored_embeddings(["P#hype0"], namespace="default")
        assert set(backend.existing_vector_ids(namespace="default")) == {"P", "P#hype0"}
        backend.upsert_vector("P#hype1", [0.2] * 384, namespace="default")
        backend.upsert_vector("P#hype1", [0.3] * 384, namespace="other")
        await client.store("parent updated", entry_id="P")
        assert not backend.vector_exists("P#hype1", namespace="default")
        assert backend.vector_exists("P#hype1", namespace="other")
        assert backend.get_stored_embeddings(["P#hype0"], namespace="default") == primary
        assert len(embedder.calls) == 3
        assert not hasattr(client, "_question_generator")
        backend.upsert_vector("P#hype2", [0.2] * 384, namespace="default")
        await client.forget("P")
        assert not backend.vector_exists("P#hype2", namespace="default")
        assert backend.get("P", namespace="default") is None
        assert backend.get("P#hype0", namespace="default") is not None
        assert backend.get_stored_embeddings(["P#hype0"], namespace="default") == primary
    finally:
        await client.close()


async def test_neutral_generator_warns_and_creates_no_generator_state(tmp_path):
    with pytest.warns(UserWarning, match="retired"):
        client = MemoryClient("default", mode="local", db_path=tmp_path / "empty.db", question_generator=None)
    try:
        assert not hasattr(client, "_question_generator")
    finally:
        await client.close()


async def test_actor_forget_cleans_legacy_vectors_before_canonical_delete(tmp_path, monkeypatch):
    from tests.conftest import make_entry

    pytest.importorskip("sqlite_vec")
    client = MemoryClient("default", mode="local", db_path=tmp_path / "actor.db")
    try:
        backend = client._get_backend()
        for namespace in ("default", "other"):
            backend.store(
                make_entry(entry_id="actor-parent", namespace=namespace).model_copy(update={"source_identity": "alice"})
            )
            backend.upsert_vector("actor-parent#hype0", [0.1] * 384, namespace=namespace)
        result = await client.forget(actor="alice")
        assert result["entries_deleted"] == 1
        assert not backend.vector_exists("actor-parent#hype0", namespace="default")
        assert backend.vector_exists("actor-parent#hype0", namespace="other")
    finally:
        await client.close()


async def test_no_vectors_allows_ordinary_store_recall_forget(tmp_path, monkeypatch):
    client = MemoryClient("default", mode="local", db_path=tmp_path / "lexical.db")
    backend = client._get_backend()
    monkeypatch.setattr(backend, "_vec_available", False)
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    try:
        await client.store("zebra retrieval guidance", entry_id="lexical")
        assert [r["memory_id"] for r in await client.recall("zebra")] == ["lexical"]
        await client.forget("lexical")
        assert backend.get("lexical", namespace="default") is None
    finally:
        await client.close()


async def test_sdk_recall_preserves_canonical_suffix_without_legacy_hits(tmp_path, monkeypatch):
    pytest.importorskip("sqlite_vec")
    client = MemoryClient("default", mode="local", db_path=tmp_path / "recall.db")
    monkeypatch.setattr(client, "_get_embedder", lambda: _FakeEmbedder())
    try:
        await client.store("zebra retrieval guidance", entry_id="P#hype0")
        backend = client._get_backend()
        backend.upsert_vector("P#hype0#hype0", [0.1] * 384, namespace="default")
        backend.upsert_vector("orphan#hype0", [0.1] * 384, namespace="default")
        assert [r["memory_id"] for r in await client.recall("zebra", limit=1)] == ["P#hype0"]
    finally:
        await client.close()
