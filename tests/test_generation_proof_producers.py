"""SDK single/bulk generation writes carry the exact encoded input's identity."""

import pytest

from trw_memory.client import BulkStoreRequest
from trw_memory.embeddings.provenance import EmbeddingSpace


@pytest.mark.parametrize("bulk", [False, True])
async def test_sdk_producer_persists_proof_for_generated_input(client, monkeypatch, bulk):
    pytest.importorskip("sqlite_vec")
    backend = client._get_backend()
    assert backend.vec_available
    dimension = client._config.embedding_dim

    class Provider:
        inputs = []

        def embed(self, text):
            self.inputs.append(text)
            return [1.0] + [0.0] * (dimension - 1)

        def embed_batch(self, texts):
            return [self.embed(text) for text in texts]

        def embedding_space(self):
            return EmbeddingSpace("a" * 64, "fixture-encoding", dimension)

    provider = Provider()
    monkeypatch.setattr(client, "_get_embedder", lambda: provider)
    if bulk:
        result = await client.bulk_store([BulkStoreRequest(content="Generation evidence", detail="Exact detail")])
        assert result.stored == 1
    else:
        await client.store("Generation evidence", detail="Exact detail")
    entries = backend.list_entries(namespace="default")
    entry = next(row for row in entries if row.content == "Generation evidence")
    record = backend.get_vector_records([entry.id], namespace="default")[entry.id]
    text = "Generation evidence Exact detail"
    assert provider.inputs == [text]
    assert record.provenance.matches(provider.embedding_space(), text, record.embedding)
