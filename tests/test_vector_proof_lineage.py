"""Copies preserve source generation; synthetic questions retain parent identity."""

import json

import pytest

from trw_memory._client_hype import expand_hype_siblings, hype_sibling_id
from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.curate import NamespaceStores, rename_namespace
from trw_memory.storage.sqlite_backend import SQLiteBackend


@pytest.fixture
def backend(tmp_path):
    pytest.importorskip("sqlite_vec")
    result = SQLiteBackend(tmp_path / "memory.db", dim=2)
    assert result.vec_available
    yield result
    result.close()


def space():
    return EmbeddingSpace("a" * 64, "fixture-encoding", 2)


@pytest.mark.parametrize("proof_state", ["current", "stale-input", "unknown"])
def test_namespace_move_preserves_bytes_and_only_current_source_proof(backend, proof_state):
    entry = MemoryEntry(id="entry", namespace="project:source", content="Summary", detail="Detail")
    backend.store(entry)
    vector = [0.6, 0.8]
    proof = (
        VectorProvenance.for_vector(space(), "Summary Detail" if proof_state == "current" else "Old input", vector)
        if proof_state != "unknown"
        else None
    )
    backend.upsert_vector(entry.id, vector, namespace=entry.namespace, provenance=proof)
    result = rename_namespace(NamespaceStores.shared(backend), "project:source", "project:destination")
    record = backend.get_vector_records([entry.id], namespace="project:destination")[entry.id]
    assert result.moved == 1
    assert record.embedding == pytest.approx(vector)
    assert record.provenance == (proof if proof_state == "current" else None)
    assert backend.get_vector_records([entry.id], namespace="project:source") == {}


def test_question_generation_binds_question_and_current_parent(backend):
    parent = MemoryEntry(id="parent", content="Summary", detail="Detail")
    question = "How does this documented behavior work?"

    class Provider:
        def embed(self, text):
            assert text == question
            return [1.0, 0.0]

        def embedding_space(self):
            return space()

    class Generator:
        def generate(self, entry):
            assert entry is parent
            return [question]

    with backend.transaction():
        backend.store(parent)
        expand_hype_siblings(
            backend=backend,
            config=MemoryConfig(hype_enabled=True),
            entry=parent,
            embedder=Provider(),
            generator=Generator(),
        )
    record = backend.get_vector_records([hype_sibling_id(parent.id, 0)], namespace="default")[
        hype_sibling_id(parent.id, 0)
    ]
    proof = record.provenance
    assert proof.matches(
        space(), question, record.embedding, input_role="generated-question", parent_text="Summary Detail"
    )
    assert not proof.matches(
        space(), question, record.embedding, input_role="generated-question", parent_text="Changed Detail"
    )
    assert not proof.matches(space(), question, record.embedding)  # never ordinary document evidence


def test_legacy_document_proof_without_parent_field_still_parses():
    proof = VectorProvenance.for_vector(space(), "document", [1.0, 0.0])
    raw = json.loads(proof.to_json())
    del raw["parent_input_sha256"]
    assert VectorProvenance.from_json(json.dumps(raw)) == proof


def test_generated_question_without_parent_identity_is_unknown():
    proof = VectorProvenance.for_vector(space(), "document", [1.0, 0.0])
    raw = json.loads(proof.to_json())
    raw["input_role"] = "generated-question"
    assert VectorProvenance.from_json(json.dumps(raw)) is None
