"""Generation records distinguish compatible evidence from unknown legacy data."""

from __future__ import annotations

import json
import struct

import pytest

from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance, input_digest, vector_digest


def _space() -> EmbeddingSpace:
    return EmbeddingSpace("a" * 64, "sentence-transformers/normalized/v1", 2)


def test_roundtrip_matches_actual_float32_storage_bytes() -> None:
    vector = [0.1, 0.2]
    proof = VectorProvenance.for_vector(_space(), "exact document\n", vector)
    stored = struct.unpack("2f", struct.pack("2f", *vector))
    recovered = VectorProvenance.from_json(proof.to_json())
    assert recovered == proof
    assert recovered.matches(_space(), "exact document\n", stored)
    assert not recovered.matches(_space(), "exact document", stored)


def test_model_encoding_role_dimension_and_bytes_are_independent_constraints() -> None:
    proof = VectorProvenance.for_vector(_space(), "document", [1.0, 0.0])
    assert not proof.matches(EmbeddingSpace("b" * 64, _space().encoding, 2), "document", [1.0, 0.0])
    assert not proof.matches(EmbeddingSpace("a" * 64, "other-policy", 2), "document", [1.0, 0.0])
    assert not proof.matches(_space(), "document", [1.0, 0.0], input_role="generated-question")
    assert not proof.matches_vector([1.0])
    assert not proof.matches_vector([0.0, 1.0])


@pytest.mark.parametrize("raw", [None, "", "{}", "[]", "null", '{"version":1}', "not-json"])
def test_absent_or_malformed_proof_stays_unknown(raw) -> None:
    assert VectorProvenance.from_json(raw) is None


@pytest.mark.parametrize("change", [{"version": 2}, {"version": True}, {"input_sha256": "current-model"}])
def test_unsupported_records_are_not_repaired(change) -> None:
    data = json.loads(VectorProvenance.for_vector(_space(), "document", [1.0, 0.0]).to_json())
    data.update(change)
    assert VectorProvenance.from_json(json.dumps(data)) is None


@pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")], [1e300]])
def test_invalid_vector_cannot_receive_proof(vector) -> None:
    with pytest.raises(ValueError):
        vector_digest(vector)


def test_input_identity_is_exact_not_semantic_normalization() -> None:
    assert input_digest("A") != input_digest("a")
    assert input_digest("a\nb") != input_digest("a b")


def test_dimension_mismatch_is_rejected_before_generation_record() -> None:
    with pytest.raises(ValueError, match="dimension"):
        VectorProvenance.for_vector(_space(), "document", [1.0])


def test_optional_descriptor_never_loads_unknown_provider() -> None:
    from trw_memory.embeddings.provenance import generation_provenance_kwargs

    class Unknown:
        def available(self):
            raise AssertionError("must not initialize provider")

        def embed(self, text):
            raise AssertionError("must not infer again")

    assert generation_provenance_kwargs(Unknown(), "document", [1.0, 0.0]) == {}


def test_known_descriptor_binds_generated_input_and_wrong_shape_stays_unknown() -> None:
    from trw_memory.embeddings.provenance import generation_provenance_kwargs

    class Known:
        def embedding_space(self):
            return _space()

    result = generation_provenance_kwargs(Known(), "document", [1.0, 0.0])
    assert result["provenance"].matches(_space(), "document", [1.0, 0.0])
    assert generation_provenance_kwargs(Known(), "document", [1.0]) == {}
