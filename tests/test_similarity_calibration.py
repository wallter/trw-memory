"""Model-aware similarity thresholds, pinned against the measured calibration set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trw_memory.embeddings import calibrated_threshold
from trw_memory.embeddings._similarity_calibration import _ANCHORS, encoder_model, register_space_model
from trw_memory.embeddings.local import LocalEmbeddingProvider
from trw_memory.embeddings.provenance import EmbeddingSpace
from trw_memory.graph import create_similarity_edges
from trw_memory.lifecycle.dedup import check_duplicate
from trw_memory.models.config import MemoryConfig

from ._test_dedup_support import StubEmbedder, make_entry
from ._test_graph_support import _make_conn, _make_entry

BGE = "BAAI/bge-small-en-v1.5"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
_FIXTURE = Path(__file__).parent / "data" / "similarity_calibration_scores.json"
_DUPLICATES = frozenset({"near_exact", "paraphrase"})
#: Reference threshold -> labels that count as a correct hit there. The 0.95
#: skip drops the new learning outright, so only near-exact edits qualify.
_POSITIVES = {0.75: _DUPLICATES, 0.85: _DUPLICATES, 0.90: _DUPLICATES, 0.92: _DUPLICATES, 0.95: {"near_exact"}}
_GRID = [round(i * 0.005, 3) for i in range(201)]


def _pairs() -> list[tuple[str, float, float]]:
    doc = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert doc["reference_model"] == MINILM
    assert doc["model"] == BGE
    return [(str(label), float(ref), float(model)) for label, ref, model in doc["pairs"]]


def _precision_recall(
    pairs: list[tuple[str, float, float]], column: int, threshold: float, positives: frozenset[str] | set[str]
) -> tuple[float, float]:
    hits = [pair for pair in pairs if pair[column] >= threshold]
    true_hits = sum(1 for pair in hits if pair[0] in positives)
    total = sum(1 for pair in pairs if pair[0] in positives)
    return (true_hits / len(hits) if hits else 1.0), true_hits / total


def _derive(pairs: list[tuple[str, float, float]], reference: float) -> float:
    """The documented rule: precision floor first, then match the reference recall."""
    positives = _POSITIVES[reference]
    target_precision, target_recall = _precision_recall(pairs, 1, reference, positives)
    floor = min(
        t
        for t in _GRID
        if all(_precision_recall(pairs, 2, u, positives)[0] >= target_precision for u in _GRID if u >= t)
    )
    recall_match = max(t for t in _GRID if _precision_recall(pairs, 2, t, positives)[1] >= target_recall)
    return max(floor, recall_match)


def _share_match(pairs: list[tuple[str, float, float]], reference: float) -> float:
    share = sum(1 for pair in pairs if pair[1] >= reference) / len(pairs)
    return max(t for t in _GRID if sum(1 for pair in pairs if pair[2] >= t) / len(pairs) >= share)


class TestAnchorsMatchTheCalibrationSet:
    def test_fixture_is_the_documented_set(self) -> None:
        labels = [label for label, _, _ in _pairs()]
        assert len(labels) == 512
        assert {label: labels.count(label) for label in set(labels)} == {
            "near_exact": 90,
            "paraphrase": 73,
            "same_topic": 168,
            "unrelated": 160,
            "template_distinct": 21,
        }

    @pytest.mark.parametrize("reference", sorted(_POSITIVES))
    def test_decision_anchor_is_rederived_from_the_scores(self, reference: float) -> None:
        assert calibrated_threshold(reference, BGE) == _derive(_pairs(), reference)

    @pytest.mark.parametrize("reference", [0.0, 0.5])
    def test_low_anchor_matches_the_share_of_pairs_above_it(self, reference: float) -> None:
        assert calibrated_threshold(reference, BGE) == _share_match(_pairs(), reference)

    @pytest.mark.parametrize("reference", sorted(_POSITIVES))
    def test_calibrated_value_keeps_reference_precision(self, reference: float) -> None:
        pairs = _pairs()
        positives = _POSITIVES[reference]
        reference_precision, _ = _precision_recall(pairs, 1, reference, positives)
        uncalibrated_precision, _ = _precision_recall(pairs, 2, reference, positives)
        calibrated_precision, _ = _precision_recall(pairs, 2, calibrated_threshold(reference, BGE), positives)
        assert calibrated_precision >= reference_precision
        if reference in (0.75, 0.85, 0.95):  # the reference value itself over-matches under bge
            assert uncalibrated_precision < reference_precision


class TestCalibratedThreshold:
    def test_reference_unknown_and_missing_encoders_are_unchanged(self) -> None:
        for encoder in (MINILM, "all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5", None, object()):
            assert calibrated_threshold(0.85, encoder) == 0.85

    def test_model_id_resolves_case_insensitively_and_by_repo_local_name(self) -> None:
        assert calibrated_threshold(0.85, "bge-small-en-v1.5") == calibrated_threshold(0.85, BGE)
        assert calibrated_threshold(0.85, "baai/BGE-SMALL-EN-V1.5") == calibrated_threshold(0.85, BGE)

    def test_values_between_anchors_interpolate_monotonically(self) -> None:
        values = [calibrated_threshold(t / 100, BGE) for t in range(-100, 101)]
        assert values == sorted(values)
        assert calibrated_threshold(0.925, BGE) == pytest.approx(0.955 + (0.98 - 0.955) * (0.005 / 0.03), abs=1e-4)
        assert calibrated_threshold(1.0, BGE) == 1.0

    def test_out_of_range_values_pass_through(self) -> None:
        """A threshold above 1.0 disables a decision in any scale."""
        assert calibrated_threshold(1.01, BGE) == 1.01
        assert calibrated_threshold(-2.0, BGE) == -2.0

    def test_default_skip_stays_above_default_merge(self) -> None:
        assert calibrated_threshold(0.95, BGE) > calibrated_threshold(0.85, BGE)

    def test_only_measured_models_are_calibrated(self) -> None:
        assert set(_ANCHORS) == {BGE.lower()}


class TestEncoderIdentity:
    def test_provider_exposes_its_model_without_loading(self) -> None:
        provider = LocalEmbeddingProvider(model_name=BGE)
        assert provider.model_name == BGE
        assert encoder_model(provider) == BGE

    def test_registered_space_resolves_to_its_model(self) -> None:
        space = EmbeddingSpace(artifact_sha256="a" * 64, encoding="test-encoder:calibration", dimensions=384)
        assert encoder_model(space) is None
        register_space_model(space, BGE)
        assert calibrated_threshold(0.75, space) == calibrated_threshold(0.75, BGE)

    def test_declared_space_names_its_model_without_a_registration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A process that never loaded the encoder (it reads another process's
        # vectors) has no registration; the declared encoding still names the model.
        from trw_memory.embeddings import _similarity_calibration
        from trw_memory.embeddings._declared_space import declared_embedding_space

        monkeypatch.setattr(_similarity_calibration, "_SPACE_MODELS", {})
        space = declared_embedding_space(BGE, "", 384)
        assert encoder_model(space) == BGE
        assert calibrated_threshold(0.75, space) == calibrated_threshold(0.75, BGE) != 0.75
        measured = EmbeddingSpace(artifact_sha256="c" * 64, encoding="test-encoder:unregistered", dimensions=384)
        assert encoder_model(measured) is None

    def test_non_string_model_name_attribute_is_ignored(self) -> None:
        class _Mock:
            model_name = 42

        assert encoder_model(_Mock()) is None
        assert encoder_model(["unhashable"]) is None


class _NamedStub(StubEmbedder):
    def __init__(self, model_name: str) -> None:
        super().__init__(available=True)
        self.model_name = model_name


def _dedup_action(model_name: str, similarity: float) -> str:
    embedder = _NamedStub(model_name)
    embedder.set_vector("new ", [1.0, 0.0])
    embedder.set_vector("old ", [similarity, (1 - similarity**2) ** 0.5])
    return check_duplicate("new", [make_entry("e1", "old")], embedder, config=MemoryConfig()).action


class TestCallSitesUseTheEncoderScale:
    def test_dedup_merges_a_bge_pair_only_above_the_calibrated_value(self) -> None:
        assert _dedup_action(MINILM, 0.87) == "merge"
        assert _dedup_action(BGE, 0.87) == "store"
        assert _dedup_action(BGE, 0.90) == "merge"
        assert _dedup_action(BGE, 0.96) == "merge"
        assert _dedup_action(BGE, 0.99) == "skip"

    def test_similarity_edges_use_the_threshold_they_are_given(self) -> None:
        candidates = [("e2", [0.8, 0.6])]
        conn = _make_conn()
        assert (
            create_similarity_edges(_make_entry("e1"), conn, embedding=[1.0, 0.0], candidate_embeddings=candidates) == 2
        )
        conn = _make_conn()
        count = create_similarity_edges(
            _make_entry("e1"),
            conn,
            embedding=[1.0, 0.0],
            candidate_embeddings=candidates,
            threshold=calibrated_threshold(0.75, BGE),
        )
        assert count == 0
