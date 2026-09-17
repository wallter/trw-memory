"""Confidence-bounded recall: the cross-encoder cutoff after re-ranking."""

from __future__ import annotations

from unittest.mock import patch

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.pipeline import hybrid_search
from trw_memory.security.namespace_scope import authorize_namespaces
from trw_memory.security.rbac import Permission

NS = "project:cut"


def _entries() -> list[MemoryEntry]:
    return [MemoryEntry(id=f"e{i}", content=f"pydantic note {i}", namespace=NS, tags=[]) for i in range(8)]


def _scope():
    return authorize_namespaces(MemoryConfig(rbac_enabled=False), [NS], Permission.READ, "test")


def _scored(query, entries, *, model_name, local_only=False):
    # e0 and e1 strong, e2 weak, the rest clearly unrelated; order flipped to prove re-ranking happened
    table = {"e0": 4.0, "e1": 2.5, "e2": -7.0, "e3": -9.0, "e4": -9.5, "e5": -10.0, "e6": -10.5, "e7": -11.0}
    return sorted(((e, table[e.id]) for e in entries), key=lambda x: x[1], reverse=True)


def test_cutoff_drops_low_confidence_rows_but_keeps_min_keep() -> None:
    with patch("trw_memory.retrieval.reranker.cross_encode_scores", side_effect=_scored):
        out = hybrid_search(
            "pydantic",
            _entries(),
            scope=_scope(),
            rerank=True,
            rerank_candidates=8,
            rerank_min_score=-8.0,
            rerank_min_keep=2,
            top_k=50,
        )
    assert [e.id for e in out] == ["e0", "e1", "e2"]


def test_min_keep_floor_applies_when_everything_is_below_the_cut() -> None:
    with patch("trw_memory.retrieval.reranker.cross_encode_scores", side_effect=_scored):
        out = hybrid_search(
            "pydantic",
            _entries(),
            scope=_scope(),
            rerank=True,
            rerank_candidates=8,
            rerank_min_score=100.0,
            rerank_min_keep=3,
            top_k=50,
        )
    assert [e.id for e in out] == ["e0", "e1", "e2"]


def test_no_cutoff_keeps_every_candidate_in_reranked_order() -> None:
    with patch("trw_memory.retrieval.reranker.cross_encode_scores", side_effect=_scored):
        out = hybrid_search(
            "pydantic",
            _entries(),
            scope=_scope(),
            rerank=True,
            rerank_candidates=8,
            rerank_min_score=None,
            top_k=50,
        )
    assert len(out) == 8 and [e.id for e in out[:3]] == ["e0", "e1", "e2"]


def test_unavailable_cross_encoder_keeps_fusion_order_and_count() -> None:
    with patch("trw_memory.retrieval.reranker.cross_encode_scores", return_value=None):
        out = hybrid_search(
            "pydantic",
            _entries(),
            scope=_scope(),
            rerank=True,
            rerank_candidates=8,
            rerank_min_score=-8.0,
            top_k=50,
        )
    assert len(out) == 8


def test_config_defaults_and_env_alias() -> None:
    cfg = MemoryConfig()
    assert cfg.recall_rerank_min_score == -8.0 and cfg.recall_rerank_min_keep == 5
    assert MemoryConfig(recall_rerank_min_score=None).recall_rerank_min_score is None


def test_confidence_bounded_merge_holds_tier_rows_to_the_same_floor() -> None:
    from trw_memory._client_recall_helpers import merge_local_candidates
    from trw_memory.retrieval.recall_selection import LocalCandidate

    cfg = MemoryConfig(recall_rerank=True, recall_rerank_min_score=-8.0)
    local = [LocalCandidate(MemoryEntry(id="h1", content="hybrid hit", namespace=NS), 1.0)]
    noise = LocalCandidate(MemoryEntry(id="w1", content="warm noise", namespace=NS), 0.4)
    good = LocalCandidate(MemoryEntry(id="w2", content="warm good", namespace=NS), 0.4)
    cold = LocalCandidate(MemoryEntry(id="c1", content="archived hit", namespace=NS), 0.3, cold=True)
    table = {"w1": -10.0, "w2": 1.0}

    def scores(query, entries, *, model_name, local_only=False):
        return sorted(((e, table[e.id]) for e in entries), key=lambda x: x[1], reverse=True)

    with patch("trw_memory.retrieval.reranker.cross_encode_scores", side_effect=scores):
        merged = merge_local_candidates(local, [noise, good, cold], 10, ["hit"], cfg, None, query="hit")
    assert [c.entry.id for c in merged] == ["h1", "w2", "c1"]
    assert all(c.tier_fallback for c in merged[1:])
    # legacy callers (no query) and a disabled floor keep every tier row
    legacy = merge_local_candidates(local, [noise, good, cold], 10, ["hit"], cfg, None)
    assert [c.entry.id for c in legacy] == ["h1", "w1", "w2", "c1"]


def test_malformed_model_output_degrades_to_none_not_an_exception() -> None:
    from trw_memory.retrieval import reranker

    class BadModel:
        def predict(self, pairs):
            return [0.5]  # wrong length

    with patch.object(reranker, "_get_model", return_value=BadModel()):
        assert reranker.cross_encode_scores("q", _entries()) is None
        assert [e.id for e in reranker.cross_encode_rerank("q", _entries())] == [e.id for e in _entries()]


def test_offline_switch_forces_local_files_only_and_never_downloads(monkeypatch) -> None:
    """TRW_OFFLINE / HF_HUB_OFFLINE / local_only must reach the cross-encoder loader
    (rerank is on by default, so this is the README's no-outbound-calls contract)."""
    from trw_memory.retrieval import reranker

    calls: list[dict] = []

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only"):
                raise OSError("not in local cache")  # what huggingface_hub raises offline

    monkeypatch.setattr(reranker, "_cross_encoder_cls", FakeCrossEncoder)
    monkeypatch.setattr(reranker, "_cross_encoder_available", True)
    monkeypatch.setattr(reranker, "_LOADED_MODELS", {})
    monkeypatch.setenv("TRW_OFFLINE", "1")
    assert reranker._get_model("some/model") is None
    assert calls[-1]["local_files_only"] is True
    assert reranker.cross_encode_scores("q", _entries(), model_name="some/model") is None
    monkeypatch.delenv("TRW_OFFLINE")
    assert reranker._get_model("some/model", local_only=True) is None
    assert calls[-1]["local_files_only"] is True
    reranker._get_model("some/model")  # online: a network-capable load is allowed
    assert calls[-1]["local_files_only"] is False


def test_min_keep_must_be_at_least_one() -> None:
    import pytest

    with pytest.raises(ValueError):
        MemoryConfig(recall_rerank_min_keep=0)
