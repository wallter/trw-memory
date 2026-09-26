"""Confidence-bounded recall: the cross-encoder cutoff after re-ranking."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval._adaptive_floor import RERANK_MIN_SCORE, adaptive_rerank_floor
from trw_memory.retrieval.pipeline import hybrid_search
from trw_memory.security.namespace_scope import authorize_namespaces
from trw_memory.security.rbac import Permission

NS = "project:cut"


def _entries() -> list[MemoryEntry]:
    return [MemoryEntry(id=f"e{i}", content=f"pydantic note {i}", namespace=NS, tags=[]) for i in range(8)]


def _scope():
    return authorize_namespaces(MemoryConfig(rbac_enabled=False), [NS], Permission.READ, "test")


def _scored(query, entries, *, model_name):
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


def test_rerank_knobs_are_no_longer_config_fields() -> None:
    """PRD-CORE-284 FR04: the floor is computed, not configured."""
    for field in ("recall_rerank", "recall_rerank_min_score", "recall_rerank_min_keep"):
        assert field not in MemoryConfig.model_fields


@pytest.mark.parametrize("limit", range(1, 11))
def test_adaptive_rerank_floor_matches_legacy_min_keep_for_limit_1_to_10(limit: int) -> None:
    """PRD-CORE-284 FR01/NFR01: exhaustive identity with the legacy fixed min_keep=5."""
    floor = adaptive_rerank_floor(limit)
    assert floor == (-8.0, 5)
    assert floor.min_score == RERANK_MIN_SCORE == -8.0
    assert floor.min_keep == 5


@pytest.mark.parametrize(
    ("limit", "min_keep"),
    # 11 and 13 are the first limits where ceil(limit/2) passes 5 (the PRD's
    # switch-matrix row "13 -> 5" contradicts its own formula; the formula wins).
    [(11, 6), (13, 7), (25, 13), (50, 25), (100, 50), (0, 0), (-3, 0)],
)
def test_adaptive_rerank_floor_scales_above_ten_and_is_empty_below_one(limit: int, min_keep: int) -> None:
    assert adaptive_rerank_floor(limit) == (-8.0, min_keep)


def _recall_client(tmp_path, monkeypatch, n: int) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "mem"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    client = MemoryClient(namespace="default", mode="local")
    backend = client._get_backend()
    for i in range(n):
        backend.store(MemoryEntry(id=f"r{i:02d}", content=f"zebra fact number {i} about stripes", namespace="default"))
    return client


@pytest.mark.parametrize(("limit", "expected"), [(10, 5), (50, 25), (3, 3)])
async def test_recall_keeps_the_adaptive_min_keep_when_everything_is_below_the_floor(
    tmp_path, monkeypatch, limit: int, expected: int
) -> None:
    """End to end: MemoryClient.recall(limit) returns exactly adaptive min_keep rows
    when the cross-encoder scores every candidate below -8."""
    client = _recall_client(tmp_path, monkeypatch, 60)

    def all_low(query, entries, **kwargs):
        return [(e, -20.0 - i * 0.01) for i, e in enumerate(entries)]

    with patch("trw_memory.retrieval.reranker.cross_encode_scores", side_effect=all_low):
        rows = await client.recall("zebra stripes", limit=limit, include_org_memories=False)
    assert len(rows) == expected


async def test_unavailable_cross_encoder_on_recall_returns_the_full_limit(tmp_path, monkeypatch) -> None:
    """FR03: with the model uncached, recall keeps fusion order and
    count -- the automatic "off" path, with no floor applied."""
    from trw_memory.retrieval import reranker

    client = _recall_client(tmp_path, monkeypatch, 60)

    class Uncached:
        def __init__(self, name, **kwargs):
            assert kwargs["local_files_only"] is True  # runtime: never a download
            raise OSError("not in local cache")

    monkeypatch.setattr(reranker, "_import_cross_encoder", lambda: True)
    monkeypatch.setattr(reranker, "_cross_encoder_cls", Uncached)
    monkeypatch.setattr(reranker, "_LOADED_MODELS", {})
    rows = await client.recall("zebra stripes", limit=50, include_org_memories=False)
    assert len(rows) == 50


def test_confidence_bounded_merge_holds_tier_rows_to_the_same_floor() -> None:
    from trw_memory._client_recall_helpers import merge_local_candidates
    from trw_memory.retrieval.recall_selection import LocalCandidate

    cfg = MemoryConfig()
    local = [LocalCandidate(MemoryEntry(id="h1", content="hybrid hit", namespace=NS), 1.0)]
    noise = LocalCandidate(MemoryEntry(id="w1", content="warm noise", namespace=NS), 0.4)
    good = LocalCandidate(MemoryEntry(id="w2", content="warm good", namespace=NS), 0.4)
    cold = LocalCandidate(MemoryEntry(id="c1", content="archived hit", namespace=NS), 0.3, cold=True)
    table = {"w1": -10.0, "w2": 1.0}

    def scores(query, entries, *, model_name):
        return sorted(((e, table[e.id]) for e in entries), key=lambda x: x[1], reverse=True)

    with patch("trw_memory.retrieval.reranker.cross_encode_scores", side_effect=scores):
        merged = merge_local_candidates(local, [noise, good, cold], 10, ["hit"], cfg, None, query="hit")
    assert [c.entry.id for c in merged] == ["h1", "w2", "c1"]
    assert all(c.tier_fallback for c in merged[1:])
    # legacy callers (no query) keep every tier row
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


def test_the_reranker_loads_cache_only_and_an_uncached_model_keeps_fusion_order(monkeypatch) -> None:
    """PLAN W40: a runtime load never downloads, whatever the environment says."""
    from trw_memory.retrieval import reranker

    calls: list[dict] = []

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            calls.append(kwargs)
            raise OSError("not in local cache")  # what huggingface_hub raises with local_files_only

    monkeypatch.setattr(reranker, "_cross_encoder_cls", FakeCrossEncoder)
    monkeypatch.setattr(reranker, "_cross_encoder_available", True)
    monkeypatch.setattr(reranker, "_LOADED_MODELS", {})
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    assert reranker._get_model("some/model") is None
    from trw_memory.embeddings.local import INFERENCE_DEVICE

    assert calls == [{"max_length": 512, "revision": "main", "local_files_only": True, "device": INFERENCE_DEVICE}]
    assert reranker.cross_encode_scores("q", _entries(), model_name="some/model") is None


def test_concurrent_first_loads_construct_the_cross_encoder_once(monkeypatch) -> None:
    """W27: two recalls at daemon start must not each load the re-ranker."""
    import threading

    from trw_memory.retrieval import reranker

    built: list[str] = []
    both_waiting = threading.Barrier(2)

    class SlowCrossEncoder:
        def __init__(self, name, **kwargs):
            built.append(name)
            threading.Event().wait(0.05)  # long enough for an unlocked second caller to enter

    monkeypatch.setattr(reranker, "_cross_encoder_cls", SlowCrossEncoder)
    monkeypatch.setattr(reranker, "_cross_encoder_available", True)
    monkeypatch.setattr(reranker, "_LOADED_MODELS", {})
    loaded: list[object] = []

    def load() -> None:
        both_waiting.wait(timeout=10)
        loaded.append(reranker._get_model("some/model"))

    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert built == ["some/model"]
    assert len(loaded) == 2 and loaded[0] is loaded[1]
