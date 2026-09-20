"""Tests that new MemoryConfig retrieval fields are forwarded to hybrid_search.

Verifies the wiring added in _client_recall_hybrid.py: recency_weight,
recency_halflife_days, fusion_mode, validity_age_decay, rerank_model,
rerank_candidates are passed through from config; rerank is unconditional and
the confidence floor comes from ``adaptive_rerank_floor`` (PRD-CORE-284).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.config import MemoryConfig


@pytest.fixture()
def wired_client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "mem"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


class TestHybridSearchConfigWiring:
    """Config fields for dead-code wiring must reach hybrid_search."""

    async def test_all_new_fields_have_sensible_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.recall_recency_weight == pytest.approx(0.0)
        assert cfg.recall_recency_halflife_days == pytest.approx(14.0)
        assert cfg.recall_fusion_mode == "rrf"
        assert cfg.recall_validity_age_decay is True
        assert "recall_rerank" not in MemoryConfig.model_fields  # PRD-CORE-284: always on
        assert cfg.recall_rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert cfg.recall_rerank_candidates == 50

    async def test_recency_weight_forwarded_from_config(self, wired_client: MemoryClient) -> None:
        await wired_client.store("recent memory for test")
        wired_client._config.recall_recency_weight = 0.7

        captured: dict = {}
        from trw_memory.retrieval import pipeline as _pipeline_mod

        original = _pipeline_mod.hybrid_search

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with patch.object(_pipeline_mod, "hybrid_search", side_effect=spy):
            await wired_client.recall("memory")

        assert "recency_weight" in captured
        assert captured["recency_weight"] == pytest.approx(0.7)

    async def test_recency_halflife_days_forwarded_from_config(self, wired_client: MemoryClient) -> None:
        await wired_client.store("halflife test entry")
        wired_client._config.recall_recency_weight = 0.3
        wired_client._config.recall_recency_halflife_days = 7.0

        captured: dict = {}
        from trw_memory.retrieval import pipeline as _pipeline_mod

        original = _pipeline_mod.hybrid_search

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with patch.object(_pipeline_mod, "hybrid_search", side_effect=spy):
            await wired_client.recall("halflife")

        assert captured.get("recency_halflife_days") == pytest.approx(7.0)

    async def test_fusion_mode_forwarded_from_config(self, wired_client: MemoryClient) -> None:
        await wired_client.store("fusion mode test entry")
        wired_client._config.recall_fusion_mode = "combmax"

        captured: dict = {}
        from trw_memory.retrieval import pipeline as _pipeline_mod

        original = _pipeline_mod.hybrid_search

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with patch.object(_pipeline_mod, "hybrid_search", side_effect=spy):
            await wired_client.recall("fusion mode")

        assert captured.get("fusion_mode") == "combmax"

    async def test_rerank_forwarded_from_config(self, wired_client: MemoryClient) -> None:
        await wired_client.store("rerank test entry one")
        await wired_client.store("rerank test entry two")
        wired_client._config.recall_rerank_candidates = 20

        captured: dict = {}
        from trw_memory.retrieval import pipeline as _pipeline_mod

        original = _pipeline_mod.hybrid_search

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        # Exercise the real hybrid path, stopping only at model execution: this
        # test owns configuration forwarding, not downloading a cross-encoder.
        with (
            patch.object(_pipeline_mod, "hybrid_search", side_effect=spy),
            patch(
                "trw_memory.retrieval.reranker.cross_encode_scores",
                side_effect=lambda query, entries, **kwargs: [(e, 1.0) for e in entries],
            ) as reranker,
        ):
            await wired_client.recall("rerank test")

        assert captured.get("rerank") is True
        assert captured.get("rerank_candidates") == 20
        reranker.assert_called_once()
        assert reranker.call_args.args[0] == "rerank test"
        assert reranker.call_args.args[1]
        assert len(reranker.call_args.args[1]) <= 20
        assert reranker.call_args.kwargs["model_name"] == wired_client._config.recall_rerank_model

    async def test_validity_age_decay_forwarded_from_config(self, wired_client: MemoryClient) -> None:
        await wired_client.store("validity decay test entry")
        wired_client._config.recall_validity_age_decay = True

        captured: dict = {}
        from trw_memory.retrieval import pipeline as _pipeline_mod

        original = _pipeline_mod.hybrid_search

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with patch.object(_pipeline_mod, "hybrid_search", side_effect=spy):
            await wired_client.recall("validity decay")

        assert captured.get("validity_age_decay") is True

    async def test_rerank_model_forwarded_from_config(self, wired_client: MemoryClient) -> None:
        custom_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        await wired_client.store("rerank model test entry")
        wired_client._config.recall_rerank_model = custom_model

        captured: dict = {}
        from trw_memory.retrieval import pipeline as _pipeline_mod

        original = _pipeline_mod.hybrid_search

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with (
            patch.object(_pipeline_mod, "hybrid_search", side_effect=spy),
            patch(
                "trw_memory.retrieval.reranker.cross_encode_scores",
                side_effect=lambda query, entries, **kwargs: [(e, 1.0) for e in entries],
            ) as reranker,
        ):
            await wired_client.recall("rerank model")

        assert captured.get("rerank_model") == custom_model
        assert captured.get("rerank") is True
        reranker.assert_called_once()
        assert reranker.call_args.args[0] == "rerank model"
        assert reranker.call_args.args[1]
        assert reranker.call_args.kwargs["model_name"] == custom_model


async def test_recall_uses_the_adaptive_floor_helper(wired_client: MemoryClient) -> None:
    """PRD-CORE-284 FR02: every floor consumer derives it from ``adaptive_rerank_floor``.

    Patching the helper alone -- no config mutation -- must move the SDK hybrid
    call AND the tier-merge refill bound; the bundled ``memory_recall`` tool
    path reranks unconditionally and (as before) applies no floor.
    """
    from trw_memory._client_recall_helpers import merge_local_candidates
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.retrieval import _adaptive_floor
    from trw_memory.retrieval import pipeline as _pipeline_mod
    from trw_memory.retrieval.recall_selection import LocalCandidate
    from trw_memory.security.namespace_scope import authorize_namespaces
    from trw_memory.security.rbac import Permission
    from trw_memory.tools._recall_retrieval import build_scored_candidates

    seen_limits: list[int] = []

    def fake_floor(limit: int) -> _adaptive_floor.RerankFloor:
        seen_limits.append(limit)
        return _adaptive_floor.RerankFloor(-3.0, 2)

    await wired_client.store("adaptive floor entry one")
    await wired_client.store("adaptive floor entry two")
    captured: dict = {}
    original = _pipeline_mod.hybrid_search

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    def all_high(query, entries, **kwargs):
        return [(e, 1.0) for e in entries]

    with (
        patch.object(_adaptive_floor, "adaptive_rerank_floor", side_effect=fake_floor),
        patch.object(_pipeline_mod, "hybrid_search", side_effect=spy),
        patch("trw_memory.retrieval.reranker.cross_encode_scores", side_effect=all_high),
    ):
        await wired_client.recall("adaptive floor", limit=7)
    assert 7 in seen_limits
    assert captured["rerank"] is True
    assert (captured["rerank_min_score"], captured["rerank_min_keep"]) == (-3.0, 2)

    # Tier-merge refill: a warm row scoring -5 clears the real -8 floor but not the patched -3.
    ns = "project:floor"
    local = [LocalCandidate(MemoryEntry(id="h1", content="hybrid hit", namespace=ns), 1.0)]
    warm = LocalCandidate(MemoryEntry(id="w1", content="warm row", namespace=ns), 0.4)
    cfg = MemoryConfig()

    def minus_five(query, entries, **kwargs):
        return [(e, -5.0) for e in entries]

    with patch("trw_memory.retrieval.reranker.cross_encode_scores", side_effect=minus_five):
        real = merge_local_candidates(local, [warm], 7, ["row"], cfg, None, query="row")
        with patch.object(_adaptive_floor, "adaptive_rerank_floor", side_effect=fake_floor):
            patched = merge_local_candidates(local, [warm], 7, ["row"], cfg, None, query="row")
    assert [c.entry.id for c in real] == ["h1", "w1"]
    assert [c.entry.id for c in patched] == ["h1"]

    # memory_recall tool path: rerank=True literal, no floor keywords.
    entry = MemoryEntry(id="t1", content="tool path entry", namespace=ns)
    scope = authorize_namespaces(MemoryConfig(rbac_enabled=False), [ns], Permission.READ, "test")
    with patch("trw_memory.tools.recall.hybrid_search_scored", return_value=[]) as scored_mock:
        build_scored_candidates(
            "tool path", [entry], cfg=cfg, scope=scope, embedder=None, stored_embeddings={}, limit=7, tags=None
        )
    kwargs = scored_mock.call_args.kwargs
    assert kwargs["rerank"] is True
    assert "rerank_min_score" not in kwargs and "rerank_min_keep" not in kwargs
