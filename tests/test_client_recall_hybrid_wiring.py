"""Tests that new MemoryConfig retrieval fields are forwarded to hybrid_search.

Verifies the wiring added in _client_recall_hybrid.py: recency_weight,
recency_halflife_days, fusion_mode, validity_age_decay, rerank,
rerank_model, rerank_candidates are all passed through from config.
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
        assert cfg.recall_rerank is False
        assert cfg.recall_rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert cfg.recall_rerank_candidates == 50

    async def test_recency_weight_forwarded_from_config(
        self, wired_client: MemoryClient
    ) -> None:
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

    async def test_recency_halflife_days_forwarded_from_config(
        self, wired_client: MemoryClient
    ) -> None:
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

    async def test_fusion_mode_forwarded_from_config(
        self, wired_client: MemoryClient
    ) -> None:
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

    async def test_rerank_forwarded_from_config(
        self, wired_client: MemoryClient
    ) -> None:
        await wired_client.store("rerank test entry one")
        await wired_client.store("rerank test entry two")
        wired_client._config.recall_rerank = True
        wired_client._config.recall_rerank_candidates = 20

        captured: dict = {}
        from trw_memory.retrieval import pipeline as _pipeline_mod

        original = _pipeline_mod.hybrid_search

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with patch.object(_pipeline_mod, "hybrid_search", side_effect=spy):
            await wired_client.recall("rerank test")

        assert captured.get("rerank") is True
        assert captured.get("rerank_candidates") == 20

    async def test_validity_age_decay_forwarded_from_config(
        self, wired_client: MemoryClient
    ) -> None:
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

    async def test_rerank_model_forwarded_from_config(
        self, wired_client: MemoryClient
    ) -> None:
        custom_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        await wired_client.store("rerank model test entry")
        wired_client._config.recall_rerank = True
        wired_client._config.recall_rerank_model = custom_model

        captured: dict = {}
        from trw_memory.retrieval import pipeline as _pipeline_mod

        original = _pipeline_mod.hybrid_search

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with patch.object(_pipeline_mod, "hybrid_search", side_effect=spy):
            await wired_client.recall("rerank model")

        assert captured.get("rerank_model") == custom_model
