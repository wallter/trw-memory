"""Tests for lifecycle/tiers.py hot and warm tier behavior."""

from __future__ import annotations

import json

import pytest

from trw_memory.lifecycle.tiers import TierManager
from trw_memory.models.config import MemoryConfig

from ._test_tiers_support import _make_entry, cfg, mgr  # noqa: F401


class TestHotTier:
    def test_hot_get_miss_returns_none(self, mgr: TierManager) -> None:
        assert mgr.hot_get("nonexistent") is None

    def test_hot_put_and_get(self, mgr: TierManager) -> None:
        entry = _make_entry("e1")
        mgr.hot_put("e1", entry)
        result = mgr.hot_get("e1")
        assert result is not None
        assert result.id == "e1"

    def test_hot_get_moves_to_mru(self, mgr: TierManager) -> None:
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e2", _make_entry("e2"))
        mgr.hot_put("e3", _make_entry("e3"))
        mgr.hot_get("e1")
        mgr.hot_put("e4", _make_entry("e4"))
        assert mgr.hot_get("e1") is not None
        assert mgr.hot_get("e2") is None

    def test_hot_put_evicts_lru_when_over_capacity(self, mgr: TierManager) -> None:
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e2", _make_entry("e2"))
        mgr.hot_put("e3", _make_entry("e3"))
        assert mgr.hot_size == 3
        mgr.hot_put("e4", _make_entry("e4"))
        assert mgr.hot_size == 3
        assert mgr.hot_get("e1") is None

    def test_hot_capacity_eviction_demotes_entry_to_warm_tier(self, mgr: TierManager) -> None:
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e2", _make_entry("e2"))
        mgr.hot_put("e3", _make_entry("e3"))

        mgr.hot_put("e4", _make_entry("e4"))

        warm_results = mgr.warm_search(["e1"], None)
        warm_ids = [str(row["id"]) for row in warm_results]
        assert "e1" in warm_ids

    def test_hot_capacity_eviction_keeps_entry_when_warm_demote_fails(self, mgr: TierManager) -> None:
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e2", _make_entry("e2"))
        mgr.hot_put("e3", _make_entry("e3"))

        original_warm_add = mgr.warm_add

        def _fail_once(entry_id: str, entry_data: dict[str, object], embedding: list[float] | None) -> None:
            raise OSError("disk full")

        mgr.warm_add = _fail_once  # type: ignore[method-assign]
        try:
            mgr.hot_put("e4", _make_entry("e4"))
        finally:
            mgr.warm_add = original_warm_add  # type: ignore[method-assign]

        assert mgr.hot_size == 3
        assert mgr.hot_get("e1") is not None
        assert mgr.hot_get("e4") is None

    def test_hot_put_refresh_existing(self, mgr: TierManager) -> None:
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e1", _make_entry("e1", importance=0.9))
        result = mgr.hot_get("e1")
        assert result is not None
        assert result.importance == pytest.approx(0.9)

    def test_hot_clear(self, mgr: TierManager) -> None:
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e2", _make_entry("e2"))
        mgr.hot_clear()
        assert mgr.hot_size == 0
        assert mgr.hot_get("e1") is None

    def test_hot_size_property(self, mgr: TierManager) -> None:
        assert mgr.hot_size == 0
        mgr.hot_put("e1", _make_entry("e1"))
        assert mgr.hot_size == 1

    def test_hot_put_update_last_accessed(self, mgr: TierManager) -> None:
        mgr.hot_put("e1", _make_entry("e1"))
        mgr.hot_put("e2", _make_entry("e2"))
        mgr.hot_put("e3", _make_entry("e3"))
        mgr.hot_put("e4", _make_entry("e4"))
        assert mgr.hot_get("e1") is None

    def test_hot_get_refreshes_ttl_eligibility(self, mgr: TierManager, cfg: MemoryConfig) -> None:
        mgr.hot_put("e1", _make_entry("e1", days_old=30))

        refreshed = mgr.hot_get("e1")
        assert refreshed is not None

        result = mgr.sweep(config=cfg)

        assert result.demoted == 0
        assert mgr.hot_get("e1") is not None

    def test_warmup_hot_from_warm_ranks_full_sidecar_before_truncating(self, mgr: TierManager) -> None:
        for idx in range(10):
            mgr.warm_add(
                f"low-{idx}",
                _make_entry(f"low-{idx}", importance=0.1).model_dump(mode="json"),
                None,
            )
        mgr.warm_add(
            "best",
            _make_entry("best", importance=0.95).model_dump(mode="json"),
            None,
        )

        loaded = mgr.warmup_hot_from_warm(max_entries=1)
        assert loaded == 1
        assert mgr.hot_get("best") is not None


class TestWarmTier:
    def test_warm_add_and_sidecar_created(self, mgr: TierManager) -> None:
        entry_data: dict[str, object] = {"id": "e1", "content": "warm entry", "tags": ["x"]}
        mgr.warm_add("e1", entry_data, None)
        assert mgr._warm_sidecar_path().exists()

    def test_warm_sidecar_contains_entry(self, mgr: TierManager) -> None:
        entry_data: dict[str, object] = {"id": "e1", "content": "test warm", "tags": ["a"]}
        mgr.warm_add("e1", entry_data, None)
        text = mgr._warm_sidecar_path().read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        ids = [record["id"] for record in records]
        assert "e1" in ids

    def test_warm_add_upsert_replaces_existing(self, mgr: TierManager) -> None:
        mgr.warm_add("e1", {"id": "e1", "content": "old", "tags": []}, None)
        mgr.warm_add("e1", {"id": "e1", "content": "new", "tags": []}, None)
        records = [
            json.loads(line)
            for line in mgr._warm_sidecar_path().read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        e1_records = [record for record in records if record.get("id") == "e1"]
        assert len(e1_records) == 1
        assert e1_records[0]["summary"] == "new"

    def test_warm_remove_clears_sidecar(self, mgr: TierManager) -> None:
        mgr.warm_add("e1", {"id": "e1", "content": "remove me", "tags": []}, None)
        mgr.warm_remove("e1")
        sidecar = mgr._warm_sidecar_path()
        if sidecar.exists():
            records = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
            ids = [record.get("id") for record in records]
            assert "e1" not in ids

    def test_warm_keyword_search_finds_match(self, mgr: TierManager) -> None:
        mgr.warm_add("e1", {"id": "e1", "content": "python programming", "tags": ["code"]}, None)
        mgr.warm_add("e2", {"id": "e2", "content": "cooking recipes", "tags": ["food"]}, None)
        results = mgr.warm_search(["python"], None)
        ids = [result["id"] for result in results]
        assert "e1" in ids
        assert "e2" not in ids

    def test_warm_keyword_search_no_match_empty(self, mgr: TierManager) -> None:
        mgr.warm_add("e1", {"id": "e1", "content": "python programming", "tags": []}, None)
        assert mgr.warm_search(["ruby"], None) == []

    def test_warm_keyword_search_matches_detail_text(self, mgr: TierManager) -> None:
        mgr.warm_add("e1", {"id": "e1", "content": "opaque title", "detail": "detail-only-hit", "tags": []}, None)
        results = mgr.warm_search(["detail-only-hit"], None)
        ids = [result["id"] for result in results]
        assert "e1" in ids

    def test_warm_search_no_tokens_returns_empty(self, mgr: TierManager) -> None:
        mgr.warm_add("e1", {"id": "e1", "content": "any content", "tags": []}, None)
        assert mgr.warm_search([], None) == []

    def test_warm_search_by_tag(self, mgr: TierManager) -> None:
        mgr.warm_add("e1", {"id": "e1", "content": "x", "tags": ["mytag"]}, None)
        results = mgr.warm_search(["mytag"], None)
        ids = [result["id"] for result in results]
        assert "e1" in ids

    def test_warm_search_reranks_by_composite_score(self, mgr: TierManager) -> None:
        stale_low = _make_entry("low", importance=0.1, days_old=60).model_dump(mode="json")
        stale_low["content"] = "shared token"
        fresh_high = _make_entry("high", importance=0.9, days_old=0).model_dump(mode="json")
        fresh_high["content"] = "shared token"
        mgr.warm_add("low", stale_low, None)
        mgr.warm_add("high", fresh_high, None)

        results = mgr.warm_search(["shared"], None, top_k=2)

        ids = [str(result["id"]) for result in results]
        assert ids == ["high", "low"]

    def test_warm_search_skips_orphaned_vector_hits(self, mgr: TierManager, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeBackend:
            def search_vectors(self, _query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
                return [("orphaned-entry", 0.0)][:top_k]

        monkeypatch.setattr(mgr._warm_store, "_get_warm_backend", lambda dim=None: _FakeBackend())

        assert mgr.warm_search(["semantic"], [1.0, 0.0], top_k=5) == []

    def test_warm_remove_deletes_vector_rows(self, mgr: TierManager, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeBackend:
            def __init__(self) -> None:
                self.vector_deleted = False

            def delete(self, _entry_id: str) -> bool:
                return False

            def delete_vector(self, _entry_id: str) -> bool:
                self.vector_deleted = True
                return True

        fake_backend = _FakeBackend()
        monkeypatch.setattr(mgr._warm_store, "_get_warm_backend", lambda dim=None: fake_backend)

        mgr.warm_add("e1", {"id": "e1", "content": "x", "tags": []}, None)

        assert mgr.warm_remove("e1") is True
        assert fake_backend.vector_deleted is True
