"""Tests for lifecycle/tiers.py hot and warm tier behavior."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from trw_memory.lifecycle.tiers import TierManager
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus

from ._test_tiers_support import cfg as _cfg_fixture  # noqa: F401
from ._test_tiers_support import mem_dir as _mem_dir_fixture  # noqa: F401
from ._test_tiers_support import mgr as _mgr_fixture  # noqa: F401


def _make_entry(
    entry_id: str = "test-id",
    content: str | None = None,
    importance: float = 0.5,
    status: str = "active",
    days_old: int = 0,
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    last_accessed_at = now - timedelta(days=days_old)
    return MemoryEntry(
        id=entry_id,
        content=content or f"content for {entry_id}",
        detail="some detail",
        tags=["tag1"],
        importance=importance,
        status=MemoryStatus(status),
        last_accessed_at=last_accessed_at,
        created_at=now,
        updated_at=now,
    )


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
        # trw-memory-2: warm_add failure drops the LRU evictee (e1) and preserves
        # the freshly written entry (e4) — previously the new write was lost
        # (see TierManager.hot_put + test_tier_thread_safety.py).
        assert mgr.hot_get("e1") is None
        assert mgr.hot_get("e4") is not None

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

    def test_search_merges_hot_warm_and_cold_results(self, mgr: TierManager) -> None:
        from trw_memory.storage.persistence import write_yaml

        mgr.hot_put("hot-entry", _make_entry("hot-entry", content="shared-token hot"))
        mgr.warm_add("warm-entry", {"id": "warm-entry", "content": "shared-token warm", "tags": []}, None)

        cold_partition = mgr._cold_dir() / "2026" / "03"
        cold_partition.mkdir(parents=True, exist_ok=True)
        write_yaml(cold_partition / "cold-entry.yaml", {"id": "cold-entry", "content": "shared-token cold", "tags": []})

        results = mgr.search(["shared-token"], top_k=10)
        ids = {str(result["id"]) for result in results}
        assert {"hot-entry", "warm-entry", "cold-entry"}.issubset(ids)


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

    def test_warm_search_score_is_cosine_from_l2_distance(
        self, mgr: TierManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """vec0 float[] returns L2 distance; for unit-normalized embeddings the
        score must be cosine = 1 - dist**2/2, not the prior 1 - dist (which
        under-scored every moderately-similar hit and could go negative)."""
        entry = _make_entry("e1", importance=0.5, days_old=0).model_dump(mode="json")
        entry["content"] = "shared token"
        mgr.warm_add("e1", entry, None)

        dist = 1.0  # L2 distance → cosine 1 - 1/2 = 0.5 (old formula gave 1-1=0.0)

        class _FakeBackend:
            def search_vectors(self, _q: list[float], top_k: int) -> list[tuple[str, float]]:
                return [("e1", dist)][:top_k]

        monkeypatch.setattr(mgr._warm_store, "_get_warm_backend", lambda dim=None: _FakeBackend())

        results = mgr.warm_search(["shared"], [1.0, 0.0], top_k=5)
        assert results, "in-sidecar hit should be returned"
        # _tier_relevance is the raw cosine the fix controls (it then feeds the
        # composite importance score). Assert it equals 1 - dist**2/2, not the
        # buggy 1 - dist.
        expected = 1.0 - dist * dist / 2.0
        assert results[0]["_tier_relevance"] == pytest.approx(expected)
        assert results[0]["_tier_relevance"] != pytest.approx(1.0 - dist)

    def test_warm_sidecar_corrupt_line_skipped_with_structured_event(self, mgr: TierManager) -> None:
        """A corrupt sidecar row must not raise: the valid entry still searches
        and lists, and a structured ``warm_tier_sidecar_corrupt_record_skipped``
        event is emitted with locality (path, line_number, error_class) but
        without the raw line contents."""
        from structlog.testing import capture_logs

        # Seed one valid record through the normal write path.
        mgr.warm_add("good", {"id": "good", "content": "python programming", "tags": ["code"]}, None)

        # Inject a corrupt JSON line *before* the valid one so the valid record
        # is on line 2 — exercises line-number locality, not just line 1.
        sidecar = mgr._warm_sidecar_path()
        valid_line = sidecar.read_text(encoding="utf-8").strip()
        sidecar.write_text("{not valid json,,,\n" + valid_line + "\n", encoding="utf-8")

        with capture_logs() as logs:
            search_results = mgr.warm_search(["python"], None)
            listed = mgr._warm_store.warm_entries()

        # Acceptance 1 + 3: valid entry still returned by search and list.
        assert "good" in [str(r["id"]) for r in search_results]
        assert "good" in [str(e.get("id")) for e in listed]

        # Acceptance 2 + 3: structured event emitted with locality, no payload.
        corrupt_events = [log for log in logs if log.get("event") == "warm_tier_sidecar_corrupt_record_skipped"]
        assert corrupt_events, "expected a structured corrupt-record event"
        event = corrupt_events[0]
        assert event["path"] == str(sidecar)
        assert event["line_number"] == 1
        assert event["error_class"] == "JSONDecodeError"
        # The raw corrupt line must never be logged.
        assert "not valid json" not in json.dumps(event)

    def test_warm_sidecar_non_utf8_row_between_valid_rows_does_not_abort(self, mgr: TierManager) -> None:
        """A non-UTF-8 byte row sandwiched between two valid records must not
        abort search/list/remove: both adjacent valid records survive, and the
        skip is logged content-free with a stable ``UnicodeDecodeError`` class
        and correct line-number locality (the bad row is line 2)."""
        from structlog.testing import capture_logs

        # Seed two valid records through the normal write path.
        mgr.warm_add("before", {"id": "before", "content": "python alpha", "tags": ["code"]}, None)
        mgr.warm_add("after", {"id": "after", "content": "python omega", "tags": ["code"]}, None)

        sidecar = mgr._warm_sidecar_path()
        valid_lines = sidecar.read_text(encoding="utf-8").strip().splitlines()
        assert len(valid_lines) == 2
        # Inject a lone continuation byte (0xff) that is not valid UTF-8 *between*
        # the two valid rows so the bad row is line 2 and a valid row follows it.
        bad_row = b"\xff\xfe not text \x80\x81"
        sidecar.write_bytes(
            valid_lines[0].encode("utf-8") + b"\n" + bad_row + b"\n" + valid_lines[1].encode("utf-8") + b"\n"
        )

        with capture_logs() as logs:
            search_results = mgr.warm_search(["python"], None)
            listed = mgr._warm_store.warm_entries()

        # Both adjacent valid records survive search and list.
        searched_ids = {str(r["id"]) for r in search_results}
        assert {"before", "after"} <= searched_ids
        listed_ids = {str(e.get("id")) for e in listed}
        assert {"before", "after"} <= listed_ids

        # The non-UTF-8 row is skipped with a content-free structured event.
        corrupt_events = [log for log in logs if log.get("event") == "warm_tier_sidecar_corrupt_record_skipped"]
        assert corrupt_events, "expected a structured corrupt-record event for the non-UTF-8 row"
        event = corrupt_events[0]
        assert event["path"] == str(sidecar)
        assert event["line_number"] == 2
        assert event["error_class"] == "UnicodeDecodeError"
        # No raw byte content may leak into the log payload.
        assert "not text" not in json.dumps(event)

        # remove() must still operate across the corrupt row.
        assert mgr._warm_store.warm_remove("before") is True
        remaining = {str(e.get("id")) for e in mgr._warm_store.warm_entries()}
        assert remaining == {"after"}

    def test_warm_sidecar_non_utf8_first_row_preserves_following_record(self, mgr: TierManager) -> None:
        """A non-UTF-8 row on line 1 must not hide a valid record on line 2."""
        mgr.warm_add("good", {"id": "good", "content": "python programming", "tags": ["x"]}, None)
        sidecar = mgr._warm_sidecar_path()
        valid_line = sidecar.read_text(encoding="utf-8").strip()
        sidecar.write_bytes(b"\xff\xff\xff\n" + valid_line.encode("utf-8") + b"\n")

        results = mgr.warm_search(["python"], None)
        assert "good" in {str(r["id"]) for r in results}

    def test_warm_remove_deletes_vector_rows(self, mgr: TierManager, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeBackend:
            def __init__(self) -> None:
                self.vector_deleted = False

            def delete(self, _entry_id: str, *, namespace: str = "default") -> bool:
                return False

            def delete_vector(self, _entry_id: str, *, namespace: str = "default") -> bool:
                self.vector_deleted = True
                return True

        fake_backend = _FakeBackend()
        monkeypatch.setattr(mgr._warm_store, "_get_warm_backend", lambda dim=None: fake_backend)

        mgr.warm_add("e1", {"id": "e1", "content": "x", "tags": []}, None)

        assert mgr.warm_remove("e1") is True
        assert fake_backend.vector_deleted is True
