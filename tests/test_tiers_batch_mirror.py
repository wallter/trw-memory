"""Batched tier mirroring: one sidecar read-modify-write per recall, not per row.

Recall mirrors every returned row back into the hot/warm tiers to refresh
``last_accessed_at``. Done one row at a time, each mirror re-parsed (and, for
an entry already present, rewrote) the whole warm JSONL sidecar, so a
50-result recall over a few hundred rows spent ~80% of its latency there.
These tests pin the batch path to the sequential semantics it replaced.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from trw_memory.lifecycle.tiers._manager import TierManager
from trw_memory.lifecycle.tiers._runtime import get_tier_manager, remember_entries_data_in_tiers
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry


def _entry(i: int, ns: str = "project:t") -> MemoryEntry:
    return MemoryEntry(id=f"M-{i:04d}", content=f"content {i}", namespace=ns, tags=["t"])


def _sidecar_rows(store: WarmTierStore) -> dict[str, dict[str, object]]:
    path = store._warm_sidecar_path()
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {str(r["id"]): r for r in rows}


class TestWarmAddMany:
    def test_matches_sequential_warm_add(self, tmp_path: Path) -> None:
        seq = WarmTierStore(tmp_path / "seq")
        batch = WarmTierStore(tmp_path / "batch")
        entries = [_entry(i) for i in range(6)]
        for e in entries[:4]:  # pre-existing rows in both stores
            seq.warm_add(e.id, e.model_dump(mode="json"), None)
            batch.warm_add(e.id, e.model_dump(mode="json"), None)
        # refresh two existing rows + add two new ones, one id duplicated
        updates = [(e.id, {**e.model_dump(mode="json"), "content": f"fresh {e.id}"}, None) for e in entries[2:6]]
        updates.append((entries[5].id, {**entries[5].model_dump(mode="json"), "content": "last write"}, None))
        for entry_id, data, emb in updates:
            seq.warm_add(entry_id, data, emb)
        batch.warm_add_many(updates)
        assert _sidecar_rows(batch) == _sidecar_rows(seq)
        assert _sidecar_rows(batch)[entries[5].id]["summary"] == "last write"
        assert len(_sidecar_rows(batch)) == 6

    def test_batch_parses_sidecar_once(self, tmp_path: Path) -> None:
        store = WarmTierStore(tmp_path)
        entries = [_entry(i) for i in range(20)]
        store.warm_add_many([(e.id, e.model_dump(mode="json"), None) for e in entries])
        store._sidecar_cache.invalidate()  # as if another process wrote last
        with patch.object(WarmTierStore, "_parse_sidecar_records", wraps=store._parse_sidecar_records) as spy:
            store.warm_add_many([(e.id, e.model_dump(mode="json"), None) for e in entries])
        assert spy.call_count == 1

    def test_empty_batch_is_a_noop(self, tmp_path: Path) -> None:
        store = WarmTierStore(tmp_path)
        store.warm_add_many([])
        assert not store._warm_sidecar_path().exists()


class TestHotPutMany:
    def test_evicts_lru_and_returns_evictees_without_demoting(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path), hot_max_entries=3)
        manager = TierManager(tmp_path, cfg)
        entries = [_entry(i) for i in range(5)]
        with patch.object(manager, "warm_add") as warm_add:
            evicted = manager.hot_put_many([(e.id, e) for e in entries])
        warm_add.assert_not_called()
        assert [eid for eid, _ in evicted] == ["M-0000", "M-0001"]
        assert list(manager._hot) == ["M-0002", "M-0003", "M-0004"]

    def test_refresh_moves_entry_to_mru(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path), hot_max_entries=3)
        manager = TierManager(tmp_path, cfg)
        entries = [_entry(i) for i in range(3)]
        manager.hot_put_many([(e.id, e) for e in entries])
        evicted = manager.hot_put_many([(entries[0].id, entries[0]), (_entry(9).id, _entry(9))])
        assert [eid for eid, _ in evicted] == ["M-0001"]
        assert list(manager._hot) == ["M-0002", "M-0000", "M-0009"]


class TestRememberEntriesDataInTiers:
    def test_one_warm_write_per_namespace_including_evictees(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path), hot_max_entries=2)
        payloads = [_entry(i).model_dump(mode="json") for i in range(4)]
        payloads.append(_entry(7, ns="project:other").model_dump(mode="json"))
        with patch.object(TierManager, "warm_add_many", autospec=True) as warm_add_many:
            remember_entries_data_in_tiers(cfg, payloads)
        by_ns = {call.args[0]._namespace: call.args[1] for call in warm_add_many.call_args_list}
        assert set(by_ns) == {"project:t", "project:other"}
        ids = [item[0] for item in by_ns["project:t"]]
        # evictees (M-0000, M-0001 fell out of a 2-slot hot tier) precede the recalled rows
        assert ids[:2] == ["M-0000", "M-0001"]
        assert ids[2:] == ["M-0000", "M-0001", "M-0002", "M-0003"]
        assert [item[0] for item in by_ns["project:other"]] == ["M-0007"]

    def test_invalid_payload_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path))
        with patch.object(TierManager, "warm_add_many", autospec=True) as warm_add_many:
            remember_entries_data_in_tiers(cfg, [{"id": "bad"}, _entry(1).model_dump(mode="json")])
        assert warm_add_many.call_count == 1


class TestSidecarParseCache:
    def test_repeat_reads_skip_parsing_until_the_file_changes(self, tmp_path: Path) -> None:
        store = WarmTierStore(tmp_path)
        entries = [_entry(i) for i in range(5)]
        store.warm_add_many([(e.id, e.model_dump(mode="json"), None) for e in entries])
        path = store._warm_sidecar_path()
        with patch.object(WarmTierStore, "_parse_sidecar_records", wraps=store._parse_sidecar_records) as parse:
            store._sidecar_cache.invalidate()  # drop the writer's re-seed so the first read parses
            first = list(store._iter_sidecar_records(path))
            second = list(store._iter_sidecar_records(path))
            assert parse.call_count == 1
            assert [r for _, r in first] == [r for _, r in second]
            # annotating a yielded record must not leak into the next read
            second[0][1]["_tier_relevance"] = 0.9
            third = list(store._iter_sidecar_records(path))
            assert "_tier_relevance" not in third[0][1]
            # this store's own write re-seeds the cache: no re-parse (see test_warm_sidecar_cache.py)
            store.warm_add(_entry(9).id, _entry(9).model_dump(mode="json"), None)
            fourth = list(store._iter_sidecar_records(path))
        assert parse.call_count == 1
        assert len(fourth) == 6

    def test_evictees_survive_a_failed_warm_write(self, tmp_path: Path) -> None:
        """B-1 (release-verify 2026-09-17): evictees leave hot only once warm has them."""
        cfg = MemoryConfig(storage_path=str(tmp_path), hot_max_entries=2)
        payloads = [_entry(i).model_dump(mode="json") for i in range(4)]
        with patch.object(TierManager, "warm_add_many", autospec=True, side_effect=OSError("disk full")):
            remember_entries_data_in_tiers(cfg, payloads)
        manager = get_tier_manager(cfg, "project:t")
        assert set(manager._hot) == {"M-0000", "M-0001", "M-0002", "M-0003"}

        # a REAL warm write (no patch) drops the evictees back to capacity and
        # the evictees are readable from the warm sidecar afterwards
        remember_entries_data_in_tiers(cfg, [_entry(9).model_dump(mode="json")])
        assert len(manager._hot) == 2
        assert "M-0009" in manager._hot
        rows = _sidecar_rows(manager._warm_store)
        assert {"M-0000", "M-0001", "M-0002", "M-0009"} <= set(rows)

    def test_a_value_error_from_warm_keeps_evictees_too(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path), hot_max_entries=1)
        payloads = [_entry(i).model_dump(mode="json") for i in range(3)]
        with patch.object(TierManager, "warm_add_many", autospec=True, side_effect=ValueError("bad row")):
            remember_entries_data_in_tiers(cfg, payloads)
        assert len(get_tier_manager(cfg, "project:t")._hot) == 3
