"""Warm sidecar cache: a write re-seeds it, so recall's access-time mirror is not a re-parse.

Recall mirrors every returned row's ``last_accessed_at`` into the warm JSONL
sidecar. The parsed-row cache is keyed on the file's ``(mtime_ns, size)``, so
before 2026-09-18 that rewrite forced the NEXT recall to re-parse the whole
file. Every test here holds the same oracle: what the cache serves must equal a
fresh parse of the bytes on disk, line numbers included.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from trw_memory.client import MemoryClient
from trw_memory.lifecycle.tiers._runtime import get_tier_manager
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.lifecycle.tiers._warm_sidecar_cache import parse_sidecar
from trw_memory.models.memory import MemoryEntry

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _payload(i: int, *, accessed: datetime = T0, content: str = "") -> dict[str, object]:
    entry = MemoryEntry(id=f"M-{i:04d}", content=content or f"content {i}", namespace="project:t", tags=["t"])
    data = entry.model_dump(mode="json")
    data["last_accessed_at"] = accessed.isoformat()
    return data


def _seeded(tmp_path: Path, n: int = 20) -> WarmTierStore:
    store = WarmTierStore(tmp_path)
    store.warm_add_many([(f"M-{i:04d}", _payload(i), None) for i in range(n)])
    return store


def _served(store: WarmTierStore) -> list[tuple[int, dict[str, object]]]:
    return list(store._iter_sidecar_records(store._warm_sidecar_path()))


def _on_disk(store: WarmTierStore) -> list[tuple[int, dict[str, object]]]:
    return parse_sidecar(store._warm_sidecar_path()).rows


def _parses(store: WarmTierStore) -> Any:
    """Count full parses; the wrapped call still reads the file."""
    return patch.object(WarmTierStore, "_parse_sidecar_records", wraps=store._parse_sidecar_records)


def _accessed(store: WarmTierStore, entry_id: str) -> object:
    return next(e for e in store.warm_entries() if e["id"] == entry_id)["last_accessed_at"]


class TestOwnWriteReseedsTheCache:
    def test_access_time_rewrite_is_a_cache_hit_that_matches_disk(self, tmp_path: Path) -> None:
        store = _seeded(tmp_path)
        _served(store)
        later = T0 + timedelta(days=3)
        with _parses(store) as parse:
            store.warm_add_many([(f"M-{i:04d}", _payload(i, accessed=later), None) for i in (2, 5, 7)])
            served = _served(store)
            assert _accessed(store, "M-0005") == later.isoformat()
            assert _accessed(store, "M-0001") == T0.isoformat()
        assert parse.call_count == 0
        assert served == _on_disk(store)

    def test_first_write_seeds_the_cache(self, tmp_path: Path) -> None:
        store = WarmTierStore(tmp_path)
        with _parses(store) as parse:
            store.warm_add_many([(f"M-{i:04d}", _payload(i), None) for i in range(3)])
            served = _served(store)
        assert parse.call_count == 0
        assert served == _on_disk(store)

    def test_append_keeps_disk_line_numbers_past_corrupt_and_blank_lines(self, tmp_path: Path) -> None:
        store = _seeded(tmp_path, n=2)
        sidecar = store._warm_sidecar_path()
        with sidecar.open("a", encoding="utf-8") as fh:
            fh.write("{not json\n\n[1, 2]\n")
        _served(store)
        with _parses(store) as parse:
            store.warm_add_many([("M-0100", _payload(100), None), ("M-0101", _payload(101), None)])
            served = _served(store)
        assert parse.call_count == 0
        assert served == _on_disk(store)
        assert [line for line, rec in served if rec["id"] in {"M-0100", "M-0101"}] == [6, 7]

    def test_append_to_a_file_without_a_trailing_newline_terminates_it(self, tmp_path: Path) -> None:
        store = WarmTierStore(tmp_path)
        sidecar = store._warm_sidecar_path()
        sidecar.write_text(json.dumps({"id": "M-0000", "summary": "s", "tags": [], "entry": _payload(0)}))
        _served(store)
        with _parses(store) as parse:
            store.warm_add_many([("M-0001", _payload(1), None)])
            served = _served(store)
        # the writer knows the tail is unterminated and fixes it, so the cache extends
        assert parse.call_count == 0
        assert served == _on_disk(store)
        assert [rec["id"] for _, rec in served] == ["M-0000", "M-0001"]

    def test_purge_reseeds_the_cache(self, tmp_path: Path) -> None:
        store = _seeded(tmp_path, n=5)
        _served(store)
        with _parses(store) as parse:
            assert store.purge_sidecar_entry("M-0003") is True
            served = _served(store)
        assert parse.call_count == 0
        assert served == _on_disk(store)
        assert "M-0003" not in {rec["id"] for _, rec in served}

    def test_cached_rows_do_not_alias_the_callers_payload(self, tmp_path: Path) -> None:
        store = WarmTierStore(tmp_path)
        payload = _payload(0)
        store.warm_add_many([("M-0000", payload, None)])
        tags = payload["tags"]
        assert isinstance(tags, list)
        tags.append("mutated-after-write")
        assert _served(store) == _on_disk(store)


class TestForeignWriteInvalidates:
    """Two stores on one directory stand in for two processes sharing a store."""

    def test_each_process_sees_the_others_access_time_mirror(self, tmp_path: Path) -> None:
        a = _seeded(tmp_path)
        b = WarmTierStore(tmp_path)
        _served(a)
        _served(b)
        later = T0 + timedelta(days=1)
        # a different content length changes the size, so the key change does not
        # depend on the filesystem's timestamp resolution
        b.warm_add_many([("M-0004", _payload(4, accessed=later, content="rewritten by b"), None)])
        with _parses(a) as parse:
            assert _accessed(a, "M-0004") == later.isoformat()
        assert parse.call_count == 1
        assert _served(a) == _on_disk(a)

        latest = T0 + timedelta(days=2)
        a.warm_add_many([("M-0009", _payload(9, accessed=latest, content="rewritten by a"), None)])
        with _parses(b) as parse:
            assert _accessed(b, "M-0009") == latest.isoformat()
            assert _accessed(b, "M-0004") == later.isoformat()
        assert parse.call_count == 1

    def test_a_raw_external_append_is_seen(self, tmp_path: Path) -> None:
        store = _seeded(tmp_path, n=3)
        _served(store)
        record = {"id": "M-0777", "summary": "external", "tags": [], "entry": _payload(777)}
        with store._warm_sidecar_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        with _parses(store) as parse:
            served = _served(store)
        assert parse.call_count == 1
        assert "M-0777" in {rec["id"] for _, rec in served}


class TestRecallPath:
    async def test_recall_access_mirror_leaves_the_next_recall_a_cache_hit(self, memory_client: MemoryClient) -> None:
        with patch.object(memory_client, "_get_embedder", return_value=None):
            for i in range(6):
                await memory_client.store(content=f"sunrise painting lesson number {i}", tags=["art"])
            await memory_client.recall("sunrise painting", limit=5)
            warm = get_tier_manager(memory_client._config, "default")._warm_store
            before = datetime.now(timezone.utc)
            with _parses(warm) as parse:
                first = await memory_client.recall("sunrise painting", limit=5)
                await memory_client.recall("painting lesson", limit=5)
        assert first
        assert parse.call_count == 0
        disk = {rec["id"]: rec for _, rec in _on_disk(warm)}
        assert _served(warm) == _on_disk(warm)
        # access statistics still land on disk, where promotion and other processes read them
        for result in first:
            entry = disk[result["memory_id"]]["entry"]
            assert isinstance(entry, dict)
            assert datetime.fromisoformat(str(entry["last_accessed_at"])) >= before
        await memory_client.close()
