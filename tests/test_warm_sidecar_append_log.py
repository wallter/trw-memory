"""Warm sidecar append log: recall's access mirror costs O(rows touched), not O(file).

Recall mirrors ``last_accessed_at`` (and the other access counters) for the k
rows it returned into the warm JSONL sidecar. That used to be a full rewrite:
a ``json.dumps`` of every row and O(file) bytes per recall. An access-only
refresh is now appended and supersedes the earlier row; readers see the last
row per id; content changes and accumulated debt still rewrite atomically.

Oracles used throughout: what a reader is served equals a fresh parse of the
bytes on disk, and equals the "last row per id" view an older trw-memory
reader builds from the same bytes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import trw_memory
from trw_memory.client import MemoryClient
from trw_memory.lifecycle.tiers import _warm_sidecar_cache
from trw_memory.lifecycle.tiers._runtime import get_tier_manager
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.lifecycle.tiers._warm_sidecar_cache import parse_sidecar
from trw_memory.models.memory import MemoryEntry

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _payload(i: int, *, accessed: datetime = T0, content: str = "", count: int = 0) -> dict[str, object]:
    entry = MemoryEntry(
        id=f"M-{i:04d}",
        content=content or f"content {i}",
        namespace="project:t",
        tags=["t"],
        created_at=T0,
        updated_at=T0,
        valid_from=T0,
    )
    data = entry.model_dump(mode="json")
    data["last_accessed_at"] = accessed.isoformat()
    data["access_count"] = count
    return data


def _seeded(tmp_path: Path, n: int = 20) -> WarmTierStore:
    store = WarmTierStore(tmp_path)
    store.warm_add_many([(f"M-{i:04d}", _payload(i), None) for i in range(n)])
    return store


def _touch(store: WarmTierStore, ids: list[int], accessed: datetime, count: int = 1) -> None:
    store.warm_add_many([(f"M-{i:04d}", _payload(i, accessed=accessed, count=count), None) for i in ids])


def _physical(store: WarmTierStore) -> list[dict[str, object]]:
    text = store._warm_sidecar_path().read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _old_reader_view(store: WarmTierStore) -> dict[str, object]:
    """What a pre-append-log reader builds: ``{id: entry}``, later lines winning."""
    return {str(r["id"]): r["entry"] for r in _physical(store)}


def _entries(store: WarmTierStore) -> list[dict[str, object]]:
    return store.warm_entries()


def _served(store: WarmTierStore) -> list[tuple[int, dict[str, object]]]:
    return list(store._iter_sidecar_records(store._warm_sidecar_path()))


class TestAccessRefreshAppends:
    def test_refresh_appends_only_the_touched_rows(self, tmp_path: Path) -> None:
        store = _seeded(tmp_path, n=50)
        sidecar = store._warm_sidecar_path()
        size_before = sidecar.stat().st_size
        later = T0 + timedelta(days=2)
        with patch("trw_memory.lifecycle.tiers._warm.os.replace") as replace:
            _touch(store, [3, 7], later)
        replace.assert_not_called()
        appended = sidecar.read_bytes()[size_before:].decode("utf-8").splitlines()
        assert [json.loads(line)["id"] for line in appended] == ["M-0003", "M-0007"]
        stamps = {e["id"]: e["last_accessed_at"] for e in _entries(store)}
        assert stamps["M-0003"] == later.isoformat()
        assert stamps["M-0004"] == T0.isoformat()
        assert len(stamps) == 50

    def test_live_view_matches_the_rewrite_model_and_older_readers(self, tmp_path: Path) -> None:
        store = _seeded(tmp_path, n=6)
        _touch(store, [1, 4], T0 + timedelta(days=1), count=1)
        _touch(store, [1], T0 + timedelta(days=2), count=2)
        # the old rewrite moved an updated row to the end: same order here
        assert [e["id"] for e in _entries(store)] == ["M-0000", "M-0002", "M-0003", "M-0005", "M-0004", "M-0001"]
        assert {e["id"]: e for e in _entries(store)} == _old_reader_view(store)
        assert _served(store) == parse_sidecar(store._warm_sidecar_path()).rows
        fresh = WarmTierStore(tmp_path)
        assert fresh.warm_entries() == _entries(store)
        assert _entries(store)[-1]["access_count"] == 2

    def test_keyword_search_does_not_return_a_superseded_copy(self, tmp_path: Path) -> None:
        store = _seeded(tmp_path, n=4)
        for day in range(1, 4):
            _touch(store, [2], T0 + timedelta(days=day))
        hits = store._warm_keyword_search(["content"], top_k=10)
        assert sorted(h["id"] for h in hits) == ["M-0000", "M-0001", "M-0002", "M-0003"]


class TestContentChangesRewrite:
    def test_changed_content_leaves_no_stale_copy_on_disk(self, tmp_path: Path) -> None:
        store = _seeded(tmp_path, n=5)
        _touch(store, [2], T0 + timedelta(days=1))  # appended: two physical copies of M-0002
        assert sum(r["id"] == "M-0002" for r in _physical(store)) == 2
        store.warm_add_many([("M-0002", _payload(2, content="redacted"), None)])
        copies = [r for r in _physical(store) if r["id"] == "M-0002"]
        assert len(copies) == 1
        assert "content 2" not in store._warm_sidecar_path().read_text(encoding="utf-8")
        assert _entries(store)[-1]["content"] == "redacted"

    def test_purge_removes_every_copy(self, tmp_path: Path) -> None:
        store = _seeded(tmp_path, n=5)
        _touch(store, [3], T0 + timedelta(days=1))
        _touch(store, [3], T0 + timedelta(days=2))
        assert store.purge_sidecar_entry("M-0003") is True
        assert "M-0003" not in {r["id"] for r in _physical(store)}
        assert store.purge_sidecar_entry("M-0003") is False
        assert _served(store) == parse_sidecar(store._warm_sidecar_path()).rows


class TestCompaction:
    def test_debt_past_the_threshold_compacts_and_keeps_the_live_view(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_warm_sidecar_cache, "COMPACT_MIN_DEAD", 8)
        store = _seeded(tmp_path, n=10)
        max_lines = 0
        for step in range(1, 40):
            _touch(store, [step % 10, (step * 3) % 10], T0 + timedelta(hours=step), count=step)
            max_lines = max(max_lines, len(_physical(store)))
            assert {e["id"]: e for e in _entries(store)} == _old_reader_view(store)
        # dead lines never exceed max(live, floor): the file stays within live + that bound
        assert max_lines <= 10 + max(10, 8) + 2
        parsed = parse_sidecar(store._warm_sidecar_path())
        assert parsed.dead <= max(len(parsed), 8)
        assert len(parsed) == 10
        latest = _entries(store)[-1]
        assert latest["access_count"] == 39

    def test_compaction_drops_corrupt_lines(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_warm_sidecar_cache, "COMPACT_MIN_DEAD", 2)
        store = _seeded(tmp_path, n=3)
        with store._warm_sidecar_path().open("a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        for day in range(1, 6):
            _touch(store, [0], T0 + timedelta(days=day))
        assert "{not json" not in store._warm_sidecar_path().read_text(encoding="utf-8")
        assert len(_entries(store)) == 3


class TestCrashSafety:
    def test_a_torn_append_loses_only_that_access_update(self, tmp_path: Path) -> None:
        store = _seeded(tmp_path, n=4)
        sidecar = store._warm_sidecar_path()
        # a crash mid-append: half of an access refresh for M-0001, no newline
        record = {"id": "M-0001", "summary": "content 1", "tags": ["t"], "entry": _payload(1, count=9)}
        with sidecar.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record)[:40])
        reader = WarmTierStore(tmp_path)
        by_id = {e["id"]: e for e in reader.warm_entries()}
        assert len(by_id) == 4
        assert by_id["M-0001"]["content"] == "content 1"
        assert by_id["M-0001"]["access_count"] == 0  # the torn update is lost, the row is not
        # the next write terminates the torn tail instead of fusing with it
        later = T0 + timedelta(days=5)
        _touch(reader, [2], later)
        assert _served(reader) == parse_sidecar(sidecar).rows
        assert {e["id"]: e["last_accessed_at"] for e in WarmTierStore(tmp_path).warm_entries()}["M-0002"] == (
            later.isoformat()
        )

    def test_a_rewrite_never_exposes_a_partial_file(self, tmp_path: Path) -> None:
        store = _seeded(tmp_path, n=4)
        before = store._warm_sidecar_path().read_bytes()
        with (
            patch("trw_memory.lifecycle.tiers._warm.os.replace", side_effect=OSError("crash before rename")),
            pytest.raises(OSError),
        ):
            store.warm_add_many([("M-0001", _payload(1, content="changed"), None)])
        assert store._warm_sidecar_path().read_bytes() == before


_CHILD = """
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.models.memory import MemoryEntry

base, start, rounds = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
store = WarmTierStore(base)
t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
for step in range(1, rounds + 1):
    items = []
    for i in (start, start + 1):
        data = MemoryEntry(
            id=f"M-{i:04d}", content=f"content {i}", namespace="project:t", tags=["t"],
            created_at=t0, updated_at=t0, valid_from=t0,
        ).model_dump(mode="json")
        data["last_accessed_at"] = (t0 + timedelta(minutes=step)).isoformat()
        data["access_count"] = step
        items.append((f"M-{i:04d}", data, None))
    store.warm_add_many(items)
    store.warm_entries()
"""


class TestCrossProcess:
    def _spawn(self, tmp_path: Path, start: int, rounds: int) -> subprocess.Popen[bytes]:
        env = {**os.environ, "PYTHONPATH": str(Path(trw_memory.__file__).parents[1])}
        return subprocess.Popen([sys.executable, "-c", _CHILD, str(tmp_path), str(start), str(rounds)], env=env)

    def test_concurrent_writers_lose_no_access_update(self, tmp_path: Path) -> None:
        store = _seeded(tmp_path, n=8)
        store.warm_entries()  # warm this process's cache before the others write
        rounds = 30
        children = [self._spawn(tmp_path, start, rounds) for start in (0, 2, 4)]
        for child in children:
            assert child.wait(timeout=120) == 0
        by_id = {e["id"]: e for e in store.warm_entries()}
        assert len(by_id) == 8
        for i in range(6):
            assert by_id[f"M-{i:04d}"]["access_count"] == rounds
        assert by_id["M-0006"]["access_count"] == 0
        # every refresh was an append (below the compaction floor) and none fused or tore
        assert len(_physical(store)) == 8 + 3 * rounds * 2
        parsed = parse_sidecar(store._warm_sidecar_path())
        assert parsed.dead == 3 * rounds * 2
        assert {e["id"]: e for e in store.warm_entries()} == _old_reader_view(store)


class TestRecallPath:
    async def test_repeat_recalls_append_access_stats_without_rewriting(self, memory_client: MemoryClient) -> None:
        with patch.object(memory_client, "_get_embedder", return_value=None):
            for i in range(6):
                await memory_client.store(content=f"sunrise painting lesson number {i}", tags=["art"])
            await memory_client.recall("sunrise painting", limit=5)
            warm = get_tier_manager(memory_client._config, "default")._warm_store
            before = datetime.now(timezone.utc)
            with patch("trw_memory.lifecycle.tiers._warm.os.replace", wraps=os.replace) as replace:
                first = await memory_client.recall("sunrise painting", limit=5)
                await memory_client.recall("painting lesson", limit=5)
        assert first
        assert replace.call_count == 0
        live = {e["id"]: e for e in warm.warm_entries()}
        for result in first:
            assert datetime.fromisoformat(str(live[result["memory_id"]]["last_accessed_at"])) >= before
        assert live == _old_reader_view(warm)
        await memory_client.close()
