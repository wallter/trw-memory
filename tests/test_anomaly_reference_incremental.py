"""The incrementally maintained anomaly reference matches a full re-read.

``_anomaly_reference.reference_view`` keeps each namespace's reference window
in memory and applies the backend's change feed instead of re-reading 200 rows
per store. These tests pin (a) equivalence with a fresh full read after every
kind of write (insert, upsert, retire, quarantine flag, canary, blank row,
back-dated insert, in-process delete), in a namespace smaller than the fetch
buffer and in one larger than it, 200 random writes each; (b) that another process's writes are picked
up; (c) the one documented lag (another process's non-top delete) is closed by
the age bound; (d) the SQLite change token and feed contract; and (e) that the
throttled ``anomaly_stats.yaml`` is persisted, flushed on close, and recovered.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import trw_memory
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.security import _anomaly_reference, _runtime_anomaly
from trw_memory.security._anomaly_reference import (
    _ranked,
    _view,
    fetch_reference,
    reference_view,
)
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.persistence import read_yaml
from trw_memory.storage.sqlite_backend import SQLiteBackend

NS = "project:ref"


class _ReadCounter(SQLiteBackend):
    full_reads = 0

    def list_entries(self, **kwargs: Any) -> list[MemoryEntry]:  # type: ignore[override]
        type(self).full_reads += 1
        return super().list_entries(**kwargs)


def _fresh(backend: StorageBackend) -> _anomaly_reference.ReferenceView:
    """The view a per-store full re-read (the pre-cache behaviour) would produce."""
    return _view(_ranked(fetch_reference(NS, backend)))


def _entry(entry_id: str, rng: random.Random, **fields: Any) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content="w" * rng.randint(5, 400),
        tags=[f"t{i}" for i in range(rng.randint(0, 4))],
        importance=round(rng.random(), 2),
        namespace=NS,
        **fields,
    )


def _random_op(backend: SQLiteBackend, rng: random.Random, ids: list[str], step: int) -> None:
    op = rng.choice(
        ["insert", "insert", "insert", "upsert", "retire", "quarantine", "canary", "blank", "old", "delete"]
    )
    if op == "insert" or not ids:
        ids.append(f"N-{step}")
        backend.store(_entry(ids[-1], rng))
    elif op == "upsert":
        backend.store(_entry(rng.choice(ids), rng))
    elif op == "retire":
        backend.update(rng.choice(ids), namespace=NS, status=MemoryStatus.OBSOLETE)
    elif op == "quarantine":
        backend.update(rng.choice(ids), namespace=NS, metadata={"quarantined": "true"})
    elif op == "canary":
        ids.append(f"C-{step}")
        backend.store(_entry(ids[-1], rng, metadata={"system_canary": "true"}))
    elif op == "blank":
        ids.append(f"B-{step}")
        backend.store(MemoryEntry(id=ids[-1], content="   ", namespace=NS))
    elif op == "old":
        ids.append(f"O-{step}")
        backend.store(_entry(ids[-1], rng, updated_at=datetime.now(timezone.utc) - timedelta(days=rng.randint(1, 9))))
    else:
        backend.delete(ids.pop(rng.randrange(len(ids))), namespace=NS)


@pytest.mark.parametrize("seed_rows", [30, 260])
def test_incremental_view_matches_a_full_read_after_every_write(tmp_path: Path, seed_rows: int) -> None:
    rng = random.Random(seed_rows)
    backend = SQLiteBackend(tmp_path / "mem.db")
    ids = [f"S-{i}" for i in range(seed_rows)]
    backend.store_many([_entry(entry_id, rng) for entry_id in ids])

    for step in range(200):
        _random_op(backend, rng, ids, step)
        view = reference_view(NS, backend)
        full_read = _ranked(fetch_reference(NS, backend))
        assert view == _view(full_read), f"diverged after step {step}"
        # The invariant behind that equality: the cached rows are exactly the
        # ranked prefix a full read returns, not just a prefix that scores alike.
        window = _anomaly_reference._CACHE[(str(tmp_path / "mem.db"), NS)].window
        assert window is not None
        assert window.rows == full_read, f"buffer diverged after step {step}"
    backend.close()


def test_an_update_stamped_with_the_top_timestamp_is_merged(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "mem.db")
    backend.store_many([MemoryEntry(id=f"A{i}", content="a" * 10, namespace=NS) for i in range(12)])
    reference_view(NS, backend)
    top = backend.list_entries(namespace=NS, limit=1)[0]
    # Same microsecond as the cached top row: only the feed's ">=" arm returns it.
    backend.update("A0", namespace=NS, content="b" * 900, updated_at=top.updated_at)
    backend.store(MemoryEntry(id="new", content="c", namespace=NS))  # moves the token

    assert reference_view(NS, backend) == _fresh(backend)
    backend.close()


def test_single_row_stores_do_not_re_read_the_window(tmp_path: Path) -> None:
    rng = random.Random(7)
    backend = _ReadCounter(tmp_path / "mem.db")
    backend.store_many([_entry(f"S-{i}", rng) for i in range(300)])
    _ReadCounter.full_reads = 0

    for i in range(100):
        reference_view(NS, backend)
        backend.store(_entry(f"N-{i}", rng))

    assert _ReadCounter.full_reads == 1  # the first score seeds; every later store is merged from the feed
    backend.close()


def test_feed_overflow_and_a_departing_ranked_row_fall_back_to_a_full_read(tmp_path: Path) -> None:
    rng = random.Random(3)
    backend = _ReadCounter(tmp_path / "mem.db")
    backend.store_many([_entry(f"S-{i}", rng) for i in range(260)])
    reference_view(NS, backend)
    _ReadCounter.full_reads = 0

    backend.store_many([_entry(f"X-{i}", rng) for i in range(70)])  # more rows than the largest feed
    assert reference_view(NS, backend) == _fresh(backend)
    assert _ReadCounter.full_reads == 2  # reseed + the comparison read

    newest = backend.list_entries(namespace=NS, status=MemoryStatus.ACTIVE, limit=1)[0]
    backend.update(newest.id, namespace=NS, status=MemoryStatus.ARCHIVED)  # row 201 must move up
    _ReadCounter.full_reads = 0
    assert reference_view(NS, backend) == _fresh(backend)
    assert _ReadCounter.full_reads == 2
    backend.close()


def test_a_back_dated_row_below_the_buffer_does_not_fill_a_departed_slot(tmp_path: Path) -> None:
    rng = random.Random(9)
    backend = SQLiteBackend(tmp_path / "mem.db")
    backend.store_many([_entry(f"S-{i}", rng) for i in range(260)])
    reference_view(NS, backend)
    newest = backend.list_entries(namespace=NS, status=MemoryStatus.ACTIVE, limit=1)[0]
    # One interval: a ranked row leaves AND a row ranked below the whole buffer arrives.
    backend.update(newest.id, namespace=NS, status=MemoryStatus.OBSOLETE)
    backend.store(_entry("ancient", rng, updated_at=datetime(2001, 1, 1, tzinfo=timezone.utc)))

    reference_view(NS, backend)
    window = _anomaly_reference._CACHE[(str(tmp_path / "mem.db"), NS)].window
    assert window is not None
    assert window.rows == _ranked(fetch_reference(NS, backend))  # row 201 moved up, not "ancient"
    backend.close()


def _run_other_process(db: Path, script: str) -> None:
    src = str(Path(trw_memory.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    code = (
        textwrap.dedent(
            f"""
        from datetime import datetime, timedelta, timezone
        from pathlib import Path
        from trw_memory.models.memory import MemoryEntry, MemoryStatus
        from trw_memory.storage.sqlite_backend import SQLiteBackend
        backend = SQLiteBackend(Path({str(db)!r}))
        NS = {NS!r}
        """
        )
        + textwrap.dedent(script)
        + "\nbackend.close()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=120)


def test_another_processes_writes_are_picked_up(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    rng = random.Random(11)
    backend = SQLiteBackend(db)
    backend.store_many([_entry(f"S-{i}", rng) for i in range(40)])
    before = reference_view(NS, backend)

    _run_other_process(
        db,
        """
        backend.store(MemoryEntry(id="P-1", content="x" * 5000, tags=["a"] * 9, namespace=NS))
        backend.store(MemoryEntry(id="P-2", content="y", namespace=NS,
                                  updated_at=datetime.now(timezone.utc) - timedelta(days=3)))
        backend.update("S-5", namespace=NS, status=MemoryStatus.OBSOLETE)
        backend.delete("P-1", namespace=NS)  # the newest row: its removal lowers the token's maximum
        backend.store(MemoryEntry(id="P-3", content="z" * 70, namespace=NS))
        """,
    )

    after = reference_view(NS, backend)
    assert after == _fresh(backend)
    assert after != before
    backend.close()


def test_another_processes_deep_delete_lags_at_most_the_reseed_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "mem.db"
    rng = random.Random(5)
    backend = SQLiteBackend(db)
    backend.store_many([_entry(f"S-{i}", rng) for i in range(40)])
    reference_view(NS, backend)

    _run_other_process(db, 'backend.delete("S-20", namespace=NS)')  # neither the newest row nor the top rowid
    assert reference_view(NS, backend) != _fresh(backend)  # the documented lag: invisible to the token

    monkeypatch.setattr(_anomaly_reference, "_RESEED_MAX_AGE_S", 0.0)
    assert reference_view(NS, backend) == _fresh(backend)
    backend.close()


def test_backend_without_a_change_token_re_reads_every_call(tmp_path: Path) -> None:
    from trw_memory.storage.yaml_backend import YAMLBackend

    backend = YAMLBackend(tmp_path / "yaml")
    for i in range(12):
        backend.store(MemoryEntry(id=f"Y-{i}", content="c" * (i + 1), namespace=NS))
    calls: list[int] = []
    real = backend.list_entries

    def counting(**kwargs: Any) -> list[MemoryEntry]:
        calls.append(1)
        return real(**kwargs)

    backend.list_entries = counting  # type: ignore[method-assign]
    first = reference_view(NS, backend)
    backend.store(MemoryEntry(id="Y-new", content="n" * 99, namespace=NS))
    second = reference_view(NS, backend)

    assert len(calls) == 2
    assert first.stats.sample_count == 12
    assert second.stats.sample_count == 13


class TestSQLiteChangeFeed:
    def test_token_is_stable_without_writes_and_ignores_access_bookkeeping(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "mem.db")
        backend.store(MemoryEntry(id="A", content="a", namespace=NS))
        token = backend.namespace_change_token(NS)
        backend.increment_recall_access(["A"], namespace=NS)
        backend.store(MemoryEntry(id="Z", content="other namespace", namespace="project:other"))

        assert backend.namespace_change_token(NS) == token
        backend.close()

    def test_every_write_kind_moves_the_token(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "mem.db")
        backend.store_many([MemoryEntry(id=f"A{i}", content="a", namespace=NS) for i in range(3)])
        seen = {backend.namespace_change_token(NS)}
        writes = [
            lambda: backend.store(MemoryEntry(id="A1", content="upsert", namespace=NS)),
            lambda: backend.store(
                MemoryEntry(id="old", content="b", namespace=NS, updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
            ),
            lambda: backend.update("A0", namespace=NS, status=MemoryStatus.OBSOLETE),
            lambda: backend.delete("A2", namespace=NS),
            lambda: backend.delete_by_namespace(NS),
        ]
        for write in writes:
            write()
            token = backend.namespace_change_token(NS)
            assert token not in seen
            seen.add(token)
        backend.close()

    def test_feed_returns_changed_rows_newest_first_and_none_on_overflow(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(tmp_path / "mem.db")
        backend.store_many([MemoryEntry(id=f"A{i}", content="a", namespace=NS) for i in range(5)])
        token = backend.namespace_change_token(NS)
        assert token is not None
        backend.store(MemoryEntry(id="new", content="n", namespace=NS))
        backend.update("A0", namespace=NS, status=MemoryStatus.OBSOLETE)

        changed = backend.entries_changed_since(NS, token, limit=10)
        assert changed is not None
        assert [entry.id for entry in changed][:2] == ["A0", "new"]
        assert {entry.id for entry in changed} <= {"A0", "new", "A4", "A3", "A2", "A1"}  # plus the old top row(s)
        assert backend.entries_changed_since(NS, token, limit=1) is None
        backend.close()


class TestAnomalyStatsFile:
    def _config(self, tmp_path: Path) -> MemoryConfig:
        return MemoryConfig(storage_path=str(tmp_path / "store"))

    def _stats(self, count: int) -> _anomaly_reference.AnomalyStats:
        return _anomaly_reference.AnomalyStats(sample_count=count, dimensions={})

    def test_writes_are_throttled_and_the_latest_is_flushed(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        path = Path(config.quarantine_path).parent / "anomaly_stats.yaml"
        _runtime_anomaly.write_anomaly_stats(config, self._stats(1))
        assert read_yaml(path)["sample_count"] == 1
        for count in range(2, 6):
            _runtime_anomaly.write_anomaly_stats(config, self._stats(count))
        assert read_yaml(path)["sample_count"] == 1  # deferred

        _runtime_anomaly.flush_anomaly_stats(config)
        assert read_yaml(path)["sample_count"] == 5

    def test_the_deferred_count_bounds_the_lag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_runtime_anomaly, "_STATS_WRITE_MAX_DEFERRED", 3)
        config = self._config(tmp_path)
        path = Path(config.quarantine_path).parent / "anomaly_stats.yaml"
        for count in range(1, 5):
            _runtime_anomaly.write_anomaly_stats(config, self._stats(count))
        assert read_yaml(path)["sample_count"] == 4  # 1 written, 2-3 deferred, the 3rd deferral forces 4

    async def test_client_close_flushes_and_a_cold_process_recovers_the_same_stats(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trw_memory.client import MemoryClient

        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "store"))
        monkeypatch.setattr("trw_memory.client.MemoryClient._get_embedder", lambda self: None)

        async def store_rows(rows: range) -> tuple[MemoryConfig, dict[str, object]]:
            client = MemoryClient(namespace=NS, mode="local")
            for i in rows:
                await client.store(f"row {i} " + "w" * i)
            stats_path = Path(client._config.quarantine_path).parent / "anomaly_stats.yaml"
            await client.close()
            return client._config, read_yaml(stats_path)

        config, persisted = await store_rows(range(15))
        # Row 15 was scored against rows 1-14; that deferred snapshot is written by close().
        assert persisted["sample_count"] == 14

        _anomaly_reference._CACHE.clear()  # the next process starts cold
        with create_backend_from_config(config, NS) as backend:
            expected = _fresh(backend).stats  # what the 16th row is scored against: a full read of rows 1-15
        _config, recovered = await store_rows(range(15, 16))
        assert recovered["sample_count"] == expected.sample_count == 15
        assert recovered["dimensions"] == expected.dimensions
