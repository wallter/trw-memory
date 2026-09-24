"""The anomaly reference is not re-read per scored row, and the stats file is not rewritten per row.

Every row of a ``bulk_store`` is scored before any row is persisted, so each
row already saw the same reference window. Outside a batch, a single-row writer
brings a cached window up to date from the backend's change feed instead of
re-reading it (``_anomaly_reference``), and ``anomaly_stats.yaml`` is written
at most once per interval (``write_anomaly_stats``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security import _anomaly_reference, _runtime_anomaly
from trw_memory.security._runtime_anomaly import score_anomaly, shared_anomaly_reference, write_anomaly_stats
from trw_memory.storage.sqlite_backend import SQLiteBackend


class _CountingBackend(SQLiteBackend):
    list_calls = 0

    def list_entries(self, **kwargs: Any) -> list[MemoryEntry]:  # type: ignore[override]
        type(self).list_calls += 1
        return super().list_entries(**kwargs)


@pytest.fixture
def backend(tmp_path: Path) -> SQLiteBackend:
    _CountingBackend.list_calls = 0
    store = _CountingBackend(tmp_path / "mem.db")
    for i in range(5):
        store.store(MemoryEntry(id=f"M-{i}", content=f"reference row {i} " * (i + 1), namespace="project:a"))
    return store


def _writes(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    calls: list[object] = []
    monkeypatch.setattr(_runtime_anomaly, "write_yaml", lambda path, payload: calls.append(payload))
    return calls


def test_outside_the_scope_an_unchanged_namespace_is_read_once(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writes = _writes(monkeypatch)
    config = MemoryConfig(storage_path=str(tmp_path / "store"))
    for i in range(3):
        _anomaly, stats = score_anomaly(
            MemoryEntry(id=f"N-{i}", content="new", namespace="project:a"), backend, config=config
        )
        write_anomaly_stats(config, stats)

    assert _CountingBackend.list_calls == 1  # later scores reuse the cached window: the token did not move
    assert len(writes) == 1  # the first write is immediate, the rest are deferred
    _runtime_anomaly.flush_anomaly_stats(config)
    assert len(writes) == 2  # the pending snapshot is written on flush
    _runtime_anomaly.flush_anomaly_stats(config)
    assert len(writes) == 2  # nothing pending


def test_a_write_to_the_namespace_is_merged_without_a_full_read(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _writes(monkeypatch)
    config = MemoryConfig(storage_path=str(tmp_path / "store"))
    score_anomaly(MemoryEntry(id="N-0", content="new", namespace="project:a"), backend, config=config)
    backend.store(MemoryEntry(id="N-1", content="persisted row", namespace="project:a"))
    _anomaly, stats = score_anomaly(MemoryEntry(id="N-2", content="new", namespace="project:a"), backend, config=config)

    assert _CountingBackend.list_calls == 1  # the feed, not list_entries, delivered N-1
    assert stats.sample_count == 6


def test_inside_the_scope_one_read_and_one_write_per_namespace(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writes = _writes(monkeypatch)
    config = MemoryConfig(storage_path=str(tmp_path / "store"))
    outside = [
        score_anomaly(
            MemoryEntry(id=f"O-{i}", content="x" * (i * 400 + 1), namespace="project:a"), backend, config=config
        )[0]
        for i in range(3)
    ]
    _anomaly_reference._CACHE.clear()
    _CountingBackend.list_calls = 0

    with shared_anomaly_reference():
        results = []
        for i in range(3):
            anomaly, stats = score_anomaly(
                MemoryEntry(id=f"O-{i}", content="x" * (i * 400 + 1), namespace="project:a"), backend, config=config
            )
            write_anomaly_stats(config, stats)
            results.append(anomaly)
        score_anomaly(MemoryEntry(id="B-0", content="other", namespace="project:b"), backend, config=config)

    assert results == outside  # identical verdicts to per-row scoring
    assert _CountingBackend.list_calls == 2  # one per namespace
    assert len(writes) == 1


async def test_bulk_store_reads_the_reference_once_per_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory._client_bulk_store import BulkStoreRequest
    from trw_memory.client import MemoryClient

    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "store"))
    monkeypatch.setattr("trw_memory.client.MemoryClient._get_embedder", lambda self: None)
    reads: list[str] = []
    real = _anomaly_reference.fetch_reference

    def spy(namespace: str, store: Any) -> Any:
        reads.append(namespace)
        return real(namespace, store)

    monkeypatch.setattr(_anomaly_reference, "fetch_reference", spy)
    client = MemoryClient(namespace="project:bulk", mode="local")
    summary = await client.bulk_store([BulkStoreRequest(content=f"row {i}") for i in range(6)])
    second = await client.bulk_store([BulkStoreRequest(content=f"more {i}") for i in range(6)])
    await client.close()

    assert summary.stored == 6
    assert second.stored == 6
    assert reads == ["project:bulk"]  # the second batch merged the first one's rows from the change feed
