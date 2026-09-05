"""Regression tests for bounded SQLite bind lists."""

from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import make_entry
from trw_memory.exceptions import StorageError
from trw_memory.storage._sql_utils import iter_bind_chunks
from trw_memory.storage._vector_ops import get_stored_embeddings
from trw_memory.storage.sqlite_backend import SQLiteBackend


class _Cursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._rows


def test_bind_chunk_boundaries() -> None:
    assert [len(chunk) for chunk in iter_bind_chunks(list(range(901)))] == [900, 1]
    assert [len(chunk) for chunk in iter_bind_chunks(list(range(899)), reserved_bindings=2)] == [898, 1]
    assert [len(chunk) for chunk in iter_bind_chunks(list(range(451)), bindings_per_item=2)] == [450, 1]


def test_get_stored_embeddings_chunks_and_merges_results() -> None:
    blob = struct.pack("f", 0.5)

    class _Connection:
        def __init__(self) -> None:
            self.param_counts: list[int] = []

        def execute(self, _sql: str, params: list[str]) -> _Cursor:
            self.param_counts.append(len(params))
            return _Cursor([(entry_id, blob) for entry_id in params])

    connection = _Connection()
    ids = [f"M-{index}" for index in range(901)]

    result = get_stored_embeddings(connection, threading.Lock(), vec_available=True, entry_ids=ids)

    assert connection.param_counts == [900, 1]
    assert set(result) == set(ids)


class _RecordingConnection:
    def __init__(self, real: Any, *, fail_second_write: bool = False) -> None:
        self._real = real
        self._fail_second_write = fail_second_write
        self.bound_writes: list[int] = []
        self.bound_reads: list[int] = []

    def execute(self, sql: str, params: Any = ()) -> Any:
        if "recall_count" in sql and sql.lstrip().startswith("UPDATE memories"):
            self.bound_writes.append(len(params))
            if self._fail_second_write and len(self.bound_writes) == 2:
                raise sqlite3.OperationalError("too many SQL variables")
        if "SELECT DISTINCT namespace" in sql and " IN (" in sql:
            self.bound_reads.append(len(params))
        return self._real.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def test_recall_access_and_namespace_filters_stay_below_bind_ceiling() -> None:
    backend = SQLiteBackend(Path(":memory:"))
    ids = [f"M-{index}" for index in range(899)]
    backend.store_many([make_entry(entry_id=entry_id) for entry_id in ids])
    real = backend._conn
    recording = _RecordingConnection(real)
    backend._conn = recording  # type: ignore[assignment]
    try:
        assert backend.increment_recall_access(ids) == 899
        assert recording.bound_writes == [900, 3]
        assert backend.list_namespaces(required_namespaces=["default", *[f"missing-{i}" for i in range(900)]]) == [
            "default"
        ]
        assert recording.bound_reads == [900, 1]
    finally:
        backend._conn = real  # type: ignore[assignment]
        backend.close()


def test_recall_access_rolls_back_when_later_chunk_fails() -> None:
    backend = SQLiteBackend(Path(":memory:"))
    ids = [f"M-{index}" for index in range(899)]
    backend.store_many([make_entry(entry_id=entry_id) for entry_id in ids])
    real = backend._conn
    backend._conn = _RecordingConnection(real, fail_second_write=True)  # type: ignore[assignment]
    try:
        with pytest.raises(StorageError, match="too many SQL variables"):
            backend.increment_recall_access(ids)
    finally:
        backend._conn = real  # type: ignore[assignment]

    try:
        first = backend.get(ids[0], namespace="default")
        last = backend.get(ids[-1], namespace="default")
        assert first is not None and first.recall_count == 0
        assert last is not None and last.recall_count == 0
    finally:
        backend.close()
