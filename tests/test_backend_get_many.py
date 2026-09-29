"""PRD-CORE-318 FR02: ``StorageBackend.get_many`` returns what ``get`` would, and every backend has it.

``get_many`` is an abstract read method of the ABC, so the PRD-CORE-333 census, which
enumerates the ABC's read methods for its quarantine filter, covers it by construction.
"""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend


def _entry(entry_id: str, namespace: str = "default") -> MemoryEntry:
    return MemoryEntry(id=entry_id, content=f"content {entry_id}", namespace=namespace)


def _backend(kind: str, tmp_path: Path) -> StorageBackend:
    return SQLiteBackend(tmp_path / "memory.db") if kind == "sqlite" else YAMLBackend(tmp_path / "entries")


@pytest.mark.parametrize("kind", ["sqlite", "yaml"])
def test_get_many_matches_get(kind: str, tmp_path: Path) -> None:
    backend = _backend(kind, tmp_path)
    try:
        for entry_id in ("a", "b", "c"):
            backend.store(_entry(entry_id))
        backend.store(_entry("foreign", namespace="other"))
        asked = ["c", "a", "missing", "foreign", "a"]

        found = backend.get_many(asked, namespace="default")

        expected = {i: e for i in asked if (e := backend.get(i, namespace="default")) is not None}
        assert found == expected
        assert set(found) == {"a", "c"}
        assert backend.get_many([], namespace="default") == {}
    finally:
        backend.close()


def test_get_many_spans_bind_chunks(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "memory.db")
    try:
        ids = [f"e{i:05}" for i in range(2500)]
        backend.store_many([_entry(i) for i in ids])
        assert set(backend.get_many(ids, namespace="default")) == set(ids)
    finally:
        backend.close()


def test_quarantined_id_passed_to_get_many_is_not_returned(tmp_path: Path) -> None:
    """A row the backend's read path quarantines (undecodable UTF-8) never comes back."""
    backend = SQLiteBackend(tmp_path / "memory.db")
    try:
        backend.store(_entry("bad"))
        backend.store(_entry("keep"))
        with sqlite3.connect(backend._db_path) as connection:
            connection.execute("UPDATE memories SET detail=CAST(X'FF' AS TEXT) WHERE id='bad'")

        found = backend.get_many(["bad", "keep"], namespace="default")

        assert set(found) == {"keep"}
        assert backend.quarantine_count_utf8 == 1
    finally:
        backend.close()


def _concrete_backends() -> list[type[StorageBackend]]:
    seen: list[type[StorageBackend]] = []
    pending = list(StorageBackend.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if not inspect.isabstract(cls) and cls.__module__.startswith("trw_memory."):
            seen.append(cls)
    return seen


def test_census_every_backend_implements_get_many() -> None:
    """Q2 census: get_many is abstract on the ABC and each shipped backend defines its own."""
    assert "get_many" in StorageBackend.__abstractmethods__
    backends = _concrete_backends()
    assert {cls.__name__ for cls in backends} >= {"SQLiteBackend", "YAMLBackend"}
    assert [cls.__name__ for cls in backends if cls.get_many is StorageBackend.get_many] == []
