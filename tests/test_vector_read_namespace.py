"""Namespace-qualified vector reads must not alias equal IDs across stores."""

from __future__ import annotations

import sqlite3
import struct
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.storage._vector_ops import get_stored_embeddings


@pytest.fixture
def connection() -> Iterator[sqlite3.Connection]:
    # Exercise the actual reader SQL/decoder without requiring sqlite-vec.
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE vec_index(rowid INTEGER PRIMARY KEY, entry_id TEXT, namespace TEXT)")
    conn.execute("CREATE TABLE vec_memories(rowid INTEGER PRIMARY KEY, embedding BLOB)")
    for index, (namespace, vector) in enumerate(
        [("default", [1.0, 0.0]), ("user:other", [0.0, 1.0]), ("", [-1.0, 0.0])], start=1
    ):
        conn.execute("INSERT INTO vec_index VALUES(?, ?, ?)", (index, "same-id", namespace))
        conn.execute("INSERT INTO vec_memories VALUES(?, ?)", (index, struct.pack("2f", *vector)))
    yield conn
    conn.close()


@pytest.mark.parametrize(
    ("namespace", "expected"),
    [("default", [1.0, 0.0]), ("user:other", [0.0, 1.0]), ("", [-1.0, 0.0])],
)
def test_explicit_scope_resolves_collision_before_mapping(connection, namespace, expected) -> None:
    assert get_stored_embeddings(
        connection, threading.Lock(), vec_available=True, entry_ids=["same-id"], namespace=namespace
    ) == {"same-id": expected}


@pytest.mark.parametrize("namespace", ["absent", "' OR 1=1 --"])
def test_missing_or_sql_shaped_namespace_does_not_widen(connection, namespace) -> None:
    assert (
        get_stored_embeddings(
            connection, threading.Lock(), vec_available=True, entry_ids=["same-id"], namespace=namespace
        )
        == {}
    )


def test_omitted_and_none_keep_legacy_unscoped_behavior(connection) -> None:
    kwargs = {"vec_available": True, "entry_ids": ["same-id"]}
    omitted = get_stored_embeddings(connection, threading.Lock(), **kwargs)
    assert omitted == get_stored_embeddings(connection, threading.Lock(), namespace=None, **kwargs)
    assert set(omitted) == {"same-id"}  # Legacy collisions are intentionally not disambiguated.


def test_namespace_binding_fits_sqlite_chunk_limit(connection, monkeypatch) -> None:
    monkeypatch.setattr("trw_memory.storage._sql_utils.SQLITE_SAFE_BIND_LIMIT", 3)

    class LimitedConnection:
        def execute(self, sql, params):
            assert len(params) <= 3
            return connection.execute(sql, params)

    # Four IDs force two statements once the namespace binding is reserved.
    assert get_stored_embeddings(
        LimitedConnection(),
        threading.Lock(),
        vec_available=True,
        entry_ids=["missing-1", "same-id", "missing-2", "missing-3"],
        namespace="default",
    ) == {"same-id": [1.0, 0.0]}


def test_empty_ids_and_unavailable_vectors_do_not_query(connection) -> None:
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    assert (
        get_stored_embeddings(connection, threading.Lock(), vec_available=True, entry_ids=[], namespace="default") == {}
    )
    assert (
        get_stored_embeddings(
            connection, threading.Lock(), vec_available=False, entry_ids=["same-id"], namespace="default"
        )
        == {}
    )
    assert statements == []


def test_real_sqlite_vec_backend_forwards_namespace(tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    backend = SQLiteBackend(tmp_path / "memory.db", dim=2)
    try:
        if not backend.vec_available:
            pytest.skip("sqlite-vec virtual table unavailable")
        backend.upsert_vector("same-id", [1.0, 0.0], namespace="default")
        backend.upsert_vector("same-id", [0.0, 1.0], namespace="user:other")
        assert backend.get_stored_embeddings(["same-id"], namespace="default") == {"same-id": [1.0, 0.0]}
        assert backend.get_stored_embeddings(["same-id"], namespace="user:other") == {"same-id": [0.0, 1.0]}
    finally:
        backend.close()
