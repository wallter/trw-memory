"""``vector_space_census``: a provenance-only count of a namespace's vectors by claimed space.

Consumers (trw-mcp learn-time dedup) trust a dense verdict only when the whole
namespace is provably in the loaded space, so the census must never mistake an
unknown record for a real space, never merge two spaces that share part of
their identity, never span namespaces, and never read the vector blobs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend

pytest.importorskip("sqlite_vec")

SPACE_A = EmbeddingSpace("a" * 64, "test-encoder:a", 3)
SPACE_B = EmbeddingSpace("b" * 64, "test-encoder:b", 3)
#: Same artifact bytes as SPACE_A, different encoding contract: a different space.
SPACE_A_OTHER_ENCODING = EmbeddingSpace("a" * 64, "test-encoder:a-query-prefixed", 3)
VECTOR = [1.0, 0.0, 0.0]


def _put(backend: SQLiteBackend, entry_id: str, space: EmbeddingSpace | None, namespace: str = "default") -> None:
    entry = MemoryEntry(id=entry_id, content=f"content {entry_id}", namespace=namespace)
    backend.store(entry)
    proof = VectorProvenance.for_vector(space, f"{entry.content} {entry.detail}", VECTOR) if space else None
    backend.upsert_vector(entry_id, VECTOR, namespace=namespace, provenance=proof)


@pytest.fixture()
def backend(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = SQLiteBackend(tmp_path / "census.db", dim=3)
    if not store.supports_vectors():
        store.close()
        pytest.skip("sqlite-vec did not load")
    yield store
    store.close()


def test_counts_by_full_space_identity_with_unknown_under_none(backend: SQLiteBackend) -> None:
    _put(backend, "a1", SPACE_A)
    _put(backend, "a2", SPACE_A)
    _put(backend, "b1", SPACE_B)
    _put(backend, "a-enc", SPACE_A_OTHER_ENCODING)
    _put(backend, "legacy", None)
    _put(backend, "malformed", SPACE_A)
    backend._conn.execute("UPDATE vec_index SET provenance_json = '{not json' WHERE entry_id = 'malformed'")
    backend._conn.commit()

    assert backend.vector_space_census(namespace="default") == {
        SPACE_A: 2,
        SPACE_B: 1,
        SPACE_A_OTHER_ENCODING: 1,
        None: 2,
    }


def test_is_scoped_to_one_namespace(backend: SQLiteBackend) -> None:
    _put(backend, "mine", SPACE_A, namespace="default")
    _put(backend, "theirs", SPACE_B, namespace="other")

    assert backend.vector_space_census(namespace="default") == {SPACE_A: 1}
    assert backend.vector_space_census(namespace="nobody") == {}


def test_reads_no_vector_blob(backend: SQLiteBackend) -> None:
    _put(backend, "a1", SPACE_A)
    statements: list[str] = []
    backend._conn.set_trace_callback(statements.append)
    try:
        backend.vector_space_census(namespace="default")
    finally:
        backend._conn.set_trace_callback(None)

    assert statements, "the census must query the store"
    assert not any("vec_memories" in sql for sql in statements)


def test_backends_without_the_capability_report_no_census() -> None:
    """The interface default is None ("unknown"), never an empty census ("all clean")."""
    assert StorageBackend.vector_space_census(object(), namespace="default") is None  # type: ignore[arg-type]


def test_a_sql_error_is_no_census_with_a_warning_never_an_empty_one() -> None:
    import sqlite3
    import threading
    from unittest.mock import MagicMock

    import structlog

    from trw_memory.storage._vector_ops import vector_space_census

    conn = MagicMock()
    conn.execute.side_effect = sqlite3.OperationalError("no such table: vec_index")
    with structlog.testing.capture_logs() as events:
        census = vector_space_census(conn, threading.RLock(), vec_available=True, namespace="default")

    assert census is None
    assert [event["log_level"] for event in events if event["event"] == "vector_space_census_error"] == ["warning"]


def test_without_sqlite_vec_there_is_no_census() -> None:
    import threading
    from unittest.mock import MagicMock

    from trw_memory.storage._vector_ops import vector_space_census

    conn = MagicMock()
    assert vector_space_census(conn, threading.RLock(), vec_available=False, namespace="default") is None
    conn.execute.assert_not_called()


def test_a_non_string_namespace_is_refused(backend: SQLiteBackend) -> None:
    with pytest.raises(TypeError):
        backend.vector_space_census(namespace=None)  # type: ignore[arg-type]


def test_an_orphan_vector_never_stands_in_for_a_row_without_one(backend: SQLiteBackend) -> None:
    """C12 rc4: an orphan (a vector whose row was deleted while sqlite-vec was unavailable) offset a vectorless row,
    so memory_similar trusted a window that never compared that row."""
    _put(backend, "a1", SPACE_A)
    orphan = VectorProvenance.for_vector(SPACE_A, "gone", VECTOR)
    backend.upsert_vector("gone", VECTOR, namespace="default", provenance=orphan)  # no memories row
    backend.store(MemoryEntry(id="bare", content="content bare", namespace="default"))  # a row with no vector

    assert backend.vector_space_census(namespace="default") == {SPACE_A: 1}
    assert backend.count(namespace="default") == 2
