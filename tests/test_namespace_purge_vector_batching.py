"""Whole-namespace purge removes vectors in bind chunks, not one entry at a time.

``delete_namespace`` used to loop ``delete_vector_internal`` per entry — one
SELECT plus two DELETEs each — while every other sidecar cleanup on the same
code path (``purge_edges_for``, ``purge_tag_postings_for``) issued a single
chunked ``IN``-clause DELETE. These tests pin both halves: the rows really go,
other namespaces are untouched, and the statement count is a function of chunk
count rather than entry count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

pytest.importorskip("sqlite_vec")

_DIM = 8
_ENTRIES_PER_NAMESPACE = 50


def _embedding(seed: int) -> list[float]:
    return [float((seed + i) % 7) / 7.0 for i in range(_DIM)]


def _seed(backend: SQLiteBackend, namespace: str, count: int) -> list[str]:
    ids: list[str] = []
    for index in range(count):
        entry_id = f"{namespace}-{index:03d}"
        backend.store(
            MemoryEntry(
                id=entry_id,
                content=f"entry {entry_id}",
                namespace=namespace,
                tags=["purge"],
            )
        )
        backend.upsert_vector(entry_id, _embedding(index), namespace=namespace)
        ids.append(entry_id)
    return ids


def _vec_rows(backend: SQLiteBackend, namespace: str) -> int:
    return int(backend._conn.execute("SELECT COUNT(*) FROM vec_index WHERE namespace = ?", (namespace,)).fetchone()[0])


def _vec_memory_rows(backend: SQLiteBackend) -> int:
    return int(backend._conn.execute("SELECT COUNT(*) FROM vec_memories").fetchone()[0])


@pytest.fixture
def seeded_backend(tmp_path: Path) -> SQLiteBackend:
    backend = SQLiteBackend(tmp_path / "purge.db", dim=_DIM)
    if not backend.vec_available:
        pytest.skip("sqlite-vec virtual table unavailable in this build")
    _seed(backend, "project:doomed", _ENTRIES_PER_NAMESPACE)
    _seed(backend, "project:keeper", 5)
    return backend


def test_delete_namespace_removes_every_vector_row(seeded_backend: SQLiteBackend) -> None:
    assert _vec_rows(seeded_backend, "project:doomed") == _ENTRIES_PER_NAMESPACE

    deleted = seeded_backend.delete_by_namespace("project:doomed")

    assert deleted == _ENTRIES_PER_NAMESPACE
    assert _vec_rows(seeded_backend, "project:doomed") == 0
    # vec_memories is keyed by rowid only, so a stale row there is invisible to a
    # namespace count — assert the global total dropped to the survivors alone.
    assert _vec_memory_rows(seeded_backend) == 5


def test_delete_namespace_leaves_other_namespaces_intact(seeded_backend: SQLiteBackend) -> None:
    seeded_backend.delete_by_namespace("project:doomed")

    assert _vec_rows(seeded_backend, "project:keeper") == 5
    survivors = seeded_backend._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE namespace = ?", ("project:keeper",)
    ).fetchone()[0]
    assert survivors == 5
    hits = seeded_backend.search_vectors(_embedding(0), top_k=10, namespace="project:keeper")
    assert hits and all(entry_id.startswith("project:keeper-") for entry_id, _ in hits)


def test_delete_namespace_vector_purge_is_chunked_not_per_entry(
    seeded_backend: SQLiteBackend,
) -> None:
    """Statement count must not scale with the number of embedded entries."""
    statements: list[str] = []
    seeded_backend._conn.set_trace_callback(statements.append)
    try:
        seeded_backend.delete_by_namespace("project:doomed")
    finally:
        seeded_backend._conn.set_trace_callback(None)

    # vec0 rewrites each virtual-table DELETE into per-row work on its shadow
    # tables and traces those with a leading ``--``; they are engine internals,
    # not statements this code issues, so count only the top-level ones.
    vec_statements = [
        sql for sql in statements if not sql.lstrip().startswith("--") and ("vec_index" in sql or "vec_memories" in sql)
    ]
    # Two DELETEs per bind chunk; 50 entries is one chunk. The per-entry loop
    # issued 3 statements per entry (SELECT + 2 DELETEs) = 150.
    assert len(vec_statements) == 2, vec_statements
