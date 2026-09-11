"""Partial SQLite updates refresh only indexes whose inputs were supplied."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.sync.delta import DeltaTracker


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    db = SQLiteBackend(tmp_path / "indices.db")
    for namespace in ("alpha", "beta"):
        db.store(
            MemoryEntry(
                id="shared",
                namespace=namespace,
                content="oldcontent",
                detail="olddetail",
                tags=["oldtag"],
            )
        )
    yield db
    db.close()


def _postings(backend: SQLiteBackend, namespace: str) -> list[str]:
    return [
        row[0]
        for row in backend._conn.execute(
            "SELECT tag FROM memory_tags WHERE namespace = ? AND entry_id = ? ORDER BY tag",
            (namespace, "shared"),
        ).fetchall()
    ]


@pytest.mark.parametrize(
    "fields",
    [
        {"q_value": 0.8, "q_observations": 2, "outcome_history": ["success"]},
        {"metadata": {"source": "test"}},
        {"importance": 0.9},
    ],
)
def test_nonindexed_update_preserves_search_without_index_writes(
    backend: SQLiteBackend,
    fields: dict[str, object],
) -> None:
    before = backend.get("shared", namespace="alpha")
    assert before is not None
    statements: list[str] = []
    backend._conn.set_trace_callback(statements.append)
    updated = backend.update("shared", namespace="alpha", **fields)
    backend._conn.set_trace_callback(None)
    assert updated is not None
    assert updated.sync_seq == before.sync_seq + 1
    assert updated.sync_hash == DeltaTracker.compute_sync_hash(updated)
    assert updated.last_synced_at is None
    assert not any(
        sql.lstrip().upper().startswith(("INSERT", "DELETE", "UPDATE"))
        and ("memories_fts" in sql or "memory_tags" in sql)
        for sql in statements
    )
    for namespace in ("alpha", "beta"):
        for query in ("oldcontent", "olddetail", "oldtag"):
            assert [e.id for e in backend.search_fts(query, namespace=namespace)] == ["shared"]
        assert [e.id for e in backend.search("oldcontent", tags=["oldtag"], namespace=namespace)] == ["shared"]
        assert _postings(backend, namespace) == ["oldtag"]
    assert backend.get("shared", namespace="beta").sync_seq == before.sync_seq


@pytest.mark.parametrize(
    "field,old,new",
    [
        ("content", "oldcontent", "newcontent"),
        ("detail", "olddetail", "newdetail"),
        ("tags", "oldtag", "newtag"),
    ],
)
def test_indexed_update_refreshes_only_its_namespace(
    backend: SQLiteBackend,
    field: str,
    old: str,
    new: str,
) -> None:
    statements: list[str] = []
    backend._conn.set_trace_callback(statements.append)
    backend.update("shared", namespace="alpha", **{field: [new] if field == "tags" else new})
    backend._conn.set_trace_callback(None)
    assert backend.search_fts(old, namespace="alpha") == []
    assert [e.id for e in backend.search_fts(new, namespace="alpha")] == ["shared"]
    assert [e.id for e in backend.search_fts(old, namespace="beta")] == ["shared"]
    assert backend.search_fts(new, namespace="beta") == []
    assert _postings(backend, "alpha") == ([new] if field == "tags" else ["oldtag"])
    assert _postings(backend, "beta") == ["oldtag"]
    if field != "tags":
        assert not any("memory_tags" in sql for sql in statements)


def test_empty_tags_clear_both_indexes(backend: SQLiteBackend) -> None:
    backend.update("shared", namespace="alpha", tags=[])
    assert backend.search_fts("oldtag", namespace="alpha") == []
    assert _postings(backend, "alpha") == []
    assert [e.id for e in backend.search_fts("oldtag", namespace="beta")] == ["shared"]
    assert _postings(backend, "beta") == ["oldtag"]
