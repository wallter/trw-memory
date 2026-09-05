"""Tag-neighbour derivation issues a fixed number of statements.

``derive_tag_neighbours`` used to call a ``COUNT(*)`` per neighbour to get the
denominator of the Jaccard weight — N+1 round trips bounded only by
``graph_tag_derive_top_k`` (up to 200), against a table the same call had
already scanned. The totals are now one ``GROUP BY``. These tests pin the
statement count AND the weights, so the batching cannot quietly change ranking.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.tag_derivation import DerivedTagNeighbour, derive_tag_neighbours
from trw_memory.storage.sqlite_backend import SQLiteBackend

_NAMESPACE = "project:tagderive"


def _config(**overrides: int) -> MemoryConfig:
    base: dict[str, int] = {
        "graph_tag_min_shared_tags": 2,
        "graph_tag_max_tag_postings": 500,
        "graph_tag_derive_top_k": 200,
    }
    base.update(overrides)
    return MemoryConfig(**base)  # type: ignore[arg-type]


def _store(backend: SQLiteBackend, entry_id: str, tags: list[str]) -> None:
    backend.store(MemoryEntry(id=entry_id, content=f"content {entry_id}", namespace=_NAMESPACE, tags=tags))


def _reference_neighbours(conn: sqlite3.Connection, root_id: str, *, config: MemoryConfig) -> list[DerivedTagNeighbour]:
    """Recompute the same relation with an independent per-neighbour COUNT(*).

    This is the pre-batching arithmetic, written out here so the batched query
    is checked against something other than itself.
    """
    min_shared = config.graph_tag_min_shared_tags

    def count(entry_id: str) -> int:
        row = conn.execute(
            "SELECT COUNT(*) FROM memory_tags WHERE namespace = ? AND entry_id = ?",
            (_NAMESPACE, entry_id),
        ).fetchone()
        return int(row[0]) if row else 0

    root_tags = [
        str(row[0])
        for row in conn.execute(
            "SELECT t.tag FROM memory_tags t WHERE t.namespace = ? AND t.entry_id = ? "
            "AND (SELECT COUNT(*) FROM memory_tags p WHERE p.namespace = t.namespace AND p.tag = t.tag) <= ?",
            (_NAMESPACE, root_id, config.graph_tag_max_tag_postings),
        ).fetchall()
    ]
    if len(root_tags) < min_shared:
        return []
    placeholders = ", ".join("?" for _ in root_tags)
    rows = conn.execute(
        f"SELECT entry_id, COUNT(*) AS shared FROM memory_tags "
        f"WHERE namespace = ? AND tag IN ({placeholders}) AND entry_id != ? "
        "GROUP BY entry_id HAVING shared >= ? ORDER BY shared DESC, entry_id ASC LIMIT ?",
        (_NAMESPACE, *root_tags, root_id, min_shared, config.graph_tag_derive_top_k),
    ).fetchall()

    root_total = count(root_id)
    out: list[DerivedTagNeighbour] = []
    for entry_id, shared in rows:
        union = root_total + count(str(entry_id)) - int(shared)
        weight = round(int(shared) / union, 4) if union > 0 else 0.0
        out.append(DerivedTagNeighbour(str(entry_id), int(shared), min(weight, 1.0)))
    return out


@pytest.fixture
def backend(tmp_path: Path) -> SQLiteBackend:
    store = SQLiteBackend(tmp_path / "tags.db")
    _store(store, "ROOT", ["alpha", "beta", "gamma"])
    # Neighbours with deliberately different total tag counts, so a wrong
    # denominator changes the weight rather than cancelling out.
    _store(store, "N-broad", ["alpha", "beta", "d1", "d2", "d3", "d4"])
    _store(store, "N-narrow", ["alpha", "beta"])
    _store(store, "N-three", ["alpha", "beta", "gamma"])
    _store(store, "N-one", ["alpha", "zzz"])  # below min_shared, must not appear
    return store


def test_batched_totals_match_per_neighbour_computation(backend: SQLiteBackend) -> None:
    config = _config()
    result = derive_tag_neighbours(backend._conn, "ROOT", namespace=_NAMESPACE, config=config)

    assert result == _reference_neighbours(backend._conn, "ROOT", config=config)
    assert [n.entry_id for n in result] == ["N-three", "N-broad", "N-narrow"]
    # Jaccard denominators differ per neighbour: 3/3, 2/7, 2/3.
    assert {n.entry_id: n.weight for n in result} == {
        "N-three": 1.0,
        "N-broad": round(2 / 7, 4),
        "N-narrow": round(2 / 3, 4),
    }


def test_statement_count_is_constant_in_neighbour_count(tmp_path: Path) -> None:
    store = SQLiteBackend(tmp_path / "many.db")
    _store(store, "ROOT", ["alpha", "beta"])
    for index in range(60):
        _store(store, f"N-{index:03d}", ["alpha", "beta", f"own-{index}"])

    statements: list[str] = []
    store._conn.set_trace_callback(statements.append)
    try:
        result = derive_tag_neighbours(store._conn, "ROOT", namespace=_NAMESPACE, config=_config())
    finally:
        store._conn.set_trace_callback(None)

    assert len(result) == 60
    # root tags + neighbour GROUP BY + totals GROUP BY. The per-neighbour form
    # issued 2 + 60 + 1.
    assert len(statements) <= 3, statements


def test_root_without_enough_tags_returns_empty(backend: SQLiteBackend) -> None:
    _store(backend, "LONE", ["solo"])
    assert derive_tag_neighbours(backend._conn, "LONE", namespace=_NAMESPACE, config=_config()) == []


def test_missing_memory_tags_table_degrades_to_empty(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "pre-schema-5.db")
    try:
        assert derive_tag_neighbours(conn, "ROOT", namespace=_NAMESPACE, config=_config()) == []
    finally:
        conn.close()
