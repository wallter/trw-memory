"""PRD-CORE-309 B71-13 -- a namespace move (merge, rename, checkout import) writes a page at a time.

``_move_rows`` used to do a ``get``, ``store``, ``upsert_vector`` and ``delete`` per row, and both
the store and the delete ran ``DELETE FROM memories_fts WHERE id = ?`` -- a full scan of the FTS
index, since ``id`` is not an FTS5 key. A 20k-row import was quadratic. These tests pin the batched
path's statement shape (deterministic, gating), its effects (every row, vector, tag and edge
arrives; conflicts stay behind per id; the source drains) and its wall clock (local budget).
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests._optional_extras import vec_unavailable
from tests._timing import assert_budget
from trw_memory._graph_primitives import _upsert_edge
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces import curate
from trw_memory.namespaces.curate import NamespaceStores, merge_namespace, rename_namespace
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend
from trw_memory.tools.checkout_import import memory_import_checkout_impl

_NS = "project:acme-1a2b3c4d"
_DIM = 4
_AT = datetime(2026, 9, 23, tzinfo=timezone.utc)
_PER_ROW_FTS_DELETE = re.compile(r"DELETE FROM memories_fts WHERE id = ")
_PER_ROW_GET = re.compile(r"FROM memories WHERE namespace = \S+ AND id = ")


def _row(index: int, namespace: str) -> MemoryEntry:
    return MemoryEntry(
        id=f"L-{index:06d}",
        namespace=namespace,
        content=f"row {index} quokka",
        detail=f"detail {index}",
        tags=["moved", f"t{index % 3}"],
        created_at=_AT,
        updated_at=_AT,
        valid_from=_AT,
        last_accessed_at=_AT,
    )


def _vector(index: int) -> list[float]:
    vector = [0.0] * _DIM
    vector[index % _DIM] = 1.0
    return vector


def _fill(store: SQLiteBackend, rows: int, namespace: str = "default", *, edges: int = 0) -> None:
    with store.transaction():
        store.store_many([_row(i, namespace) for i in range(rows)])
        for i in range(rows):
            store.upsert_vector(f"L-{i:06d}", _vector(i), namespace=namespace)
        with store._lock:
            for i in range(0, 2 * edges, 2):
                _upsert_edge(
                    store._conn, f"L-{i:06d}", f"L-{i + 1:06d}", "related_to", 0.5, _AT.isoformat(), namespace=namespace
                )


@pytest.fixture
def pair(tmp_path: Path) -> Iterator[tuple[SQLiteBackend, SQLiteBackend]]:
    source = SQLiteBackend(tmp_path / "work.db", dim=_DIM)
    destination = SQLiteBackend(tmp_path / "user.db", dim=_DIM)
    if not destination.vec_available:
        vec_unavailable("sqlite-vec unavailable")
    yield source, destination
    source.close()
    destination.close()


def _traced(*stores: SQLiteBackend) -> list[str]:
    statements: list[str] = []
    for store in stores:
        store._conn.set_trace_callback(statements.append)
    return statements


def test_a_merge_issues_no_per_row_fts_delete_or_get(
    pair: tuple[SQLiteBackend, SQLiteBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fails on the per-row loop: it ran two FTS full scans and one ``get`` per moved row."""
    source, destination = pair
    monkeypatch.setattr(curate, "_BATCH_LIMIT", 10)
    _fill(source, 40)
    destination.store(_row(3, _NS))
    statements = _traced(source, destination)

    result = merge_namespace(NamespaceStores(source=source, destination=destination), "default", _NS)

    assert (result.moved, result.skipped) == (39, 1)
    pages = 4
    assert not [s for s in statements if _PER_ROW_FTS_DELETE.search(s)]
    assert not [s for s in statements if _PER_ROW_GET.search(s)]
    # PRD-CORE-330: purge_fts_rows_for now also purges the memories_fts_rowid
    # map alongside memories_fts itself (2 IN-list DELETEs per page per side,
    # not 1), so the bound doubles -- still one statement per table per page,
    # never one per row.
    fts_deletes = [s for s in statements if s.startswith("DELETE FROM memories_fts")]
    assert len(fts_deletes) <= 4 * pages  # up to 2 IN-list purges (memories_fts + its rowid map) per page per side


def test_a_merge_moves_every_row_vector_tag_and_edge_and_keeps_conflicts(
    pair: tuple[SQLiteBackend, SQLiteBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = pair
    monkeypatch.setattr(curate, "_BATCH_LIMIT", 7)  # conflicts straddle page boundaries
    _fill(source, 30, edges=5)
    conflicts = {"L-000002", "L-000007", "L-000019"}
    for entry_id in conflicts:
        destination.store(_row(int(entry_id[2:]), _NS).model_copy(update={"content": "the destination's own"}))

    result = merge_namespace(NamespaceStores(source=source, destination=destination), "default", _NS)

    moved = {f"L-{i:06d}" for i in range(30)} - conflicts
    assert (result.moved, result.skipped, result.status) == (27, 3, "merged")
    assert {e.id for e in destination.list_entries(namespace=_NS, limit=100)} == moved | conflicts
    assert {e.id for e in source.list_entries(namespace="default", limit=100)} == conflicts
    assert all(destination.get(i, namespace=_NS).content == "the destination's own" for i in conflicts)  # type: ignore[union-attr]
    records = destination.get_vector_records(sorted(moved), namespace=_NS)
    assert set(records) == moved and records["L-000005"].embedding == pytest.approx(_vector(5))
    assert source.existing_vector_ids(namespace="default") == conflicts
    assert {(e.source_id, e.target_id) for e in destination.graph_edges(_NS)} == {
        (f"L-{i:06d}", f"L-{i + 1:06d}") for i in range(0, 10, 2)
    }
    assert source.graph_edges("default") == []  # every edge touches a moved row
    # FTS and tag postings follow each row: searchable in the destination, gone from the source.
    assert {e.id for e in destination.search_fts("quokka", namespace=_NS, top_k=100)} == moved
    assert {e.id for e in source.search_fts("quokka", namespace="default", top_k=100)} == conflicts
    assert _postings(destination, _NS) == {(i, t) for i in moved for t in ("moved", f"t{int(i[2:]) % 3}")} | {
        (i, t) for i in conflicts for t in ("moved", f"t{int(i[2:]) % 3}")
    }
    assert {entry_id for entry_id, _tag in _postings(source, "default")} == conflicts


def _postings(store: SQLiteBackend, namespace: str) -> set[tuple[str, str]]:
    with store._lock:
        rows = store._conn.execute("SELECT entry_id, tag FROM memory_tags WHERE namespace = ?", (namespace,))
        return {(str(entry_id), str(tag)) for entry_id, tag in rows}


def test_a_rename_on_one_store_moves_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(curate, "_BATCH_LIMIT", 4)
    with SQLiteBackend(tmp_path / "one.db", dim=_DIM) as store:
        _fill(store, 10, "project:old-00000000", edges=2)
        result = rename_namespace(NamespaceStores.shared(store), "project:old-00000000", _NS)
        assert (result.moved, result.status) == (10, "renamed")
        assert store.count(namespace="project:old-00000000") == 0
        assert len(store.existing_vector_ids(namespace=_NS)) == 10
        assert len(store.graph_edges(_NS)) == 2


def test_the_yaml_fallbacks_loop_the_per_row_methods(tmp_path: Path) -> None:
    store = YAMLBackend(tmp_path / "yaml")
    for i in range(3):
        store.store(_row(i, "default"))
    assert store.existing_ids(["L-000000", "L-000002", "L-nope"], namespace="default") == {"L-000000", "L-000002"}
    assert store.existing_ids(["L-000000"], namespace="elsewhere") == set()
    assert store.delete_many(["L-000000", "L-nope"], namespace="default") == 1
    assert store.count(namespace="default") == 2


_CHECKOUT_SKIP = pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "setlimit"), reason="checkout import needs Python 3.11+"
)


def _checkout_with_identical_head(source: SQLiteBackend, destination: SQLiteBackend, rows: int, identical: int) -> None:
    _fill(source, rows, edges=rows // 200 or 2)
    source.close()
    destination.store_many([_row(i, _NS) for i in range(identical)])  # identical rows: compared, then skipped
    for i in range(identical):
        destination.upsert_vector(f"L-{i:06d}", _vector(i), namespace=_NS)


@_CHECKOUT_SKIP
def test_a_paged_checkout_import_moves_new_rows_and_skips_identical_ones(
    tmp_path: Path, pair: tuple[SQLiteBackend, SQLiteBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gating half of the B71-13 budget below: what a paged import writes, deterministically."""
    source, destination = pair
    monkeypatch.setattr(curate, "_BATCH_LIMIT", 7)  # the identical rows straddle page boundaries
    _checkout_with_identical_head(source, destination, 60, 10)

    answer = memory_import_checkout_impl(_NS, str(tmp_path / "work.db"), [], backend=destination, deadline_seconds=120)

    assert (answer["status"], answer["moved"], answer["skipped"]) == ("ok", 50, 10)
    assert destination.count(namespace=_NS) == 60


@pytest.mark.requires_local_timing
@_CHECKOUT_SKIP
def test_a_20k_row_checkout_import_finishes_within_ten_seconds(
    tmp_path: Path, pair: tuple[SQLiteBackend, SQLiteBackend]
) -> None:
    """The B71-13 budget: ~30 s served (minutes in-process) before batching; <=10 s after.

    Wall clock only (PRD-QUAL-141): the moved/skipped counts are pinned by the gating test above;
    here a run that did not do the full import is charged an unbounded time instead.
    """
    source, destination = pair
    _checkout_with_identical_head(source, destination, 20_000, 50)

    started = time.perf_counter()
    answer = memory_import_checkout_impl(_NS, str(tmp_path / "work.db"), [], backend=destination, deadline_seconds=120)
    elapsed = time.perf_counter() - started

    # An import that refused or stopped early did not do the measured work: its time counts as unbounded.
    did_the_work = (answer["status"], answer["moved"], answer["skipped"]) == ("ok", 19_950, 50)
    assert_budget("checkout_import_20k_rows", elapsed if did_the_work else float("inf"), 10.0, "s")
