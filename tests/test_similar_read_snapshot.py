"""B71-60: ``memory_similar`` reads its window, records, census, count and rows in one snapshot.

A write from another connection that lands between those reads must not leave the
answer mixing two states of the store (a census that disagrees with the window).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests._optional_extras import vec_unavailable
from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.similar import _knn

pytest.importorskip("sqlite_vec")

SPACE = EmbeddingSpace("a" * 64, "test-encoder:a", 3)
QUERY = [1.0, 0.0, 0.0]
VECTORS = {"L-a": [1.0, 0.0, 0.0], "L-b": [0.6, 0.8, 0.0]}


@pytest.fixture()
def stores(tmp_path: Path) -> Iterator[tuple[SQLiteBackend, SQLiteBackend]]:
    reader = SQLiteBackend(tmp_path / "m.db", dim=3)
    if not reader.supports_vectors():
        reader.close()
        vec_unavailable("sqlite-vec did not load")
    for entry_id, vector in VECTORS.items():
        entry = MemoryEntry(id=entry_id, content=f"content {entry_id}", namespace="default")
        reader.store(entry)
        proof = VectorProvenance.for_vector(SPACE, f"{entry.content} {entry.detail}", vector)
        reader.upsert_vector(entry_id, vector, namespace="default", provenance=proof)
    writer = SQLiteBackend(tmp_path / "m.db", dim=3)  # a second connection: another process's writes
    yield reader, writer
    writer.close()
    reader.close()


def test_a_delete_between_the_window_and_its_records_does_not_split_the_answer(
    stores: tuple[SQLiteBackend, SQLiteBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    reader, writer = stores
    records = reader.get_vector_records

    def interleaved(entry_ids: list[str], *, namespace: str) -> object:
        assert writer.delete("L-b", namespace="default")  # lands after the KNN window was read
        return records(entry_ids, namespace=namespace)

    monkeypatch.setattr(reader, "get_vector_records", interleaved)

    hits = _knn(reader, QUERY, SPACE, "default", 10)

    # The window's snapshot held both rows: the answer is that snapshot, not "incomplete".
    assert hits is not None
    assert [entry_id for entry_id, _similarity, _active in hits] == ["L-a", "L-b"]
    assert reader.get("L-b", namespace="default") is None  # the delete did commit


def test_the_sqlite_vec_knn_reads_inside_the_snapshot(stores: tuple[SQLiteBackend, SQLiteBackend]) -> None:
    reader, writer = stores
    with reader.read_snapshot():
        assert reader.count(namespace="default") == 2  # the snapshot starts here
        assert writer.delete("L-b", namespace="default")
        assert {entry_id for entry_id, _ in reader.search_vectors(QUERY, namespace="default")} == {"L-a", "L-b"}
    assert {entry_id for entry_id, _ in reader.search_vectors(QUERY, namespace="default")} == {"L-a"}
