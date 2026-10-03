"""B71-83 / A14: ``vec_index.space_key`` (schema 9) makes the space census one SQL ``GROUP BY``.

The census and the dense-recall gate used to parse every row's provenance JSON in Python (~12 us/row).
Schema 9 stores a digest of each vector's claimed :class:`EmbeddingSpace` beside its provenance, so both
read an index instead. These tests pin the new code to the OLD implementation's answers, kept here as the
reference: the same census and the same gate verdict on a mixed store (several spaces, NULL and malformed
provenance, orphan vectors, and a row with two vectors), and a v8 store's backfill.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage import _schema
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend

pytest.importorskip("sqlite_vec")

SPACE_A = EmbeddingSpace("a" * 64, "test-encoder:a", 3, model_id="model-a")
SPACE_B = EmbeddingSpace("b" * 64, "test-encoder:b", 3)
SPACE_A_OTHER_ENCODING = EmbeddingSpace("a" * 64, "test-encoder:a-query-prefixed", 3)
VECTOR = [1.0, 0.0, 0.0]


def _old_census(conn: sqlite3.Connection, namespace: str) -> dict[EmbeddingSpace | None, int]:
    """The pre-schema-9 census, verbatim: parse every existing row's provenance, count distinct ids per space."""
    rows = conn.execute(
        "SELECT v.entry_id, v.provenance_json FROM vec_index v "
        "JOIN memories m ON m.namespace = v.namespace AND m.id = v.entry_id WHERE v.namespace = ?",
        (namespace,),
    ).fetchall()
    census: dict[EmbeddingSpace | None, set[str]] = {}
    for entry_id, raw in rows:
        proof = VectorProvenance.from_json(raw)
        census.setdefault(proof.space if proof is not None else None, set()).add(entry_id)
    return {space: len(ids) for space, ids in census.items()}


def _old_proves(census: object, space: EmbeddingSpace, *, rows: int) -> bool:
    """The pre-schema-9 ``_space_gate._census_proves``, verbatim."""
    if not isinstance(census, dict) or not census:
        return False
    counts = list(census.values())
    if not all(type(count) is int and count > 0 for count in counts):
        return False
    return all(key == space for key in census) and sum(counts) >= rows


def _new_proves(backend: SQLiteBackend, space: EmbeddingSpace, namespace: str) -> bool:
    proven = backend.vectors_proven_in_space(namespace=namespace, space=space)
    return type(proven) is int and 0 < proven >= backend.count(namespace=namespace)


def _put(backend: SQLiteBackend, entry_id: str, space: EmbeddingSpace | None, *, row: bool = True) -> None:
    if row:
        backend.store(MemoryEntry(id=entry_id, content=f"content {entry_id}", namespace="default"))
    proof = VectorProvenance.for_vector(space, entry_id, VECTOR) if space else None
    backend.upsert_vector(entry_id, VECTOR, namespace="default", provenance=proof)


def _drop_unique(conn: sqlite3.Connection) -> None:
    """A pre-schema-5 layout: ``vec_index`` without ``UNIQUE (namespace, entry_id)``, so a row can hold two vectors."""
    conn.executescript(
        """
        CREATE TABLE vec_index_loose (rowid INTEGER PRIMARY KEY AUTOINCREMENT, entry_id TEXT NOT NULL,
            namespace TEXT NOT NULL DEFAULT 'default', provenance_json TEXT DEFAULT NULL, space_key TEXT DEFAULT NULL);
        INSERT INTO vec_index_loose SELECT rowid, entry_id, namespace, provenance_json, space_key FROM vec_index;
        DROP TABLE vec_index;
        ALTER TABLE vec_index_loose RENAME TO vec_index;
        """
    )


def _second_vector(conn: sqlite3.Connection, entry_id: str, space: EmbeddingSpace) -> None:
    proof = VectorProvenance.for_vector(space, entry_id, VECTOR)
    conn.execute(
        "INSERT INTO vec_index(entry_id, namespace, provenance_json, space_key) VALUES (?, 'default', ?, ?)",
        (entry_id, proof.to_json(), space.key),
    )


@pytest.fixture()
def backend(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = SQLiteBackend(tmp_path / "space-key.db", dim=3)
    if not store.supports_vectors():
        store.close()
        pytest.skip("sqlite-vec did not load")
    yield store
    store.close()


def _mixed(backend: SQLiteBackend) -> None:
    for entry_id, space in [("a1", SPACE_A), ("a2", SPACE_A), ("b1", SPACE_B), ("enc", SPACE_A_OTHER_ENCODING)]:
        _put(backend, entry_id, space)
    _put(backend, "legacy", None)
    _put(backend, "malformed", SPACE_A)
    _put(backend, "orphan-a", SPACE_A, row=False)
    _put(backend, "orphan-b", SPACE_B, row=False)
    backend.store(MemoryEntry(id="bare", content="content bare", namespace="default"))
    conn = backend._conn
    conn.execute("UPDATE vec_index SET provenance_json = '{not json', space_key = NULL WHERE entry_id = 'malformed'")
    _drop_unique(conn)
    _second_vector(conn, "a1", SPACE_A)  # same space twice: one row
    _second_vector(conn, "b1", SPACE_A)  # a row in two spaces
    conn.commit()


def test_the_key_is_the_full_identity_and_never_the_model_name() -> None:
    assert SPACE_A.key == EmbeddingSpace("a" * 64, "test-encoder:a", 3).key
    assert (
        len({SPACE_A.key, SPACE_B.key, SPACE_A_OTHER_ENCODING.key, EmbeddingSpace("a" * 64, "test-encoder:a", 4).key})
        == 4
    )
    assert int(SPACE_A.key, 16) >= 0 and len(SPACE_A.key) == 64


def test_the_writer_records_the_key_of_the_space_it_stores(backend: SQLiteBackend) -> None:
    _put(backend, "a1", SPACE_A)
    _put(backend, "legacy", None)
    _put(backend, "a1-cleared", SPACE_A)
    _put(backend, "a1-cleared", None)  # a legacy rewrite clears the claim with its proof

    keys = dict(backend._conn.execute("SELECT entry_id, space_key FROM vec_index").fetchall())
    assert keys == {"a1": SPACE_A.key, "legacy": None, "a1-cleared": None}


def test_census_matches_the_old_implementation_on_a_mixed_store(backend: SQLiteBackend) -> None:
    _mixed(backend)

    expected = _old_census(backend._conn, "default")
    assert expected == {SPACE_A: 3, SPACE_B: 1, SPACE_A_OTHER_ENCODING: 1, None: 2}
    assert backend.vector_space_census(namespace="default") == expected
    assert backend.vector_space_census(namespace="nobody") == {} == _old_census(backend._conn, "nobody")


def test_a_key_its_sample_proof_does_not_reproduce_counts_as_unknown(backend: SQLiteBackend) -> None:
    """Fail closed: a group whose provenance cannot rebuild its own key is never read as a real space."""
    _put(backend, "a1", SPACE_A)
    backend._conn.execute("UPDATE vec_index SET space_key = ?", (SPACE_B.key,))
    backend._conn.commit()

    assert backend.vector_space_census(namespace="default") == {None: 1}


@pytest.mark.parametrize(
    ("rows", "space"),
    [
        ([("a1", SPACE_A), ("a2", SPACE_A)], SPACE_A),
        ([("a1", SPACE_A), ("a2", SPACE_A)], SPACE_B),
        ([("a1", SPACE_A), ("bare", "no-vector")], SPACE_A),
        ([("a1", SPACE_A), ("orphan", "orphan-b")], SPACE_A),
        ([("a1", SPACE_A), ("orphan", "orphan-a"), ("bare", "no-vector")], SPACE_A),
        ([("a1", SPACE_A), ("b1", SPACE_B)], SPACE_A),
        ([("a1", SPACE_A), ("legacy", None)], SPACE_A),
        ([("a1", SPACE_A), ("bad", "malformed")], SPACE_A),
        ([("a1", SPACE_A), ("a1", "second-b")], SPACE_A),
        ([("a1", SPACE_A), ("a1", "second-a")], SPACE_A),
        ([("bare", "no-vector")], SPACE_A),
        ([], SPACE_A),
    ],
)
def test_the_gate_gives_the_old_verdict(
    backend: SQLiteBackend, rows: list[tuple[str, object]], space: EmbeddingSpace
) -> None:
    for entry_id, kind in rows:
        if kind == "no-vector":
            backend.store(MemoryEntry(id=entry_id, content=f"content {entry_id}", namespace="default"))
        elif kind in ("orphan-a", "orphan-b"):
            _put(backend, entry_id, SPACE_A if kind == "orphan-a" else SPACE_B, row=False)
        elif kind == "malformed":
            _put(backend, entry_id, SPACE_A)
            backend._conn.execute(
                "UPDATE vec_index SET provenance_json = '[]', space_key = NULL WHERE entry_id = ?", (entry_id,)
            )
        elif kind in ("second-a", "second-b"):
            _drop_unique(backend._conn)
            _second_vector(backend._conn, entry_id, SPACE_A if kind == "second-a" else SPACE_B)
        else:
            _put(backend, entry_id, kind)  # type: ignore[arg-type]
    backend._conn.commit()
    old = _old_proves(_old_census(backend._conn, "default"), space, rows=backend.count(namespace="default"))

    assert _new_proves(backend, space, "default") is old


def test_backends_without_the_capability_prove_nothing() -> None:
    assert StorageBackend.vectors_proven_in_space(object(), namespace="default", space=SPACE_A) is None  # type: ignore[arg-type]


def test_the_gate_answers_none_without_sqlite_vec_or_on_a_sql_error() -> None:
    import threading
    from unittest.mock import MagicMock

    from trw_memory.storage._vector_provenance_reads import vectors_proven_in_space

    conn = MagicMock()
    assert vectors_proven_in_space(conn, threading.RLock(), False, "d", "k") is None
    conn.execute.assert_not_called()
    conn.execute.side_effect = sqlite3.OperationalError("no such column: space_key")
    assert vectors_proven_in_space(conn, threading.RLock(), True, "d", "k") is None
    assert conn.execute.call_count == 1

    # Control: with sqlite-vec available and no SQL error the same call answers the proven count.
    live = MagicMock()
    live.execute.return_value.fetchone.return_value = (3,)
    assert vectors_proven_in_space(live, threading.RLock(), True, "d", "k") == 3
    assert live.execute.call_args.args[1] == ("d", "k")


def _as_v8(path: Path) -> None:
    """Rewind a store to the schema-8 shape: no ``space_key`` column, no index, stamped 8."""
    with sqlite3.connect(path) as conn:
        conn.execute("DROP INDEX idx_vec_index_space")
        conn.execute("ALTER TABLE vec_index DROP COLUMN space_key")
        conn.execute("PRAGMA user_version = 8")


def test_the_v9_migration_backfills_through_from_json_and_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v8.db"
    store = SQLiteBackend(path, dim=3)
    if not store.supports_vectors():
        store.close()
        pytest.skip("sqlite-vec did not load")
    for index in range(7):
        _put(store, f"a{index}", SPACE_A if index % 2 else SPACE_B)
    _put(store, "legacy", None)
    _put(store, "malformed", SPACE_A)
    _put(store, "orphan", SPACE_B, row=False)
    store._conn.execute("UPDATE vec_index SET provenance_json = '{\"space\": 1}' WHERE entry_id = 'malformed'")
    store._conn.commit()
    expected_census = _old_census(store._conn, "default")
    store.close()
    _as_v8(path)

    reopened = SQLiteBackend(path, dim=3)
    try:
        conn = reopened._conn
        # PRD-CORE-330 bumped SCHEMA_VERSION past 9 (memories_fts_rowid, schema 10) --
        # this assertion only needs "stamped to the current version", not a literal 9.
        assert conn.execute("PRAGMA user_version").fetchone()[0] == _schema.SCHEMA_VERSION
        keys = conn.execute("SELECT entry_id, provenance_json, space_key FROM vec_index").fetchall()
        for _entry_id, raw, key in keys:
            proof = VectorProvenance.from_json(raw)
            assert key == (proof.space.key if proof is not None else None)
        assert reopened.vector_space_census(namespace="default") == expected_census
        _schema._migrate_v9_vector_space_key(conn.cursor())  # a rerun changes nothing
        assert conn.execute("SELECT entry_id, provenance_json, space_key FROM vec_index").fetchall() == keys
    finally:
        reopened.close()


def test_the_v9_migration_is_a_no_op_without_a_vector_table(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "plain.db")
    try:
        _schema._migrate_v9_vector_space_key(conn.cursor())
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'vec_index'").fetchall() == []
    finally:
        conn.close()


def test_the_gate_proves_one_namespace_regardless_of_anothers_keys(backend: SQLiteBackend) -> None:
    """Pre-review: another namespace's vectors in another space do not spoil (or satisfy) this one's proof."""
    _put(backend, "a1", SPACE_A)
    backend.store(MemoryEntry(id="x1", content="elsewhere", namespace="project:other"))
    backend.upsert_vector(
        "x1", VECTOR, namespace="project:other", provenance=VectorProvenance.for_vector(SPACE_B, "x1", VECTOR)
    )

    assert _new_proves(backend, SPACE_A, "default")
    assert not _new_proves(backend, SPACE_A, "project:other")


def test_a_valid_proof_with_a_missing_key_fails_closed(backend: SQLiteBackend) -> None:
    """Pre-review: a row whose key was never written (a raw write) proves nothing, whatever its provenance says."""
    _put(backend, "a1", SPACE_A)
    backend._conn.execute("UPDATE vec_index SET space_key = NULL WHERE entry_id = 'a1'")

    assert not _new_proves(backend, SPACE_A, "default")
