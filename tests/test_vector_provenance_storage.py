"""Vector provenance is atomic, qualified, and never invented for legacy bytes."""

from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path

import pytest

from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.storage._schema import _migrate_v6_vector_provenance, ensure_schema
from trw_memory.storage._vector_ops import get_vector_records
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _proof(vector: list[float]) -> VectorProvenance:
    return VectorProvenance.for_vector(
        EmbeddingSpace("a" * 64, "encode-normalized-v1", len(vector)), "document", vector
    )


@pytest.fixture
def backend(tmp_path: Path):
    pytest.importorskip("sqlite_vec")
    backend = SQLiteBackend(tmp_path / "memory.db", dim=2)
    if not backend.vec_available:
        backend.close()
        pytest.skip("sqlite-vec virtual table unavailable")
    yield backend
    backend.close()


def test_v5_migration_preserves_vector_bytes_and_is_idempotent(backend) -> None:
    backend.upsert_vector("entry", [1.0, 0.0], namespace="default")
    before = backend._conn.execute("SELECT embedding FROM vec_memories").fetchone()[0]
    backend._conn.execute("ALTER TABLE vec_index DROP COLUMN provenance_json")
    backend._conn.execute("PRAGMA user_version = 5")
    backend._conn.commit()
    ensure_schema(backend._conn)
    ensure_schema(backend._conn)
    assert backend._conn.execute("PRAGMA user_version").fetchone()[0] == 6
    assert backend._conn.execute("SELECT embedding FROM vec_memories").fetchone()[0] == before
    assert backend.get_vector_records(["entry"], namespace="default")["entry"].provenance is None


def test_qualified_write_and_read_and_legacy_clear(backend) -> None:
    vector = [0.6, 0.8]
    proof = _proof(vector)
    backend.upsert_vector("entry", vector, namespace="default", provenance=proof)
    record = backend.get_vector_records(["entry"], namespace="default")["entry"]
    assert record.embedding == pytest.approx(vector)
    assert record.provenance == proof
    backend.upsert_vector("entry", vector, namespace="default")
    assert backend.get_vector_records(["entry"], namespace="default")["entry"].provenance is None


def test_mismatched_proof_rejected_before_mutation(backend) -> None:
    proof = _proof([1.0, 0.0])
    backend.upsert_vector("entry", [1.0, 0.0], namespace="default", provenance=proof)
    with pytest.raises(ValueError, match="provenance"):
        backend.upsert_vector("entry", [0.0, 1.0], namespace="default", provenance=proof)
    assert backend.get_vector_records(["entry"], namespace="default")["entry"].provenance == proof


def test_old_sql_writer_cannot_reuse_stale_proof(backend) -> None:
    backend.upsert_vector("entry", [1.0, 0.0], namespace="default", provenance=_proof([1.0, 0.0]))
    rowid = backend._conn.execute("SELECT rowid FROM vec_index").fetchone()[0]
    backend._conn.execute("DELETE FROM vec_memories WHERE rowid=?", (rowid,))
    backend._conn.execute("INSERT INTO vec_memories(rowid,embedding) VALUES(?,?)", (rowid, struct.pack("2f", 0.0, 1.0)))
    backend._conn.commit()
    record = backend.get_vector_records(["entry"], namespace="default")["entry"]
    assert record.embedding == (0.0, 1.0)
    assert record.provenance is None


def test_transaction_rollback_restores_vector_and_proof(backend) -> None:
    before = _proof([1.0, 0.0])
    backend.upsert_vector("entry", [1.0, 0.0], namespace="default", provenance=before)
    with pytest.raises(RuntimeError), backend.transaction():
        backend.upsert_vector("entry", [0.0, 1.0], namespace="default", provenance=_proof([0.0, 1.0]))
        raise RuntimeError("abort")
    record = backend.get_vector_records(["entry"], namespace="default")["entry"]
    assert record.embedding == (1.0, 0.0)
    assert record.provenance == before


def test_namespace_collision_keeps_proof_with_its_vector(backend) -> None:
    for namespace, vector in [("default", [1.0, 0.0]), ("user:other", [0.0, 1.0])]:
        backend.upsert_vector("same", vector, namespace=namespace, provenance=_proof(vector))
    assert backend.get_vector_records(["same"], namespace="default")["same"].provenance == _proof([1.0, 0.0])
    assert backend.get_vector_records(["same"], namespace="user:other")["same"].provenance == _proof([0.0, 1.0])
    assert backend.get_vector_records(["same"], namespace="") == {}
    with pytest.raises(TypeError):
        backend.get_vector_records(["same"], namespace=None)


@pytest.mark.parametrize("raw", [None, "{}", "not-json", '{"version":2}'])
def test_malformed_proof_is_unknown(backend, raw) -> None:
    backend.upsert_vector("entry", [1.0, 0.0], namespace="default")
    backend._conn.execute("UPDATE vec_index SET provenance_json=?", (raw,))
    assert backend.get_vector_records(["entry"], namespace="default")["entry"].provenance is None


def test_legacy_read_without_proof_column_is_unknown() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE vec_index(rowid INTEGER PRIMARY KEY, entry_id TEXT, namespace TEXT)")
        conn.execute("CREATE TABLE vec_memories(rowid INTEGER PRIMARY KEY, embedding BLOB)")
        conn.execute("INSERT INTO vec_index VALUES(1,'entry','default')")
        conn.execute("INSERT INTO vec_memories VALUES(1,?)", (struct.pack("2f", 1.0, 0.0),))
        record = get_vector_records(
            conn, threading.Lock(), vec_available=True, entry_ids=["entry"], namespace="default"
        )["entry"]
        assert record.embedding == (1.0, 0.0)
        assert record.provenance is None
        _migrate_v6_vector_provenance(conn.cursor())
        _migrate_v6_vector_provenance(conn.cursor())
        assert conn.execute("SELECT provenance_json FROM vec_index").fetchone()[0] is None
    finally:
        conn.close()


def test_failed_replacement_preserves_both_when_outer_transaction_commits(backend) -> None:
    from trw_memory.storage._vector_ops import upsert_vector

    before = _proof([1.0, 0.0])
    backend.upsert_vector("entry", [1.0, 0.0], namespace="default", provenance=before)

    class FailVectorInsert:
        @property
        def in_transaction(self):
            return backend._conn.in_transaction

        def execute(self, sql, params=()):
            if sql.startswith("INSERT INTO vec_memories"):
                raise sqlite3.OperationalError("dimension mismatch")
            return backend._conn.execute(sql, params)

    with backend.transaction():
        backend._conn.execute("CREATE TABLE outer_work(value TEXT)")
        backend._conn.execute("INSERT INTO outer_work VALUES('preserved')")
        upsert_vector(
            FailVectorInsert(),
            backend._lock,
            vec_available=True,
            dim=2,
            entry_id="entry",
            namespace="default",
            embedding=[0.0, 1.0],
            provenance=_proof([0.0, 1.0]),
            skip_commit=True,
        )
    record = backend.get_vector_records(["entry"], namespace="default")["entry"]
    assert record.embedding == (1.0, 0.0)
    assert record.provenance == before
    assert backend._conn.execute("SELECT value FROM outer_work").fetchone()[0] == "preserved"
