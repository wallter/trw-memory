"""PRD-CORE-272: legacy cleanup is namespace-scoped and atomic, never a purge."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.conftest import make_entry
from trw_memory.storage.sqlite_backend import SQLiteBackend


@pytest.fixture
def backend(tmp_path: Path):
    pytest.importorskip("sqlite_vec")
    backend = SQLiteBackend(tmp_path / "legacy.db")
    assert backend.supports_vectors()
    yield backend
    backend.close()


def _seed(backend, parent="P", namespace="default"):
    backend.store(make_entry(entry_id=parent, namespace=namespace))
    backend.upsert_vector(parent, [0.1] * 384, namespace=namespace)
    backend.upsert_vector(f"{parent}#hype0", [0.2] * 384, namespace=namespace)


def _snapshot(backend):
    return {
        table: backend._conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        for table in ("memories", "vec_index", "vec_memories")
    }


@pytest.mark.parametrize("parent", ["P", "P%", "P_", "P\\", "café", "P#hype0"])
def test_cleanup_exact_namespace_membership(backend, parent):
    _seed(backend, parent)
    _seed(backend, parent, "other")
    # A canonical collision in another namespace must NOT block local cleanup.
    backend.store(make_entry(entry_id=f"{parent}#hype0", namespace="other"))
    for suffix in ("#hype1", "#hype\u0660", "#hypevictim#hype0"):
        backend.upsert_vector(parent + suffix, [0.3] * 384, namespace="default")
    backend.store(make_entry(entry_id=parent + "#hype1"))
    before = _snapshot(backend)
    assert backend.hype_sibling_ids(parent, namespace="default") == [parent + "#hype0"]
    assert backend.delete_hype_siblings(parent, namespace="default") == 1
    assert backend.delete_hype_siblings(parent, namespace="default") == 0
    assert backend.vector_exists(parent + "#hype0", namespace="other")
    for suffix in ("", "#hype1", "#hype\u0660", "#hypevictim#hype0"):
        assert backend.vector_exists(parent + suffix, namespace="default")
    assert _snapshot(backend)["memories"] == before["memories"]


def test_orphans_untouched_and_namespace_required(backend):
    backend.upsert_vector("absent#hype0", [0.1] * 384, namespace="default")
    assert backend.delete_hype_siblings("absent", namespace="default") == 0
    assert backend.vector_exists("absent#hype0", namespace="default")
    with pytest.raises(TypeError):
        backend.delete_hype_siblings("absent")


def test_cleanup_failure_rolls_back_and_reopens(backend, monkeypatch):
    from trw_memory.storage import _vector_ops

    _seed(backend)
    backend.upsert_vector("P#hype1", [0.3] * 384, namespace="default")
    before = _snapshot(backend)
    original = _vector_ops.delete_vector_internal
    calls = []

    def fail_after_delete(conn, entry_id, namespace, *, allow_unavailable):
        original(conn, entry_id, namespace, allow_unavailable=allow_unavailable)
        calls.append(entry_id)
        raise sqlite3.OperationalError("injected interruption")

    with monkeypatch.context() as patch:
        patch.setattr(_vector_ops, "delete_vector_internal", fail_after_delete)
        with pytest.raises(sqlite3.OperationalError, match="interruption"):
            backend.delete_hype_siblings("P", namespace="default")
    assert calls == ["P#hype0"]
    assert _snapshot(backend) == before
    path = backend._db_path
    backend.close()
    reopened = SQLiteBackend(path)
    try:
        assert _snapshot(reopened) == before
        assert reopened.delete_hype_siblings("P", namespace="default") == 2
        assert reopened.delete_hype_siblings("P", namespace="default") == 0
        assert _snapshot(reopened)["memories"] == before["memories"]
    finally:
        reopened.close()


def test_query_errors_propagate(backend):
    _seed(backend)
    backend._conn.execute("DROP TABLE vec_index")
    with pytest.raises(sqlite3.OperationalError):
        backend.hype_sibling_ids("P", namespace="default")


def test_no_vector_capability_is_not_success(tmp_path):
    from trw_memory.storage.yaml_backend import YAMLBackend

    backend = YAMLBackend(tmp_path / "yaml")
    assert not backend.supports_vectors()
    with pytest.raises(NotImplementedError, match="unavailable"):
        backend.delete_hype_siblings("P", namespace="default")


def test_direct_storage_helper_accepts_nonreentrant_lock(backend):
    import threading

    from trw_memory.storage._vector_ops import delete_hype_siblings

    _seed(backend)

    # A lock that rejects reentry makes regression fail, rather than hang pytest.
    class CheckedLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            assert self.lock.acquire(blocking=False), "helper acquired nonreentrant lock twice"

        def __exit__(self, *args):
            self.lock.release()

    with backend.transaction():
        assert (
            delete_hype_siblings(
                backend._conn, CheckedLock(), vec_available=True, parent_id="P", namespace="default", skip_commit=True
            )
            == 1
        )


def test_cleanup_preserves_exact_primary_and_other_namespace_vector_rows(backend):
    _seed(backend)
    _seed(backend, namespace="other")
    before = _snapshot(backend)
    # Capture the selected row's key; all remaining rows must be byte-identical.
    selected_rowid = backend._conn.execute(
        "SELECT rowid FROM vec_index WHERE entry_id = ? AND namespace = ?", ("P#hype0", "default")
    ).fetchone()[0]
    assert backend.delete_hype_siblings("P", namespace="default") == 1
    after = _snapshot(backend)
    assert after["memories"] == before["memories"]
    assert len(after["vec_index"]) == len(before["vec_index"]) - 1
    assert len(after["vec_memories"]) == len(before["vec_memories"]) - 1
    assert all(row in before["vec_index"] for row in after["vec_index"])
    assert all(row in before["vec_memories"] for row in after["vec_memories"])


def test_vec0_disappearing_cannot_report_successful_cleanup(backend):
    _seed(backend)
    before = _snapshot(backend)
    real = backend._conn

    class MissingVecConnection:
        def execute(self, sql, *args):
            if sql.startswith("DELETE FROM vec_memories"):
                raise sqlite3.OperationalError("no such module: vec0")
            return real.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(real, name)

    backend._conn = MissingVecConnection()
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such module: vec0"):
            backend.delete_hype_siblings("P", namespace="default")
    finally:
        backend._conn = real
    assert _snapshot(backend) == before
    assert backend.hype_sibling_ids("P", namespace="default") == ["P#hype0"]


def test_missing_parent_uses_only_canonical_point_lookup(backend):
    # Unrelated and orphan vectors cannot turn first insertion into a namespace scan.
    backend.upsert_vector("absent#hype0", [0.1] * 384, namespace="default")
    statements = []
    backend._conn.set_trace_callback(statements.append)
    try:
        assert backend.hype_sibling_ids("absent", namespace="default") == []
    finally:
        backend._conn.set_trace_callback(None)
    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 1
    assert "FROM memories WHERE namespace" in selects[0]
    assert "vec_index" not in selects[0]
