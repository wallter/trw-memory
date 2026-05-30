"""Transaction-atomicity tests for the coupled store-path cluster (S1/S2/S3/S9).

Covers:
- S2: ``transaction()`` raises the skip-commit gate INSIDE the BEGIN IMMEDIATE
  lock so a concurrent writer in that window cannot read depth==0 and commit
  the outer transaction prematurely. Asserted structurally (lock-held during
  the depth bump) — not via a flaky timing race.
- S9: ``store()`` suppresses its commit while inside a ``transaction()`` block.
- S3: ``upsert_vector()`` suppresses its commit while inside a transaction
  (``skip_commit`` flag) so the vector lands at the outermost COMMIT.
- S1: a row + its vector written inside ``transaction()`` commit exactly once
  and are atomic — both present on success, neither on a mid-transaction error.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import pytest

from tests.conftest import make_entry
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# S2 — depth gate raised under the BEGIN-IMMEDIATE lock (structural invariant)
# ---------------------------------------------------------------------------


def test_transaction_outer_bumps_depth_while_lock_held(tmp_path: Path) -> None:
    """S2: in the outer case the depth increment happens while ``_lock`` is held.

    We wrap the real lock in a recording proxy and snapshot
    ``_skip_commit_depth`` on every acquire/release. The window where the lock
    is held during entry must include the depth transition 0 -> 1; if the bump
    were outside the lock (the bug), the depth would still read 0 at release.
    """
    backend = SQLiteBackend(tmp_path / "s2.db")
    try:
        real_lock = backend._lock
        depth_at_release: list[int] = []

        class _RecordingLock:
            def __enter__(self) -> bool:
                return bool(real_lock.__enter__())

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                # Snapshot the gate value *before* the lock is released.
                depth_at_release.append(backend._skip_commit_depth)
                real_lock.__exit__(exc_type, exc, tb)

        backend._lock = _RecordingLock()  # type: ignore[assignment]

        with backend.transaction():
            assert backend._skip_commit_depth == 1

        # The first lock release inside transaction() is the BEGIN IMMEDIATE
        # block — the depth must already be 1 at that release (bump was inside).
        assert depth_at_release[0] == 1
    finally:
        backend._lock = real_lock
        backend.close()


def test_transaction_nested_depth_is_balanced(tmp_path: Path) -> None:
    """Nested transactions only the outer issues BEGIN/COMMIT; depth balances to 0."""
    backend = SQLiteBackend(tmp_path / "nested.db")
    try:
        assert backend._skip_commit_depth == 0
        with backend.transaction():
            assert backend._skip_commit_depth == 1
            with backend.transaction():
                assert backend._skip_commit_depth == 2
            assert backend._skip_commit_depth == 1
        assert backend._skip_commit_depth == 0
    finally:
        backend.close()


def test_transaction_depth_restored_after_exception(tmp_path: Path) -> None:
    """An error inside transaction() rolls back and restores depth to 0."""
    backend = SQLiteBackend(tmp_path / "err.db")
    try:
        with pytest.raises(RuntimeError, match="boom"):
            with backend.transaction():
                assert backend._skip_commit_depth == 1
                raise RuntimeError("boom")
        assert backend._skip_commit_depth == 0
        # Backend is still usable after the rollback.
        backend.store(make_entry(entry_id="M-after-err"))
        assert backend.get("M-after-err") is not None
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# S9 — store() defers commit inside a transaction
# ---------------------------------------------------------------------------


def test_store_inside_transaction_defers_commit(tmp_path: Path) -> None:
    """S9: a store() inside transaction() is not visible until the outer COMMIT.

    We use a SECOND read-only connection to the same on-disk DB to observe
    commit visibility independently of the writer connection.
    """
    import sqlite3

    db_path = tmp_path / "s9.db"
    backend = SQLiteBackend(db_path)
    try:
        observer = sqlite3.connect(str(db_path))
        try:
            with backend.transaction():
                backend.store(make_entry(entry_id="M-defer"))
                # Mid-transaction: the writer staged the row but has NOT
                # committed — the observer connection must not see it yet.
                seen_mid = observer.execute(
                    "SELECT COUNT(*) FROM memories WHERE id = ?", ("M-defer",)
                ).fetchone()[0]
                assert seen_mid == 0, "store() committed prematurely inside transaction()"

            # After the outermost COMMIT the row is durable and visible.
            seen_after = observer.execute(
                "SELECT COUNT(*) FROM memories WHERE id = ?", ("M-defer",)
            ).fetchone()[0]
            assert seen_after == 1
        finally:
            observer.close()
    finally:
        backend.close()


def test_store_standalone_commits_immediately(tmp_path: Path) -> None:
    """A store() outside any transaction commits right away (regression guard)."""
    import sqlite3

    db_path = tmp_path / "s9b.db"
    backend = SQLiteBackend(db_path)
    try:
        backend.store(make_entry(entry_id="M-now"))
        observer = sqlite3.connect(str(db_path))
        try:
            count = observer.execute(
                "SELECT COUNT(*) FROM memories WHERE id = ?", ("M-now",)
            ).fetchone()[0]
            assert count == 1
        finally:
            observer.close()
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# S1 + S3 — row + vector atomic, exactly one commit
# ---------------------------------------------------------------------------


def _vec_entry(entry_id: str) -> MemoryEntry:
    return make_entry(entry_id=entry_id, content="vector content", detail="d")


def test_store_and_vector_in_transaction_are_atomic_on_success(tmp_path: Path) -> None:
    """S1/S3: row + vector written in one transaction are both present after COMMIT."""
    pytest.importorskip("sqlite_vec")
    db_path = tmp_path / "s1ok.db"
    backend = SQLiteBackend(db_path)
    if not backend._vec_available:
        backend.close()
        pytest.skip("sqlite-vec extension not available")
    try:
        emb = [1.0] * backend._dim
        entry = _vec_entry("M-atomic")
        with backend.transaction():
            backend.store(entry)
            backend.upsert_vector(entry.id, emb)
        assert backend.get("M-atomic") is not None
        assert backend.vector_exists("M-atomic") is True
    finally:
        backend.close()


def test_store_and_vector_rollback_leaves_neither(tmp_path: Path) -> None:
    """S1/S3: an error after store()+upsert inside transaction() rolls back BOTH.

    Without the fix the row and vector would have been committed by their own
    unconditional commits before the error, leaving orphaned state.
    """
    pytest.importorskip("sqlite_vec")
    import sqlite3

    db_path = tmp_path / "s1rollback.db"
    backend = SQLiteBackend(db_path)
    if not backend._vec_available:
        backend.close()
        pytest.skip("sqlite-vec extension not available")
    try:
        emb = [1.0] * backend._dim
        entry = _vec_entry("M-rollback")
        with pytest.raises(RuntimeError, match="mid-tx"):
            with backend.transaction():
                backend.store(entry)
                backend.upsert_vector(entry.id, emb)
                # Simulate a crash/error AFTER both writes but BEFORE COMMIT.
                raise RuntimeError("mid-tx")

        # Neither the row nor the vector survived — observed via a fresh
        # connection so we are reading committed state only.
        observer = sqlite3.connect(str(db_path))
        try:
            row_count = observer.execute(
                "SELECT COUNT(*) FROM memories WHERE id = ?", ("M-rollback",)
            ).fetchone()[0]
            assert row_count == 0, "row survived a rolled-back transaction"
        finally:
            observer.close()
        assert backend.get("M-rollback") is None
        assert backend.vector_exists("M-rollback") is False
    finally:
        backend.close()


def test_upsert_vector_standalone_still_commits(tmp_path: Path) -> None:
    """S3 regression: outside a transaction, upsert_vector() commits immediately."""
    pytest.importorskip("sqlite_vec")
    db_path = tmp_path / "s3standalone.db"
    backend = SQLiteBackend(db_path)
    if not backend._vec_available:
        backend.close()
        pytest.skip("sqlite-vec extension not available")
    try:
        emb = [1.0] * backend._dim
        entry = _vec_entry("M-vec-now")
        backend.store(entry)
        backend.upsert_vector(entry.id, emb)
        assert backend.vector_exists("M-vec-now") is True
    finally:
        backend.close()


def test_transaction_commits_exactly_once(tmp_path: Path) -> None:
    """A store()+upsert batch inside transaction() issues exactly ONE commit.

    Counts commit() calls on the live connection via a thin delegating proxy
    (the C-extension Connection's ``commit`` attribute is read-only, so it
    cannot be patched in place): store() (S9) and upsert_vector() (S3) both
    defer, so only the outermost transaction COMMIT fires — proving the writes
    are coalesced rather than triple-committed.
    """
    pytest.importorskip("sqlite_vec")
    db_path = tmp_path / "once.db"
    backend = SQLiteBackend(db_path)
    if not backend._vec_available:
        backend.close()
        pytest.skip("sqlite-vec extension not available")
    try:
        real_conn = backend._conn
        commit_calls = {"n": 0}

        class _CommitCountingConn:
            def commit(self) -> None:
                commit_calls["n"] += 1
                real_conn.commit()

            def __getattr__(self, name: str) -> object:
                return getattr(real_conn, name)

        backend._conn = _CommitCountingConn()
        try:
            emb = [1.0] * backend._dim
            entry = _vec_entry("M-once")
            with backend.transaction():
                backend.store(entry)
                backend.upsert_vector(entry.id, emb)
        finally:
            backend._conn = real_conn

        assert commit_calls["n"] == 1, f"expected exactly one commit, got {commit_calls['n']}"
    finally:
        backend.close()
