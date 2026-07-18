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
- S8: ``delete_by_namespace`` removes entries + wiki_refs + vectors inside ONE
  transaction so a crash can never leave orphan wiki refs or vector rows.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from types import TracebackType

import pytest

from tests.conftest import make_entry
from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend


class _CommitFailingConn:
    """Connection proxy that fails commits and records rollback cleanup."""

    def __init__(self, real: sqlite3.Connection) -> None:
        self._real = real
        self.rolled_back = False

    def commit(self) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    def rollback(self) -> None:
        self.rolled_back = True
        self._real.rollback()

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# S2 — depth gate raised under the BEGIN-IMMEDIATE lock (structural invariant)
# ---------------------------------------------------------------------------


def test_transaction_outer_holds_lock_until_depth_is_restored(tmp_path: Path) -> None:
    """The connection lock spans the complete outer depth lifecycle.

    We wrap the real lock in a recording proxy and snapshot depth on release.
    A single release at restored depth zero proves the lock was not dropped
    between BEGIN and the final COMMIT/ROLLBACK cleanup.
    """
    backend = SQLiteBackend(tmp_path / "s2.db")
    real_lock = backend._lock  # bind before try so the finally can always restore it
    try:
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

        assert depth_at_release
        assert set(depth_at_release) == {0}
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
                seen_mid = observer.execute("SELECT COUNT(*) FROM memories WHERE id = ?", ("M-defer",)).fetchone()[0]
                assert seen_mid == 0, "store() committed prematurely inside transaction()"

            # After the outermost COMMIT the row is durable and visible.
            seen_after = observer.execute("SELECT COUNT(*) FROM memories WHERE id = ?", ("M-defer",)).fetchone()[0]
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
            count = observer.execute("SELECT COUNT(*) FROM memories WHERE id = ?", ("M-now",)).fetchone()[0]
            assert count == 1
        finally:
            observer.close()
    finally:
        backend.close()


@pytest.mark.parametrize("operation", ["store", "update", "delete"])
def test_wiki_side_effect_failure_rolls_back_canonical_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Canonical rows and their wiki references commit or roll back together."""
    import sqlite3

    from trw_memory.wiki import storage as wiki_storage

    db_path = tmp_path / f"wiki_{operation}.db"
    backend = SQLiteBackend(db_path)
    try:
        entry_id = f"M-wiki-{operation}"
        if operation != "store":
            backend.store(make_entry(entry_id=entry_id, content="before"))

        helper = "purge_wiki_refs_for_entry" if operation == "delete" else "replace_wiki_refs_for_entry"

        def fail_wiki_side_effect(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("wiki write failed")

        monkeypatch.setattr(wiki_storage, helper, fail_wiki_side_effect)
        with pytest.raises(RuntimeError, match="wiki write failed"):
            if operation == "store":
                backend.store(make_entry(entry_id=entry_id, content="new"))
            elif operation == "update":
                backend.update(entry_id, content="after", metadata={})
            else:
                backend.delete(entry_id)

        observer = sqlite3.connect(str(db_path))
        try:
            row = observer.execute("SELECT content FROM memories WHERE id = ?", (entry_id,)).fetchone()
        finally:
            observer.close()

        if operation == "store":
            assert row is None
        else:
            assert row == ("before",)
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# increment_*_counts / delete defer commit inside a transaction
# ---------------------------------------------------------------------------


def test_increment_session_counts_inside_transaction_defers_commit(tmp_path: Path) -> None:
    """increment_session_counts() must not commit prematurely in a transaction."""
    import sqlite3

    db_path = tmp_path / "inc_sess.db"
    backend = SQLiteBackend(db_path)
    try:
        backend.store(make_entry(entry_id="M-sess"))
        observer = sqlite3.connect(str(db_path))
        try:
            baseline = observer.execute(
                "SELECT COALESCE(session_count, 0) FROM memories WHERE id = ?", ("M-sess",)
            ).fetchone()[0]
            with backend.transaction():
                backend.increment_session_counts(["M-sess"])
                seen_mid = observer.execute(
                    "SELECT COALESCE(session_count, 0) FROM memories WHERE id = ?", ("M-sess",)
                ).fetchone()[0]
                assert seen_mid == baseline, "increment committed prematurely inside transaction()"
            seen_after = observer.execute(
                "SELECT COALESCE(session_count, 0) FROM memories WHERE id = ?", ("M-sess",)
            ).fetchone()[0]
            assert seen_after == baseline + 1
        finally:
            observer.close()
    finally:
        backend.close()


def test_increment_access_counts_inside_transaction_defers_commit(tmp_path: Path) -> None:
    """increment_access_counts() must not commit prematurely in a transaction."""
    import sqlite3

    db_path = tmp_path / "inc_acc.db"
    backend = SQLiteBackend(db_path)
    try:
        backend.store(make_entry(entry_id="M-acc"))
        observer = sqlite3.connect(str(db_path))
        try:
            baseline = observer.execute(
                "SELECT COALESCE(access_count, 0) FROM memories WHERE id = ?", ("M-acc",)
            ).fetchone()[0]
            with backend.transaction():
                backend.increment_access_counts(["M-acc"])
                seen_mid = observer.execute(
                    "SELECT COALESCE(access_count, 0) FROM memories WHERE id = ?", ("M-acc",)
                ).fetchone()[0]
                assert seen_mid == baseline, "increment committed prematurely inside transaction()"
            seen_after = observer.execute(
                "SELECT COALESCE(access_count, 0) FROM memories WHERE id = ?", ("M-acc",)
            ).fetchone()[0]
            assert seen_after == baseline + 1
        finally:
            observer.close()
    finally:
        backend.close()


def test_delete_inside_transaction_defers_commit(tmp_path: Path) -> None:
    """delete() must not commit prematurely in a transaction (and rolls back)."""
    import sqlite3

    db_path = tmp_path / "del_defer.db"
    backend = SQLiteBackend(db_path)
    try:
        backend.store(make_entry(entry_id="M-del"))
        observer = sqlite3.connect(str(db_path))
        try:
            with backend.transaction():
                backend.delete("M-del")
                seen_mid = observer.execute("SELECT COUNT(*) FROM memories WHERE id = ?", ("M-del",)).fetchone()[0]
                assert seen_mid == 1, "delete() committed prematurely inside transaction()"
            seen_after = observer.execute("SELECT COUNT(*) FROM memories WHERE id = ?", ("M-del",)).fetchone()[0]
            assert seen_after == 0
        finally:
            observer.close()
    finally:
        backend.close()


def test_delete_rolls_back_with_transaction_on_error(tmp_path: Path) -> None:
    """A delete() staged in a transaction that later raises is NOT persisted."""
    import sqlite3

    db_path = tmp_path / "del_rollback.db"
    backend = SQLiteBackend(db_path)
    try:
        backend.store(make_entry(entry_id="M-keep"))
        observer = sqlite3.connect(str(db_path))
        try:
            with pytest.raises(RuntimeError):
                with backend.transaction():
                    backend.delete("M-keep")
                    raise RuntimeError("boom")
            seen_after = observer.execute("SELECT COUNT(*) FROM memories WHERE id = ?", ("M-keep",)).fetchone()[0]
            assert seen_after == 1, "delete() persisted despite transaction rollback"
        finally:
            observer.close()
    finally:
        backend.close()


def test_delete_rolls_back_on_commit_failure_outside_transaction(tmp_path: Path) -> None:
    """trw-memory-13: when delete() runs OUTSIDE a transaction() and commit()
    raises, the partial multi-statement delete must be rolled back — not left
    dangling for a concurrent writer's commit to persist.
    """
    db_path = tmp_path / "del_commit_fail.db"
    backend = SQLiteBackend(db_path)

    try:
        backend.store(make_entry(entry_id="M-keep", content="searchable content"))

        real_conn = backend._conn
        proxy = _CommitFailingConn(real_conn)
        backend._conn = proxy  # type: ignore[assignment]

        with pytest.raises(StorageError):
            backend.delete("M-keep")

        # The except path rolled back the staged (uncommitted) delete.
        assert proxy.rolled_back, "delete() did not roll back after commit failure"

        # Restore the real connection and verify the row survived (delete
        # reverted) — no FTS/vec/graph orphan was committed.
        backend._conn = real_conn  # type: ignore[assignment]
        assert backend.get("M-keep") is not None
    finally:
        backend.close()


def test_increment_counts_are_bounded_at_max(tmp_path: Path) -> None:
    """Counters saturate at the cap instead of growing without bound."""
    from trw_memory.storage import _crud_ops

    db_path = tmp_path / "counter_cap.db"
    backend = SQLiteBackend(db_path)
    try:
        backend.store(make_entry(entry_id="M-cap"))
        # Seed the counters to exactly the cap.
        with backend._lock:
            backend._conn.execute(
                "UPDATE memories SET session_count = ?, access_count = ? WHERE id = ?",
                (_crud_ops._MAX_COUNTER, _crud_ops._MAX_COUNTER, "M-cap"),
            )
            backend._conn.commit()

        backend.increment_session_counts(["M-cap"])
        backend.increment_access_counts(["M-cap"])

        with backend._lock:
            session_count, access_count = backend._conn.execute(
                "SELECT session_count, access_count FROM memories WHERE id = ?", ("M-cap",)
            ).fetchone()
        assert session_count == _crud_ops._MAX_COUNTER
        assert access_count == _crud_ops._MAX_COUNTER
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
            row_count = observer.execute("SELECT COUNT(*) FROM memories WHERE id = ?", ("M-rollback",)).fetchone()[0]
            assert row_count == 0, "row survived a rolled-back transaction"
        finally:
            observer.close()
        assert backend.get("M-rollback") is None
        assert backend.vector_exists("M-rollback") is False
    finally:
        backend.close()


def test_delete_vector_inside_transaction_defers_commit(tmp_path: Path) -> None:
    """v0.9.2: delete_vector() inside transaction() rolls back with the tx.

    The public delete_vector() previously committed unconditionally — the same
    premature-commit bug class fixed in _crud_ops for v0.9.1 but missed here. If
    it still committed eagerly, the vector delete would survive a rolled-back
    transaction. We assert the vector is restored after rollback.
    """
    pytest.importorskip("sqlite_vec")
    db_path = tmp_path / "delvec_defer.db"
    backend = SQLiteBackend(db_path)
    if not backend._vec_available:
        backend.close()
        pytest.skip("sqlite-vec extension not available")
    try:
        emb = [1.0] * backend._dim
        entry = _vec_entry("M-delvec")
        backend.store(entry)
        backend.upsert_vector(entry.id, emb)
        assert backend.vector_exists("M-delvec") is True

        with pytest.raises(RuntimeError, match="mid-tx"):
            with backend.transaction():
                deleted = backend.delete_vector("M-delvec")
                assert deleted is True
                raise RuntimeError("mid-tx")

        # The vector delete was staged in the rolled-back transaction, so the
        # vector must still be present. An eager commit would have removed it.
        assert backend.vector_exists("M-delvec") is True, "delete_vector() committed prematurely inside transaction()"
    finally:
        backend.close()


def test_delete_vector_standalone_still_commits(tmp_path: Path) -> None:
    """v0.9.2 regression: outside a transaction, delete_vector() commits now."""
    pytest.importorskip("sqlite_vec")
    db_path = tmp_path / "delvec_standalone.db"
    backend = SQLiteBackend(db_path)
    if not backend._vec_available:
        backend.close()
        pytest.skip("sqlite-vec extension not available")
    try:
        emb = [1.0] * backend._dim
        entry = _vec_entry("M-delvec-now")
        backend.store(entry)
        backend.upsert_vector(entry.id, emb)
        assert backend.delete_vector("M-delvec-now") is True
        assert backend.vector_exists("M-delvec-now") is False
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


@pytest.mark.parametrize("operation", ["upsert", "delete", "delete_hype"])
def test_standalone_vector_writes_roll_back_on_commit_failure(tmp_path: Path, operation: str) -> None:
    """A failed standalone commit must not leak into a later transaction."""
    pytest.importorskip("sqlite_vec")
    backend = SQLiteBackend(tmp_path / f"{operation}_commit_failure.db")
    if not backend._vec_available:
        backend.close()
        pytest.skip("sqlite-vec extension not available")

    entry_id = "P#hype0" if operation == "delete_hype" else "M-vector"
    embedding = [1.0] * backend._dim
    initially_present = operation != "upsert"
    try:
        if initially_present:
            backend.upsert_vector(entry_id, embedding)

        real_conn = backend._conn
        proxy = _CommitFailingConn(real_conn)
        backend._conn = proxy  # type: ignore[assignment]
        try:
            with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
                if operation == "upsert":
                    backend.upsert_vector(entry_id, embedding)
                elif operation == "delete":
                    backend.delete_vector(entry_id)
                else:
                    backend.delete_hype_siblings("P")
            assert proxy.rolled_back
        finally:
            backend._conn = real_conn

        assert backend.vector_exists(entry_id) is initially_present
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


# ---------------------------------------------------------------------------
# S8 — delete_by_namespace: entries + wiki_refs + vectors atomic (both-or-neither)
# ---------------------------------------------------------------------------


def _wiki_entry_with_ref(entry_id: str, *, namespace: str, slug: str) -> MemoryEntry:
    """Build an entry carrying a single outbound wiki ref so a row lands in wiki_refs.

    ``slug`` must satisfy WikiPage validation (lowercase alphanumeric, single
    hyphen separators) — independent of the entry id, which has no such rule.
    """
    from trw_memory.wiki.models import WikiPage, WikiReference

    page = WikiPage(
        kind="topic",
        slug=f"topic/{slug}",
        title=entry_id,
        outbound_refs=[WikiReference(target_slug="topic/target", ref_type="related")],
    )
    return MemoryEntry(
        id=entry_id,
        content=page.title,
        namespace=namespace,
        metadata=page.to_memory_metadata(),
    )


def test_delete_by_namespace_removes_entries_and_wiki_refs_atomically(tmp_path: Path) -> None:
    """S8: a successful namespace delete clears entries AND companion wiki_refs.

    Observed via a second connection that reads committed-only state.
    """
    import sqlite3

    db_path = tmp_path / "s8ok.db"
    backend = SQLiteBackend(db_path)
    try:
        backend.store(_wiki_entry_with_ref("M-ns1", namespace="doomed", slug="ns-one"))
        backend.store(_wiki_entry_with_ref("M-ns2", namespace="doomed", slug="ns-two"))
        backend.store(_wiki_entry_with_ref("M-keep", namespace="other", slug="ns-keep"))

        observer = sqlite3.connect(str(db_path))
        try:
            # Precondition: 2 doomed entries + 2 doomed wiki_refs are committed.
            assert observer.execute("SELECT COUNT(*) FROM memories WHERE namespace = ?", ("doomed",)).fetchone()[0] == 2
            assert (
                observer.execute("SELECT COUNT(*) FROM wiki_refs WHERE namespace = ?", ("doomed",)).fetchone()[0] == 2
            )

            deleted = backend.delete_by_namespace("doomed")
            assert deleted == 2

            # Both entries and their wiki_refs are gone; the other namespace
            # is untouched — no orphan refs survive.
            assert observer.execute("SELECT COUNT(*) FROM memories WHERE namespace = ?", ("doomed",)).fetchone()[0] == 0
            assert (
                observer.execute("SELECT COUNT(*) FROM wiki_refs WHERE namespace = ?", ("doomed",)).fetchone()[0] == 0
            )
            assert observer.execute("SELECT COUNT(*) FROM wiki_refs WHERE namespace = ?", ("other",)).fetchone()[0] == 1
        finally:
            observer.close()
    finally:
        backend.close()


def test_delete_by_namespace_rollback_leaves_entries_and_wiki_refs_intact(tmp_path: Path) -> None:
    """S8: a crash AFTER the entry DELETE but BEFORE wiki_refs cleanup rolls BOTH back.

    Without the single-transaction wrapper the memories DELETE had already
    committed on its own, so a later crash would leave orphan wiki_refs (and the
    entries gone). We force a failure on the wiki_refs DELETE and prove — via a
    fresh observer connection reading committed-only state — that the entry rows
    AND the wiki_refs are both still present (neither side of the delete landed).
    """
    import sqlite3

    db_path = tmp_path / "s8rollback.db"
    backend = SQLiteBackend(db_path)
    try:
        backend.store(_wiki_entry_with_ref("M-rb1", namespace="doomed", slug="rb-one"))
        backend.store(_wiki_entry_with_ref("M-rb2", namespace="doomed", slug="rb-two"))

        # The C-extension Connection's ``execute`` attribute is read-only, so we
        # wrap the live connection in a thin delegating proxy that raises on the
        # companion wiki_refs DELETE (after the memories DELETE has been staged
        # inside the open transaction) and forwards everything else verbatim.
        real_conn = backend._conn

        class _FailingWikiCleanupConn:
            def execute(self, sql: str, *args: object) -> object:
                if "DELETE FROM wiki_refs" in sql:
                    raise RuntimeError("crash-before-wiki-cleanup")
                return real_conn.execute(sql, *args)

            def __getattr__(self, name: str) -> object:
                return getattr(real_conn, name)

        backend._conn = _FailingWikiCleanupConn()
        try:
            with pytest.raises(RuntimeError, match="crash-before-wiki-cleanup"):
                backend.delete_by_namespace("doomed")
        finally:
            backend._conn = real_conn

        observer = sqlite3.connect(str(db_path))
        try:
            # Entries survived — the staged memories DELETE was rolled back.
            assert (
                observer.execute("SELECT COUNT(*) FROM memories WHERE namespace = ?", ("doomed",)).fetchone()[0] == 2
            ), "entries were deleted despite the rolled-back transaction"
            # wiki_refs survived too — no orphan/partial state.
            assert (
                observer.execute("SELECT COUNT(*) FROM wiki_refs WHERE namespace = ?", ("doomed",)).fetchone()[0] == 2
            ), "wiki_refs were partially cleaned despite rollback"
        finally:
            observer.close()
        # Backend's own connection agrees the entries are still live.
        assert backend.get("M-rb1") is not None
        assert backend.get("M-rb2") is not None
    finally:
        backend.close()


def test_delete_by_namespace_empty_namespace_is_noop(tmp_path: Path) -> None:
    """S8: deleting an empty namespace returns 0 and opens no transaction."""
    db_path = tmp_path / "s8empty.db"
    backend = SQLiteBackend(db_path)
    try:
        assert backend.delete_by_namespace("never-existed") == 0
    finally:
        backend.close()


def test_connection_lock_serializes_concurrent_outer_transactions(tmp_path: Path) -> None:
    """S8 follow-on: concurrent ``transaction()`` callers serialize, never nest BEGIN.

    Before the ``_txn_serializer`` guard, two threads could both issue
    ``BEGIN IMMEDIATE`` on the single shared connection, and the second raised
    ``OperationalError: cannot start a transaction within a transaction``. The
    serializer makes outer transactions mutually exclusive; here 8 threads each
    open a transaction and store one row — all must succeed with zero errors.
    """
    db_path = tmp_path / "concurrent_txn.db"
    backend = SQLiteBackend(db_path)
    try:
        errors: list[str] = []
        errors_lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker(i: int) -> None:
            try:
                barrier.wait(timeout=5)
                with backend.transaction():
                    backend.store(make_entry(entry_id=f"M-conc-{i}"))
            except Exception as exc:  # record any thread failure for assertion
                with errors_lock:
                    errors.append(repr(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"concurrent transactions raced: {errors}"
        assert backend.count() == 8
    finally:
        backend.close()


def test_nested_transaction_under_connection_lock_does_not_deadlock(tmp_path: Path) -> None:
    """The connection lock is re-entrant: same-thread nested transactions never deadlock."""
    db_path = tmp_path / "nested_serialized.db"
    backend = SQLiteBackend(db_path)
    try:
        with backend.transaction():
            with backend.transaction():
                backend.store(make_entry(entry_id="M-nested-ok"))
        assert backend.get("M-nested-ok") is not None
    finally:
        backend.close()


def test_standalone_mutator_cannot_join_rolling_back_transaction(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "cross-thread-rollback.db")
    transaction_open = threading.Event()
    allow_rollback = threading.Event()
    outsider_done = threading.Event()
    outsider_results: list[int] = []
    thread_errors: list[BaseException] = []
    errors_lock = threading.Lock()
    try:
        backend.store(make_entry(entry_id="M-victim"))

        def owner() -> None:
            try:
                with backend.transaction():
                    backend.store(make_entry(entry_id="M-owner"))
                    transaction_open.set()
                    allow_rollback.wait(timeout=5)
                    raise RuntimeError("rollback owner")
            except RuntimeError:
                pass
            except BaseException as exc:
                with errors_lock:
                    thread_errors.append(exc)

        def outsider() -> None:
            transaction_open.wait(timeout=5)
            try:
                outsider_results.append(backend.increment_access_counts(["M-victim"]))
            except BaseException as exc:
                with errors_lock:
                    thread_errors.append(exc)
            finally:
                outsider_done.set()

        owner_thread = threading.Thread(target=owner)
        outsider_thread = threading.Thread(target=outsider)
        owner_thread.start()
        assert transaction_open.wait(timeout=5)
        outsider_thread.start()
        assert not outsider_done.wait(timeout=0.1), "standalone write joined another thread's transaction"
        allow_rollback.set()
        owner_thread.join(timeout=5)
        outsider_thread.join(timeout=5)

        assert not owner_thread.is_alive()
        assert not outsider_thread.is_alive()
        assert thread_errors == []
        assert outsider_results == [1]
        assert backend.get("M-owner") is None
        victim = backend.get("M-victim")
        assert victim is not None
        assert victim.access_count == 1
    finally:
        allow_rollback.set()
        backend.close()


def test_transaction_adopts_and_commits_existing_implicit_transaction(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "adopt-commit.db")
    try:
        backend.store(make_entry(entry_id="M-victim"))
        backend._conn.execute("UPDATE memories SET access_count = 5 WHERE id = ?", ("M-victim",))
        assert backend._conn.in_transaction

        with backend.transaction():
            backend.increment_access_counts(["M-victim"])

        victim = backend.get("M-victim")
        assert victim is not None
        assert victim.access_count == 6
    finally:
        backend.close()


def test_transaction_adopts_and_rolls_back_existing_implicit_transaction(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "adopt-rollback.db")
    try:
        backend.store(make_entry(entry_id="M-victim"))
        backend._conn.execute("UPDATE memories SET access_count = 5 WHERE id = ?", ("M-victim",))
        assert backend._conn.in_transaction

        with pytest.raises(RuntimeError, match="rollback"):
            with backend.transaction():
                backend.store(make_entry(entry_id="M-owner"))
                raise RuntimeError("rollback")

        victim = backend.get("M-victim")
        assert victim is not None
        assert victim.access_count == 0
        assert backend.get("M-owner") is None
    finally:
        backend.close()


def test_delete_by_namespace_removes_vectors_atomically(tmp_path: Path) -> None:
    """S8: vec_index rows for namespace entries are cleaned up in the same delete."""
    pytest.importorskip("sqlite_vec")
    db_path = tmp_path / "s8vec.db"
    backend = SQLiteBackend(db_path)
    if not backend._vec_available:
        backend.close()
        pytest.skip("sqlite-vec extension not available")
    try:
        emb = [1.0] * backend._dim
        backend.store(make_entry(entry_id="M-vec-ns", namespace="doomed"))
        backend.upsert_vector("M-vec-ns", emb)
        assert backend.vector_exists("M-vec-ns") is True

        backend.delete_by_namespace("doomed")

        assert backend.get("M-vec-ns") is None
        assert backend.vector_exists("M-vec-ns") is False
    finally:
        backend.close()
