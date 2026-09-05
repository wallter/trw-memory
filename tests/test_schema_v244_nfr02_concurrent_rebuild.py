"""PRD-CORE-244-NFR02 — the shared schema-5 rebuild under concurrent writers.

Acceptance criteria (PRD-CORE-244 verification.mappings NFR02):

1. Three processes opening the same database file concurrently, each running
   ``ensure_schema``, apply the rebuild exactly once, none raises, and
   ``PRAGMA user_version`` reads 5 in all three.
2. The schema-5 rebuild interrupted mid-run completes cleanly on retry: no
   duplicate columns, and (per the PRD) no double-demotion of a confidence
   value.

These tests pin what the shipped code actually does: ``ensure_schema``
(``trw-memory/src/trw_memory/storage/_schema.py``) reads ``PRAGMA
user_version`` twice — once as a cheap filter, then again INSIDE the
``BEGIN IMMEDIATE`` transaction, which is the read that decides. Only that
second read is authoritative, and AC1 is the test that says so: with the
in-transaction re-check reverted (the shipped behaviour until 2026-09-04) all
three openers observe ``current == 4`` before any of them commits, and all
three run the full migration storm — measured three real invocations of
``migrate_v5_namespace_boundary`` for three openers, and six for six, with
zero errors and a converged final state, because the rebuild is
idempotent-by-accident. ``call_count == 1`` below is what fails when it is
reverted.

Two independent signals attribute the fix, so a single weakened assertion
cannot make the test vacuous: the counted ``_MIGRATIONS[5]`` invocations, and
the count of pre-migration snapshot files, which is written by a different
module (``_schema_backup``) on a different code path and is also one-per-
migration only because the snapshot now runs inside the same write lock.

GAP (reported, not silently tested around): the AC's "without double-demoting
a confidence value" names a behavior — pre-existing ``verified`` rows failing
FR02's substantiation test SHALL be demoted to ``unverified`` with
``metadata.confidence_demoted_by`` set, as part of the schema-5 rebuild — that
does not exist in ``trw_memory/storage/_schema_v5.py``. Grepped: zero matches
for ``confidence_demoted_by`` or any confidence-column write anywhere in
``trw-memory/src`` or ``trw-mcp/src`` (2026-09-04). FR02's write-time gate
(``poisoning.py::reject_unsubstantiated_verified``) blocks new unsubstantiated
writes but never touches historical rows, so a store migrated to schema 5
today keeps every pre-existing unsubstantiated ``verified`` row exactly as
written. No test below exercises demotion idempotence because there is no
demotion pass to be idempotent.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

import trw_memory.storage._dbapi  # noqa: F401  — installs pysqlite3 as ``sqlite3``
import trw_memory.storage._schema as schema_module
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._schema import SchemaLockError, ensure_schema
from trw_memory.storage._schema_backup import BACKUP_DIR_NAME
from trw_memory.storage.sqlite_backend import SQLiteBackend

pytestmark = pytest.mark.unit


def _v4_store(path: Path, rows: int) -> None:
    backend = SQLiteBackend(path)
    try:
        for index in range(rows):
            backend.store(MemoryEntry(id=f"M-{index:04d}", content=f"row {index}", namespace="project:concurrent"))
    finally:
        backend.close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()


def _rebuild_counter(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Wrap ``_MIGRATIONS[5]`` so every real rebuild invocation is counted."""
    count = 0
    lock = threading.Lock()
    real_migrate = schema_module._MIGRATIONS[5]

    def _counting_migrate(cursor: sqlite3.Cursor) -> None:
        nonlocal count
        with lock:
            count += 1
        real_migrate(cursor)

    monkeypatch.setitem(schema_module._MIGRATIONS, 5, _counting_migrate)
    return lambda: count


@pytest.mark.parametrize("openers", [3, 6])
def test_concurrent_openers_apply_schema5_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, openers: int
) -> None:
    """AC1: N concurrent opens apply the rebuild once, none raises, all read v5.

    Parametrised past the AC's three openers to six, the size of a real stdio
    fleet booting after an upgrade, because the defect scaled with the opener
    count (3 openers -> 3 rebuilds, 6 -> 6) and a fix that only serialises
    three would pass the AC while still storming in production.
    """
    db = tmp_path / "concurrent.db"
    _v4_store(db, 50)
    rebuilds = _rebuild_counter(monkeypatch)

    results: list[int] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(openers)

    def _open_and_migrate() -> None:
        conn = sqlite3.connect(db, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            barrier.wait(timeout=10)
            ensure_schema(conn)
            results.append(int(conn.execute("PRAGMA user_version").fetchone()[0]))
        except BaseException as exc:  # pragma: no cover - failure path asserted below
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=_open_and_migrate) for _ in range(openers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == [], f"a concurrent opener raised: {errors!r}"
    assert results == [5] * openers, f"every opener must observe user_version==5, got {results!r}"
    assert rebuilds() == 1, f"the schema-5 rebuild must apply exactly once, ran {rebuilds()} times"

    # Independent attribution on a different module's code path: the
    # pre-migration snapshot is taken inside the same write lock, so the
    # migration that ran is also the only one that wrote a restore point.
    # Before the fix every racing opener wrote one, all to the same
    # second-stamped filename, i.e. over each other.
    snapshots = sorted((tmp_path / BACKUP_DIR_NAME).glob("concurrent.db.pre-schema-5.*"))
    assert len(snapshots) == 1, f"exactly one pre-migration snapshot must be written, got {snapshots!r}"

    final = sqlite3.connect(db)
    assert int(final.execute("SELECT COUNT(*) FROM memories").fetchone()[0]) == 50
    final.close()


def test_opener_that_waited_out_the_migration_skips_the_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cheap pre-read never decides alone: a stale "v4" observation is re-checked.

    Reproduces the losing side of the race deterministically instead of hoping
    the scheduler produces it. The opener samples ``user_version`` as 4, and
    only THEN does another connection complete the migration — exactly the
    window the defect lived in. The re-read under the write lock must see 5
    and skip, so no rebuild runs on this connection at all.
    """
    db = tmp_path / "stale-read.db"
    _v4_store(db, 20)

    conn = sqlite3.connect(db, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 4  # the stale sample

        migrator = sqlite3.connect(db, timeout=30)
        try:
            ensure_schema(migrator)  # another opener wins the race and commits
        finally:
            migrator.close()

        rebuilds = _rebuild_counter(monkeypatch)
        ensure_schema(conn)

        assert rebuilds() == 0, "an opener whose pre-read was invalidated must not rebuild"
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 5
        assert int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]) == 20
        assert not conn.in_transaction, "the skip path must not leave the write lock held"
    finally:
        conn.close()


def test_unobtainable_write_lock_refuses_rather_than_assuming_current(tmp_path: Path) -> None:
    """A failed IMMEDIATE acquisition surfaces; it never falls through to "assume current".

    While another connection holds the write lock the version is unknowable,
    and "unknowable" resolved to "already current" would hand back a
    v4-shaped store that the caller then reads as v5. The refusal is typed,
    names the store, and leaves ``user_version`` untouched.
    """
    db = tmp_path / "locked.db"
    _v4_store(db, 5)

    blocker = sqlite3.connect(db, timeout=30)
    blocker.execute("PRAGMA journal_mode=WAL")
    blocker.execute("BEGIN IMMEDIATE")
    blocker.execute("UPDATE memories SET content = 'held' WHERE id = 'M-0000'")

    conn = sqlite3.connect(db, timeout=0.2)
    try:
        with pytest.raises(SchemaLockError) as raised:
            ensure_schema(conn)
    finally:
        conn.close()
        blocker.rollback()
        blocker.close()

    message = str(raised.value)
    assert str(db) in message, "the refusal must name the store it could not lock"
    assert "refusing to open the store" in message

    observer = sqlite3.connect(db)
    try:
        assert int(observer.execute("PRAGMA user_version").fetchone()[0]) == 4
        assert int(observer.execute("SELECT COUNT(*) FROM memories").fetchone()[0]) == 5
    finally:
        observer.close()
    assert not (tmp_path / BACKUP_DIR_NAME).exists(), "a refused open must not snapshot"


def test_interrupted_rebuild_leaves_v4_and_a_clean_retry_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2: an interruption mid-rebuild rolls back to v4 intact; the retry is a clean single application.

    Simulates the interruption the PRD names by making the schema-5 delta
    raise partway through its own work. ``ensure_schema``'s single
    ``BEGIN IMMEDIATE`` transaction (``_schema.py:558-575``) must roll the
    WHOLE storm back on that exception, per its own documented contract —
    proved here against a real connection, not asserted from the docstring.
    """
    db = tmp_path / "interrupted.db"
    _v4_store(db, 30)

    real_migrate = schema_module._MIGRATIONS[5]

    def _raising_migrate(cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT COUNT(*) FROM memories")  # do some real work first
        raise RuntimeError("simulated interruption mid-rebuild")

    monkeypatch.setitem(schema_module._MIGRATIONS, 5, _raising_migrate)

    conn = sqlite3.connect(db)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        ensure_schema(conn)

    # Rolled back: version unchanged, row count intact, no leftover rebuild-in-
    # progress temp table from the half-applied rename dance.
    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 4
    assert int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]) == 30
    leftover = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_v5_rebuild'").fetchall()
    assert leftover == [], f"a rolled-back interruption must not leave rebuild-in-progress tables: {leftover!r}"
    conn.close()

    # Clean retry with the REAL delta restored: completes, stamps v5, preserves
    # every row, and leaves no duplicate column in `memories` (AC2's explicit
    # "without duplicate columns").
    monkeypatch.setitem(schema_module._MIGRATIONS, 5, real_migrate)
    conn = sqlite3.connect(db)
    ensure_schema(conn)

    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 5
    assert int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]) == 30
    columns = [str(row[1]) for row in conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert len(columns) == len(set(columns)), f"duplicate column after retry: {columns!r}"
    conn.close()
