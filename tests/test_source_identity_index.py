"""PRD-CORE-331 FR06 / B71-102: ``idx_memories_namespace_source_identity`` (schema 11).

``SQLiteBackend.ids_by_source`` (``storage/_sqlite_backend_mixins.py``) runs
``SELECT id FROM memories WHERE namespace = ? AND source_identity = ? LIMIT ?``
with no covering index over ``(namespace, source_identity)``, so it scans every
row in the namespace. Schema 11 adds a plain composite index over that pair.

These tests are BEHAVIOURAL, against the schema-10 shape int 613d34db4 ships
(the base this lane branched from): ``_as_schema_10`` drops the new index and
rewinds ``user_version`` to 10, reproducing exactly the pre-FR06 shape (the
index is the *only* delta between schema 10 and 11, so this is equivalent to
a genuine int 613d34db4 archive without vendoring one).

Measured (not assumed): on the schema-10 shape, ``EXPLAIN QUERY PLAN`` of the
exact ``ids_by_source`` SQL does NOT show a bare ``SCAN memories`` -- the
planner instead reaches for the composite ``idx_memories_ns_status_imp``
(namespace, status, importance) and uses only its leftmost ``namespace=?``
prefix, e.g. ``SEARCH memories USING INDEX idx_memories_ns_status_imp
(namespace=?)``. That index does not cover ``source_identity``, so every row
in the namespace is still visited and filtered by residual predicate --
functionally the full-namespace scan the remaining-work map called a "SCAN",
just not literally that opcode once another (namespace-prefixed) composite
index happens to exist. After the migration, the plan names the new,
purpose-built index and needs no residual filter.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage import _schema
from trw_memory.storage._probe import StoreState, probe_store
from trw_memory.storage._untrusted_store import verify_untrusted_store
from trw_memory.storage.sqlite_backend import SQLiteBackend

_IDS_BY_SOURCE_SQL = "SELECT id FROM memories WHERE namespace = ? AND source_identity = ? LIMIT ?"


def _explain(conn: sqlite3.Connection) -> str:
    rows = conn.execute(f"EXPLAIN QUERY PLAN {_IDS_BY_SOURCE_SQL}", ("default", "agent-x", 10)).fetchall()
    return " | ".join(str(row[-1]) for row in rows)


def _as_schema_10(path: Path) -> None:
    """Rewind a fresh (schema-11) store to the schema-10 shape: no source-identity
    index, stamped 10 -- exactly what int 613d34db4 (this lane's base) produces."""
    with sqlite3.connect(path) as conn:
        conn.execute("DROP INDEX idx_memories_namespace_source_identity")
        conn.execute("PRAGMA user_version = 10")


def _populate(backend: SQLiteBackend, count: int = 50) -> None:
    for i in range(count):
        backend.store(
            MemoryEntry(
                id=f"M-{i:04d}",
                content=f"row {i}",
                namespace="default",
                source_identity="agent-x" if i % 5 == 0 else "agent-y",
            )
        )


def test_the_pre_fr06_schema_10_shape_does_not_use_a_source_identity_index(tmp_path: Path) -> None:
    """The base state this migration fixes: no index covers ``source_identity``, so the plan either
    names a bare table SCAN or reaches for another composite index's ``namespace=?`` prefix only
    (measured: ``idx_memories_ns_status_imp``) -- either way every row in the namespace is still
    visited and filtered by residual predicate, never resolved by an index alone."""
    path = tmp_path / "v10.db"
    store = SQLiteBackend(path)
    _populate(store)
    store.close()
    _as_schema_10(path)

    conn = sqlite3.connect(path)
    try:
        plan = _explain(conn)
        assert "idx_memories_namespace_source_identity" not in plan, plan
        assert "SCAN memories" in plan or "namespace=?" in plan, plan
    finally:
        conn.close()


def test_schema_11_uses_the_index_not_a_scan(tmp_path: Path) -> None:
    """After the migration, the identical SQL resolves through the new index."""
    path = tmp_path / "v11.db"
    store = SQLiteBackend(path)
    _populate(store)

    plan = _explain(store._conn)
    assert "SCAN memories" not in plan, plan
    assert "idx_memories_namespace_source_identity" in plan, plan
    store.close()


def test_ids_by_source_answers_match_whichever_shape_it_runs_on(tmp_path: Path) -> None:
    """The index changes the plan, never the answer: same rows, indexed or scanned.

    ``SQLiteBackend.__init__`` calls ``ensure_schema`` on open, which (idempotently) re-runs the
    migration storm -- opening the v10-shaped file through a backend would recreate the index
    BEFORE the "scanned" query ever ran, so the two sides would silently query the same (indexed)
    schema and this would test nothing. The scanned read therefore goes through a bare ``sqlite3``
    connection, which never migrates anything; only after that read is a backend opened (which
    migrates the file in place, schema 10 -> SCHEMA_VERSION) to take the indexed reading.
    """
    path = tmp_path / "answers.db"
    store = SQLiteBackend(path)
    _populate(store)
    store.close()
    _as_schema_10(path)

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 10
        plan = _explain(raw)
        assert "idx_memories_namespace_source_identity" not in plan, plan
        scanned = sorted(str(row[0]) for row in raw.execute(_IDS_BY_SOURCE_SQL, ("default", "agent-x", 100)))
    finally:
        raw.close()

    assert scanned and len(scanned) == 10  # every 5th of 50 rows

    migrated = SQLiteBackend(path)  # opening migrates the file in place: schema 10 -> SCHEMA_VERSION
    try:
        # Live constant, not a literal: later migrations must not break this test.
        assert migrated._conn.execute("PRAGMA user_version").fetchone()[0] == _schema.SCHEMA_VERSION
        indexed_plan = _explain(migrated._conn)
        assert "idx_memories_namespace_source_identity" in indexed_plan, indexed_plan
        indexed = sorted(migrated.ids_by_source("default", "agent-x", 100))
        assert indexed == scanned
    finally:
        migrated.close()


def test_a_v10_store_opens_and_migrates_to_11_keeping_every_row(tmp_path: Path) -> None:
    path = tmp_path / "migrate.db"
    store = SQLiteBackend(path)
    _populate(store, count=30)
    before_ids = sorted(row[0] for row in store._conn.execute("SELECT id FROM memories ORDER BY id"))
    store.close()
    _as_schema_10(path)
    assert sqlite3.connect(path).execute("PRAGMA user_version").fetchone()[0] == 10

    reopened = SQLiteBackend(path)
    try:
        conn = reopened._conn
        assert conn.execute("PRAGMA user_version").fetchone()[0] == _schema.SCHEMA_VERSION
        after_ids = sorted(row[0] for row in conn.execute("SELECT id FROM memories ORDER BY id"))
        assert after_ids == before_ids
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert "idx_memories_namespace_source_identity" in names
        # a rerun of the delta alone changes nothing further (idempotent)
        _schema._migrate_v11_source_identity_index(conn.cursor())
        assert sorted(row[0] for row in conn.execute("SELECT id FROM memories ORDER BY id")) == after_ids
    finally:
        reopened.close()


def test_a_fresh_v0_store_gets_the_index_via_bootstrap_not_only_the_migration(tmp_path: Path) -> None:
    """A brand-new database goes through ``_bootstrap_and_backfill`` (MEMORIES_INDEXES), never the
    v11 delta directly -- confirm the index is built either way."""
    path = tmp_path / "fresh.db"
    store = SQLiteBackend(path)
    try:
        names = {row[0] for row in store._conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert "idx_memories_namespace_source_identity" in names
    finally:
        store.close()


def test_the_untrusted_store_allowlist_accepts_an_imported_v11_store(tmp_path: Path) -> None:
    """A plain column index carries no expression and no WHERE clause, so
    ``_untrusted_store._REFUSED`` (which refuses only expression/partial indexes)
    admits a v11 store unchanged."""
    path = tmp_path / "checkout.db"
    store = SQLiteBackend(path)
    _populate(store, count=5)
    store.close()

    verify_untrusted_store(path)  # raises StorageError on refusal; no exception is the assertion


def test_qual_147_probe_store_reads_a_v11_store_as_ready(tmp_path: Path) -> None:
    """QUAL-147's ``probe_store`` needs nothing for schema 11: the index is not a column, so
    ``_schema.MIGRATE_COLS``' legacy-default backfill logic is untouched."""
    path = tmp_path / "probe.db"
    store = SQLiteBackend(path)
    _populate(store, count=3)
    store.close()

    result = probe_store(path)
    assert result.state == StoreState.READY
    assert result.real_rows == 3
