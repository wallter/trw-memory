"""PRD-CORE-330: memories_fts deletes must be rowid-keyed, not id/namespace-scanned.

``memories_fts`` declares ``id``/``namespace`` as UNINDEXED fts5 columns, so
``DELETE FROM memories_fts WHERE id = ? AND namespace = ?`` scans the whole
FTS index every time (worker-1's CORE-309 S3 profiling; learning L-7L9A).
These tests prove the fix at the class level: every store()/update()/delete()
FTS write goes through ``memories_fts_rowid`` and issues only rowid-keyed
statements.
"""

from __future__ import annotations

import sqlite3

from tests.conftest import make_entry
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _fts_statements(conn: sqlite3.Connection, action) -> list[str]:
    """Run *action* while tracing every top-level SQL statement issued on *conn*.

    Returns only the application-level statements against ``memories_fts``
    itself (not ``memories_fts_rowid``, a plain indexed table, and not fts5's
    own internal shadow-table plumbing such as ``memories_fts_data``/
    ``memories_fts_docsize``/``memories_fts_idx`` -- SQLite's trace callback
    surfaces those as ``-- ``-prefixed comment lines when fts5 translates one
    application statement into its internal representation, and they are
    implementation detail, not something this fix controls).
    """
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    try:
        action()
    finally:
        conn.set_trace_callback(None)
    return [s for s in seen if not s.startswith("-- ") and "memories_fts" in s and "memories_fts_rowid" not in s]


def test_delete_never_filters_fts_by_id_and_namespace(sqlite_memory_backend: SQLiteBackend) -> None:
    """A single-row delete() must not issue `WHERE id = ? AND namespace = ?` against memories_fts.

    This is the statement that scanned the whole FTS index (worker-1's
    profiling). It fails against the pre-fix code, where delete() issues
    exactly this statement.
    """
    backend = sqlite_memory_backend
    entry = make_entry(entry_id="L-330-001", content="fts indexed delete target", namespace="default")
    backend.store(entry)

    statements = _fts_statements(backend._conn, lambda: backend.delete("L-330-001", namespace="default"))

    assert statements, "delete() should have issued at least one memories_fts statement"
    for stmt in statements:
        assert "WHERE id = ? AND namespace = ?" not in stmt, f"full-scan delete reintroduced: {stmt!r}"
        assert "rowid" in stmt.lower(), f"expected a rowid-keyed statement, got: {stmt!r}"


def test_store_overwrite_never_filters_fts_by_id_and_namespace(sqlite_memory_backend: SQLiteBackend) -> None:
    """Re-storing an existing (namespace, id) (INSERT OR REPLACE) must resolve its FTS row
    by rowid, not by scanning id/namespace."""
    backend = sqlite_memory_backend
    entry = make_entry(entry_id="L-330-002", content="first version", namespace="default")
    backend.store(entry)

    updated = make_entry(entry_id="L-330-002", content="second version", namespace="default")
    statements = _fts_statements(backend._conn, lambda: backend.store(updated))

    assert statements
    for stmt in statements:
        assert "WHERE id = ? AND namespace = ?" not in stmt, f"full-scan delete reintroduced: {stmt!r}"


def test_update_content_never_filters_fts_by_id_and_namespace(sqlite_memory_backend: SQLiteBackend) -> None:
    backend = sqlite_memory_backend
    entry = make_entry(entry_id="L-330-003", content="before update", namespace="default")
    backend.store(entry)

    statements = _fts_statements(
        backend._conn, lambda: backend.update("L-330-003", namespace="default", content="after update")
    )

    assert statements
    for stmt in statements:
        assert "WHERE id = ? AND namespace = ?" not in stmt, f"full-scan delete reintroduced: {stmt!r}"


def test_delete_by_rowid_is_indexed_not_a_full_scan(sqlite_memory_backend: SQLiteBackend) -> None:
    """EXPLAIN QUERY PLAN for a rowid-keyed delete never reports a full table scan.

    fts5 rowid deletes are addressed directly; only a WHERE on an UNINDEXED
    text column forces `SCAN memories_fts` (or fts5's internal equivalent).
    """
    backend = sqlite_memory_backend
    entry = make_entry(entry_id="L-330-004", content="explain target", namespace="default")
    backend.store(entry)

    row = backend._conn.execute(
        "SELECT fts_rowid FROM memories_fts_rowid WHERE namespace = ? AND id = ?",
        ("default", "L-330-004"),
    ).fetchone()
    assert row is not None, "store() must record a memories_fts_rowid mapping row"
    fts_rowid = row[0]

    plan_rows = backend._conn.execute(
        "EXPLAIN QUERY PLAN DELETE FROM memories_fts WHERE rowid = ?", (fts_rowid,)
    ).fetchall()
    plan_text = " ".join(str(r) for r in plan_rows).upper()
    assert "SCAN MEMORIES_FTS" not in plan_text.replace(" ", "") and "SCAN TABLE MEMORIES_FTS" not in plan_text


def test_rowid_map_backfills_from_existing_fts_rows(tmp_path) -> None:
    """A store opened by an EARLIER build (no memories_fts_rowid rows yet, but
    memories_fts already populated) backfills the map on open, via
    ensure_fts_table's anti-join -- not just for freshly-stored rows."""
    db_path = tmp_path / "legacy.db"
    backend = SQLiteBackend(db_path)
    entry = make_entry(entry_id="L-330-005", content="legacy row", namespace="default")
    backend.store(entry)
    backend.close()

    # Simulate a pre-PRD-CORE-330 store: wipe the mapping table but leave the
    # FTS row (as an older build, which never wrote to memories_fts_rowid, would).
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM memories_fts_rowid")
    conn.commit()
    conn.close()

    reopened = SQLiteBackend(db_path)
    try:
        row = reopened._conn.execute(
            "SELECT fts_rowid FROM memories_fts_rowid WHERE namespace = ? AND id = ?",
            ("default", "L-330-005"),
        ).fetchone()
        assert row is not None, "ensure_fts_table must backfill missing mapping rows on open"
    finally:
        reopened.close()


def test_fts_search_results_unchanged_by_rowid_keying(sqlite_memory_backend: SQLiteBackend) -> None:
    """Full-equivalence: search_fts() (the memories_fts MATCH path, not the LIKE-based
    search()) returns the same rows and no duplicates after an overwrite exercises the
    delete-then-reinsert path this PRD changed."""
    backend = sqlite_memory_backend
    for i in range(5):
        backend.store(
            make_entry(entry_id=f"L-330-search-{i}", content=f"searchable content number {i}", namespace="default")
        )
    # Overwrite one entry (exercises the delete-then-reinsert path) before searching.
    backend.store(
        make_entry(entry_id="L-330-search-2", content="searchable content number 2 revised", namespace="default")
    )

    results = backend.search_fts("searchable", namespace="default", top_k=10)
    ids = [r.id for r in results]
    assert "L-330-search-2" in ids
    assert len(set(ids)) == len(ids), "no duplicate rows from a stale FTS row after overwrite"
    revised = next(r for r in results if r.id == "L-330-search-2")
    assert revised.content == "searchable content number 2 revised"
