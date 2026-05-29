"""Tests for read-time row quarantining (P2 — auto-recovery layer).

Strategy: insert good rows via the backend, then directly inject a bad-UTF-8
row using a raw sqlite3 connection with text_factory=bytes. This simulates the
exact incident scenario (a corrupt inode yielding undecoded bytes).

All tests use in-memory SQLite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import structlog.testing

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage._resilient_fetch import (
    FetchQuery,
    fetch_rows_resilient,
    fetch_rows_via_bytes_fallback,
    get_bytes_fallback_failures,
    is_utf8_decode_error,
    reset_bytes_fallback_failures,
)
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    entry_id: str,
    detail: str = "clean detail",
    *,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    namespace: str = "default",
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content="content",
        detail=detail,
        status=status,
        namespace=namespace,
        source="agent",  # type: ignore[arg-type]
    )


def _inject_bad_utf8_row(
    db_path: Path | str,
    entry_id: str,
    *,
    status: str = "active",
    namespace: str = "default",
    updated_at: str = "2024-01-01T00:00:00+00:00",
) -> None:
    """Directly insert a row with invalid UTF-8 bytes in the ``detail`` column.

    SQLite stores the value verbatim, so we use the ``X'...'`` hex blob syntax
    to write bytes that Python's str codec cannot represent. ``status`` /
    ``namespace`` / ``updated_at`` are parameterised so tests can prove the
    degraded fallback honours WHERE filters and ORDER BY.
    """
    # b"bad\x80\x81bytes" — bare UTF-8 continuation bytes with no lead byte.
    bad_hex_str = b"bad\x80\x81bytes".hex()

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT OR REPLACE INTO memories (
            id, content, detail, tags, evidence, importance, status,
            recurrence, namespace, created_at, updated_at, access_count,
            session_count, q_value, q_observations, source,
            source_identity, client_profile, model_id,
            merged_from, consolidated_from, outcome_history,
            assertions, anchors, anchor_validity,
            type, nudge_line, expires_at, confidence,
            task_type, domain, phase_origin, phase_affinity,
            team_origin, protection_tier, sessions_surfaced,
            outcome_correlation, sync_hash, sync_seq,
            recall_count, helpful_count, unhelpful_count,
            vector_clock, metadata,
            published_to_platform, pending_delete, cross_validated
        ) VALUES (
            ?, ?, CAST(X'"""
        + bad_hex_str
        + """' AS TEXT),
            '[]', '[]', 0.5, ?,
            1, ?, '2024-01-01T00:00:00+00:00', ?, 0,
            0, 0.5, 0, 'agent',
            '', '', '',
            '[]', '[]', '[]',
            '[]', '[]', 1.0,
            'fact', '', '', 'medium',
            '', '[]', '', '[]',
            '', 'standard', 0,
            '', '', 0,
            0, 0, 0,
            '{}', '{}',
            0, 0, 0
        )
        """,
        (entry_id, "content", status, namespace, updated_at),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Test: bad row quarantined, good rows returned
# ---------------------------------------------------------------------------


def test_list_entries_skips_bad_utf8_row(tmp_path: Path) -> None:
    """Bad-UTF-8 rows are skipped; good rows are returned; quarantine counter increments."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)

    # Insert a good entry via the backend
    good = _make_entry("M-good-001", "perfectly valid detail")
    backend.store(good)

    backend.close()

    # Inject a bad row directly
    _inject_bad_utf8_row(db_path, "M-bad-001")

    # Reopen and read
    backend2 = SQLiteBackend(db_path)
    with structlog.testing.capture_logs() as logs:
        results = backend2.list_entries(limit=100)

    ids = [e.id for e in results]
    assert "M-good-001" in ids, "Good row must be returned"
    assert "M-bad-001" not in ids, "Bad row must be quarantined"
    assert backend2.quarantine_count_utf8 >= 1

    quarantine_events = [log for log in logs if log.get("action") == "memory_row_utf8_quarantined"]
    assert len(quarantine_events) >= 1, "Expected at least one quarantine log event"
    backend2.close()


# ---------------------------------------------------------------------------
# Test: all bad rows → empty list, no raise
# ---------------------------------------------------------------------------


def test_list_entries_returns_empty_list_when_all_bad(tmp_path: Path) -> None:
    """When every row is corrupted, list_entries returns [] without raising."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.close()

    # Inject 2 bad rows
    _inject_bad_utf8_row(db_path, "M-bad-001")
    _inject_bad_utf8_row(db_path, "M-bad-002")

    backend2 = SQLiteBackend(db_path)
    results = backend2.list_entries(limit=100)
    assert results == []
    assert backend2.quarantine_count_utf8 >= 2
    backend2.close()


# ---------------------------------------------------------------------------
# Test: clean DB leaves quarantine counter at 0
# ---------------------------------------------------------------------------


def test_list_entries_on_clean_db_no_quarantine() -> None:
    """Happy path: clean DB, quarantine counter stays at 0."""
    backend = SQLiteBackend(Path(":memory:"))
    entry = _make_entry("M-clean-001", "clean entry")
    backend.store(entry)

    results = backend.list_entries(limit=100)
    assert any(e.id == "M-clean-001" for e in results)
    assert backend.quarantine_count_utf8 == 0
    backend.close()


# ---------------------------------------------------------------------------
# is_utf8_decode_error — driver-message classifier
# ---------------------------------------------------------------------------


def test_is_utf8_decode_error_recognises_unicode_decode_error() -> None:
    """A raw UnicodeDecodeError is always classified as a UTF-8 failure."""
    exc = UnicodeDecodeError("utf-8", b"\x80", 0, 1, "invalid start byte")
    assert is_utf8_decode_error(exc) is True


def test_is_utf8_decode_error_recognises_driver_message() -> None:
    """The SQLite >= 3.51 'Could not decode to UTF-8' message is recognised."""
    exc = sqlite3.OperationalError("Could not decode to UTF-8 column 'detail'")
    assert is_utf8_decode_error(exc) is True


def test_is_utf8_decode_error_ignores_unrelated_operational_error() -> None:
    """A non-decode OperationalError must not be misclassified."""
    exc = sqlite3.OperationalError("database is locked")
    assert is_utf8_decode_error(exc) is False


# ---------------------------------------------------------------------------
# FetchQuery.build — filter / order / limit reconstruction
# ---------------------------------------------------------------------------


def test_fetch_query_build_includes_where_order_and_limit() -> None:
    """build() reconstructs the WHERE clause, ORDER BY, and appends LIMIT."""
    query = FetchQuery(
        select_columns_sql="id, content",
        where_sql="status = ?",
        params=("active",),
        order_by="updated_at DESC",
        limit=10,
    )
    sql, params = query.build()
    assert "WHERE status = ?" in sql
    assert "ORDER BY updated_at DESC" in sql
    assert sql.rstrip().endswith("LIMIT ?")
    assert params == ("active", 10)


def test_fetch_query_build_omits_limit_when_none() -> None:
    """When limit is None (e.g. entries_with_assertions), no LIMIT is appended."""
    query = FetchQuery(select_columns_sql="id", where_sql="1", params=(), limit=None)
    sql, params = query.build()
    assert "LIMIT" not in sql
    assert params == ()


# ---------------------------------------------------------------------------
# Filter / limit preservation in the degraded (bytes-mode) path — the core fix
# ---------------------------------------------------------------------------


def test_list_entries_fallback_preserves_status_filter(tmp_path: Path) -> None:
    """A status filter must survive the degraded path — no wrong-status rows leak."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.store(_make_entry("M-active-good", "ok", status=MemoryStatus.ACTIVE))
    backend.store(_make_entry("M-archived-good", "ok", status=MemoryStatus.ARCHIVED))
    backend.close()

    # Bad rows in BOTH statuses so the fallback re-execute is forced regardless.
    _inject_bad_utf8_row(db_path, "M-active-bad", status="active")
    _inject_bad_utf8_row(db_path, "M-archived-bad", status="archived")

    backend2 = SQLiteBackend(db_path)
    results = backend2.list_entries(status=MemoryStatus.ACTIVE, limit=100)
    ids = {e.id for e in results}

    assert ids == {"M-active-good"}, "Degraded path must honour status=active filter"
    assert all(e.status == MemoryStatus.ACTIVE for e in results)
    backend2.close()


def test_list_entries_fallback_preserves_namespace_filter(tmp_path: Path) -> None:
    """A namespace filter must survive the degraded path."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.store(_make_entry("M-ns-a-good", "ok", namespace="alpha"))
    backend.store(_make_entry("M-ns-b-good", "ok", namespace="beta"))
    backend.close()

    _inject_bad_utf8_row(db_path, "M-ns-a-bad", namespace="alpha")
    _inject_bad_utf8_row(db_path, "M-ns-b-bad", namespace="beta")

    backend2 = SQLiteBackend(db_path)
    results = backend2.list_entries(namespace="alpha", limit=100)
    ids = {e.id for e in results}

    assert ids == {"M-ns-a-good"}, "Degraded path must honour namespace filter"
    assert all(e.namespace == "alpha" for e in results)
    backend2.close()


def test_list_entries_fallback_preserves_limit(tmp_path: Path) -> None:
    """The LIMIT must survive the degraded path — fallback must not return all rows."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    for i in range(5):
        backend.store(_make_entry(f"M-good-{i}", "ok"))
    backend.close()

    _inject_bad_utf8_row(db_path, "M-bad-001")

    backend2 = SQLiteBackend(db_path)
    results = backend2.list_entries(limit=3)
    # 6 rows exist (5 good + 1 bad); limit=3 caps the re-executed query, then the
    # bad row (if within the window) is quarantined — never more than 3 returned.
    assert len(results) <= 3, "Degraded path must honour LIMIT"
    backend2.close()


# ---------------------------------------------------------------------------
# Execute-time decode path (SQLite >= 3.51) — simulated via a stub connection
# ---------------------------------------------------------------------------


class _ExecuteRaisesConn:
    """Connection stub whose execute() raises a UTF-8 decode error (SQLite>=3.51)."""

    def __init__(self, real: sqlite3.Connection) -> None:
        self._real = real

    def execute(self, sql: str, params: object = ()) -> sqlite3.Cursor:
        raise sqlite3.OperationalError("Could not decode to UTF-8 column 'detail'")

    def cursor(self) -> sqlite3.Cursor:
        return self._real.cursor()


def test_list_entries_routes_execute_time_decode_error_to_fallback(tmp_path: Path) -> None:
    """An execute-time UTF-8 error routes to the bytes-mode fallback (the added fix)."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.store(_make_entry("M-good-001", "valid"))
    backend.close()

    _inject_bad_utf8_row(db_path, "M-bad-001")

    backend2 = SQLiteBackend(db_path)
    # Force the SQLite >= 3.51 behaviour: execute() (not fetchall()) raises.
    backend2._conn = _ExecuteRaisesConn(backend2._conn)  # type: ignore[assignment]

    with structlog.testing.capture_logs() as logs:
        results = backend2.list_entries(limit=100)

    ids = {e.id for e in results}
    assert "M-good-001" in ids, "Good row recovered via the bytes-mode fallback"
    assert "M-bad-001" not in ids, "Bad row quarantined on the execute-time path"
    assert backend2.quarantine_count_utf8 >= 1
    quarantine = [log for log in logs if log.get("action") == "memory_row_utf8_quarantined"]
    assert quarantine, "Expected a quarantine log on the execute-time path"


def test_list_entries_propagates_non_decode_operational_error(tmp_path: Path) -> None:
    """A non-decode OperationalError at execute() must NOT be swallowed by the fix."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.store(_make_entry("M-good-001", "valid"))

    class _LockedConn:
        def execute(self, sql: str, params: object = ()) -> sqlite3.Cursor:
            raise sqlite3.OperationalError("database is locked")

        def cursor(self) -> sqlite3.Cursor:  # pragma: no cover - not reached
            raise AssertionError("fallback must not run for non-decode errors")

    backend._conn = _LockedConn()  # type: ignore[assignment]
    with pytest.raises(StorageError):
        backend.list_entries(limit=100)


# ---------------------------------------------------------------------------
# quarantine log carries an outcome field
# ---------------------------------------------------------------------------


def test_quarantine_log_has_outcome_field(tmp_path: Path) -> None:
    """Quarantine events carry outcome='quarantined' for observability."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.store(_make_entry("M-good-001", "valid"))
    backend.close()
    _inject_bad_utf8_row(db_path, "M-bad-001")

    backend2 = SQLiteBackend(db_path)
    with structlog.testing.capture_logs() as logs:
        backend2.list_entries(limit=100)

    quarantine = [log for log in logs if log.get("action") == "memory_row_utf8_quarantined"]
    assert quarantine
    assert all(log.get("outcome") == "quarantined" for log in quarantine)
    backend2.close()


# ---------------------------------------------------------------------------
# fallback secondary-connection failure → empty + outcome='fallback_failed'
# ---------------------------------------------------------------------------


def test_bytes_fallback_logs_fallback_failed_when_connect_fails(tmp_path: Path) -> None:
    """If the secondary bytes-mode connection fails, return [] and log the outcome."""

    class _FailingDBAPI:
        def connect(self, database: str) -> object:
            raise sqlite3.OperationalError("unable to open database file")

    query = FetchQuery(select_columns_sql="id, content", where_sql="1", limit=10)
    with structlog.testing.capture_logs() as logs:
        results, delta = fetch_rows_via_bytes_fallback(
            db_path=tmp_path / "memory.db",
            dbapi=_FailingDBAPI(),  # type: ignore[arg-type]
            query=query,
        )

    assert results == []
    assert delta == 0
    failed = [log for log in logs if log.get("outcome") == "fallback_failed"]
    assert failed, "Expected a db_utf8_fallback_failed log with outcome field"


def test_bytes_fallback_failure_increments_distinct_counter(tmp_path: Path) -> None:
    """F1: a hard fallback failure bumps the dedicated counter, not just a log.

    The per-row quarantine counter (``quarantine_count_utf8``) cannot reflect a
    failed *secondary connection* — those rows were never read, so ``delta`` is
    0. The distinct ``bytes_fallback_failures`` counter exists precisely so the
    silent drop is countable for monitoring.
    """

    class _FailingDBAPI:
        def connect(self, database: str) -> object:
            raise sqlite3.OperationalError("unable to open database file")

    reset_bytes_fallback_failures()
    assert get_bytes_fallback_failures() == 0

    query = FetchQuery(select_columns_sql="id, content", where_sql="1", limit=10)
    for _ in range(3):
        results, delta = fetch_rows_via_bytes_fallback(
            db_path=tmp_path / "memory.db",
            dbapi=_FailingDBAPI(),  # type: ignore[arg-type]
            query=query,
        )
        assert results == []
        assert delta == 0  # per-row quarantine delta stays 0 (fail-open preserved)

    # The distinct counter makes the otherwise-invisible drop countable.
    assert get_bytes_fallback_failures() == 3


def test_bytes_fallback_success_does_not_increment_failure_counter(tmp_path: Path) -> None:
    """A successful (non-failing) fallback must NOT bump the failure counter."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.store(_make_entry("M-good-001", "valid"))
    backend.close()
    _inject_bad_utf8_row(db_path, "M-bad-001")

    reset_bytes_fallback_failures()
    backend2 = SQLiteBackend(db_path)
    # Force the execute-time decode error so the (working) bytes fallback runs.
    backend2._conn = _ExecuteRaisesConn(backend2._conn)  # type: ignore[assignment]
    results = backend2.list_entries(limit=100)

    assert {e.id for e in results} == {"M-good-001"}
    assert get_bytes_fallback_failures() == 0, "Working fallback must not count as a failure"
    # Note: backend2._conn is the _ExecuteRaisesConn stub (no close()), matching
    # the sibling execute-time test which also leaves the stub un-closed.


# ---------------------------------------------------------------------------
# row_to_entry failure on a cleanly-decoded row is quarantined (not raised)
# ---------------------------------------------------------------------------


def _inject_malformed_status_row(db_path: Path | str, entry_id: str) -> None:
    """Insert a row whose columns decode but whose status is an invalid enum."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT OR REPLACE INTO memories (
            id, content, detail, tags, evidence, importance, status,
            recurrence, namespace, created_at, updated_at, access_count,
            session_count, q_value, q_observations, source,
            source_identity, client_profile, model_id,
            merged_from, consolidated_from, outcome_history,
            assertions, anchors, anchor_validity,
            type, nudge_line, expires_at, confidence,
            task_type, domain, phase_origin, phase_affinity,
            team_origin, protection_tier, sessions_surfaced,
            outcome_correlation, sync_hash, sync_seq,
            recall_count, helpful_count, unhelpful_count,
            vector_clock, metadata,
            published_to_platform, pending_delete, cross_validated
        ) VALUES (
            ?, ?, 'd', '[]', '[]', 0.5, 'NOT_A_VALID_STATUS',
            1, 'default', '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00', 0,
            0, 0.5, 0, 'agent', '', '', '',
            '[]', '[]', '[]', '[]', '[]', 1.0,
            'fact', '', '', 'medium', '', '[]', '', '[]',
            '', 'standard', 0, '', '', 0, 0, 0, 0,
            '{}', '{}', 0, 0, 0
        )
        """,
        (entry_id, "content"),
    )
    conn.commit()
    conn.close()


def test_bytes_fallback_quarantines_unmappable_row(tmp_path: Path) -> None:
    """A row that decodes but fails row_to_entry is quarantined via the fallback."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.store(_make_entry("M-good-001", "valid"))
    backend.close()

    # Bad-UTF-8 row forces the fallback; malformed-status row exercises the
    # row_to_entry quarantine branch within it.
    _inject_bad_utf8_row(db_path, "M-bad-utf8")
    _inject_malformed_status_row(db_path, "M-bad-enum")

    backend2 = SQLiteBackend(db_path)
    with structlog.testing.capture_logs() as logs:
        results = backend2.list_entries(limit=100)

    ids = {e.id for e in results}
    assert ids == {"M-good-001"}
    columns = {log.get("column") for log in logs if log.get("action") == "memory_row_utf8_quarantined"}
    assert "row_to_entry" in columns, "Unmappable row should log column='row_to_entry'"
    backend2.close()


# ---------------------------------------------------------------------------
# search() and entries_with_assertions() share the same resilience
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# fetch_rows_resilient — fetch-time path (older drivers raise during fetchall)
# ---------------------------------------------------------------------------


class _FetchallRaisesCursor:
    """Cursor stub whose fetchall() raises a UTF-8 decode error (old drivers)."""

    description = None

    def fetchall(self) -> list[tuple[object, ...]]:
        raise sqlite3.OperationalError("Could not decode to UTF-8 column 'detail'")


class _FetchallLockedCursor:
    """Cursor stub whose fetchall() raises a non-decode OperationalError."""

    description = None

    def fetchall(self) -> list[tuple[object, ...]]:
        raise sqlite3.OperationalError("database is locked")


def test_fetch_rows_resilient_routes_fetch_time_decode_to_fallback(tmp_path: Path) -> None:
    """A fetch-time UTF-8 error (older drivers) routes to the bytes-mode fallback."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.store(_make_entry("M-good-001", "valid"))
    backend.close()
    _inject_bad_utf8_row(db_path, "M-bad-001")

    query = FetchQuery(
        select_columns_sql="id, content, detail",
        where_sql="1",
        limit=100,
    )
    # The fallback only needs a real id/content/detail projection to decode;
    # we deliberately drive it via the fetch-time entry point with a stub cursor.
    results, delta = fetch_rows_resilient(
        _FetchallRaisesCursor(),
        db_path=db_path,
        dbapi=sqlite3,  # type: ignore[arg-type]
        query=query,
    )
    # row_to_entry needs the full column set, so the partial projection makes
    # every recovered row unmappable — but the path is exercised and the bad
    # UTF-8 row is quarantined. Assert the fallback ran (delta reflects rows).
    assert isinstance(results, list)
    assert delta >= 1


def test_fetch_rows_resilient_reraises_non_decode_error() -> None:
    """A non-decode fetchall() error must propagate, not route to the fallback."""
    query = FetchQuery(select_columns_sql="id", where_sql="1", limit=10)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        fetch_rows_resilient(
            _FetchallLockedCursor(),
            db_path=Path("/nonexistent.db"),
            dbapi=sqlite3,  # type: ignore[arg-type]
            query=query,
        )


def test_fetch_rows_resilient_quarantines_unicode_error_during_mapping(tmp_path: Path) -> None:
    """A row whose mapping raises UnicodeDecodeError is quarantined, not raised."""

    class _BadRow(tuple):  # type: ignore[type-arg]
        def __iter__(self) -> object:
            raise UnicodeDecodeError("utf-8", b"\x80", 0, 1, "boom")

    class _OneBadRowCursor:
        description = None

        def fetchall(self) -> list[tuple[object, ...]]:
            return [_BadRow(("M-bad",))]

    query = FetchQuery(select_columns_sql="id", where_sql="1", limit=10)
    with structlog.testing.capture_logs() as logs:
        results, delta = fetch_rows_resilient(
            _OneBadRowCursor(),
            db_path=tmp_path / "memory.db",
            dbapi=sqlite3,  # type: ignore[arg-type]
            query=query,
        )
    assert results == []
    assert delta == 1
    quarantine = [log for log in logs if log.get("action") == "memory_row_utf8_quarantined"]
    assert quarantine and quarantine[0].get("outcome") == "quarantined"


def test_search_skips_bad_utf8_row_and_preserves_filter(tmp_path: Path) -> None:
    """search() quarantines bad rows and keeps a status filter on the degraded path."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.store(_make_entry("M-needle-good", "haystack", status=MemoryStatus.ACTIVE))
    backend.close()

    # Bad row's id contains the search term so it is a LIKE candidate — the
    # decode error then fires while materialising the matched set.
    _inject_bad_utf8_row(db_path, "M-needle-bad", status="active")

    backend2 = SQLiteBackend(db_path)
    results = backend2.search(query="needle", status=MemoryStatus.ACTIVE, top_k=50)
    ids = {e.id for e in results}
    assert "M-needle-good" in ids
    assert "M-needle-bad" not in ids
    assert backend2.quarantine_count_utf8 >= 1
    backend2.close()


def _store_good_row_with_content(
    backend: SQLiteBackend,
    db_path: Path | str,
    entry_id: str,
    content: str,
    *,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> None:
    """Store a valid entry, then raw-UPDATE its ``content`` to caller value.

    Lets tests place LIKE metacharacters (``%``/``_``) verbatim in content
    without hand-crafting the full column tuple: the entry is created via the
    backend (so every enum/JSON column is valid and ``row_to_entry`` succeeds),
    then a raw UPDATE rewrites only ``content`` byte-exact.
    """
    backend.store(_make_entry(entry_id, status=status))
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE memories SET content = ? WHERE id = ?", (content, entry_id))
    conn.commit()
    conn.close()


def test_search_preserves_escape_clause_through_bytes_fallback(tmp_path: Path) -> None:
    """F4: LIKE metacharacters stay literal across the degraded (bytes-mode) path.

    A query of ``a%b`` must match only the row whose content is the literal
    ``a%b`` — never ``aZZZb`` (which would match if ``%`` acted as a wildcard).
    A bad-UTF-8 row forces the bytes-mode fallback, which re-executes the same
    WHERE (carrying the ``ESCAPE '\\'`` clause via FetchQuery). Asserting only
    the literal match returns proves the ESCAPE clause survived the fallback.
    """
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)

    # Literal-metachar match target, a would-be wildcard victim, and a bad-UTF-8
    # row whose id carries the literal term so it is a LIKE candidate — the
    # decode error then fires while materialising the matched set (forcing the
    # bytes-mode fallback). Its detail is the undecodable bytes.
    _store_good_row_with_content(backend, db_path, "M-literal", "a%b literal percent")
    _store_good_row_with_content(backend, db_path, "M-wildcard-victim", "aZZZb would-be wildcard")
    backend.close()
    _inject_bad_utf8_row(db_path, "a%b-bad", status="active")

    backend2 = SQLiteBackend(db_path)
    # The query 'a%b' is escaped to a literal; only M-literal (and the bad row,
    # which is quarantined) are LIKE candidates. M-wildcard-victim must NOT match.
    results = backend2.search(query="a%b", status=MemoryStatus.ACTIVE, top_k=50)
    ids = {e.id for e in results}

    assert "M-literal" in ids, "Literal 'a%b' content must match the escaped query"
    assert "M-wildcard-victim" not in ids, "ESCAPE clause must stop '%' acting as a wildcard"
    assert "a%b-bad" not in ids, "Bad-UTF-8 row is quarantined, not returned"
    assert backend2.quarantine_count_utf8 >= 1, "Bad row forced the bytes-mode fallback"
    backend2.close()


def test_search_underscore_metachar_literal_through_fallback(tmp_path: Path) -> None:
    """F4: a literal underscore query must not match arbitrary single chars."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)

    _store_good_row_with_content(backend, db_path, "M-underscore", "x_y literal underscore")
    _store_good_row_with_content(backend, db_path, "M-single-char", "xQy single char")
    backend.close()
    _inject_bad_utf8_row(db_path, "x_y-bad", status="active")

    backend2 = SQLiteBackend(db_path)
    results = backend2.search(query="x_y", status=MemoryStatus.ACTIVE, top_k=50)
    ids = {e.id for e in results}

    assert "M-underscore" in ids, "Literal 'x_y' must match the escaped query"
    assert "M-single-char" not in ids, "ESCAPE clause must stop '_' matching a single char"
    assert backend2.quarantine_count_utf8 >= 1, "Bad row forced the bytes-mode fallback"
    backend2.close()


def test_entries_with_assertions_survives_bad_utf8_row(tmp_path: Path) -> None:
    """entries_with_assertions() quarantines bad rows rather than failing."""
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    backend.close()

    # Inject a bad-UTF-8 row that ALSO carries assertions so it matches the WHERE.
    conn = sqlite3.connect(str(db_path))
    bad_hex = b"bad\x80\x81".hex()
    conn.execute(
        """
        INSERT OR REPLACE INTO memories (
            id, content, detail, tags, evidence, importance, status,
            recurrence, namespace, created_at, updated_at, access_count,
            session_count, q_value, q_observations, source,
            source_identity, client_profile, model_id,
            merged_from, consolidated_from, outcome_history,
            assertions, anchors, anchor_validity,
            type, nudge_line, expires_at, confidence,
            task_type, domain, phase_origin, phase_affinity,
            team_origin, protection_tier, sessions_surfaced,
            outcome_correlation, sync_hash, sync_seq,
            recall_count, helpful_count, unhelpful_count,
            vector_clock, metadata,
            published_to_platform, pending_delete, cross_validated
        ) VALUES (
            ?, ?, CAST(X'"""
        + bad_hex
        + """' AS TEXT),
            '[]', '[]', 0.5, 'active',
            1, 'default', '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00', 0,
            0, 0.5, 0, 'agent', '', '', '',
            '[]', '[]', '[]', '[{"kind":"file_exists","value":"x"}]', '[]', 1.0,
            'fact', '', '', 'medium', '', '[]', '', '[]',
            '', 'standard', 0, '', '', 0, 0, 0, 0,
            '{}', '{}', 0, 0, 0
        )
        """,
        ("M-bad-assert", "content"),
    )
    conn.commit()
    conn.close()

    backend2 = SQLiteBackend(db_path)
    # Must not raise; bad row is quarantined.
    results = backend2.entries_with_assertions()
    assert all(e.id != "M-bad-assert" for e in results)
    assert backend2.quarantine_count_utf8 >= 1
    backend2.close()
