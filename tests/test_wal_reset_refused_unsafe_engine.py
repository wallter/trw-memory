"""PRD-CORE-248 OQ-1 (REVERSED on review) — a resetting checkpoint is REFUSED below 3.51.3.

The PRD originally proposed permitting ``TRUNCATE`` on an engine carrying the
SQLite WAL-reset bug when the caller could certify it was the only live writer,
proving that certification with a bounded ``BEGIN EXCLUSIVE`` probe. Review
reversed it: SQLite refuses ``PRAGMA wal_checkpoint`` inside a transaction, so
the probe proves exclusivity at acquisition and cannot hold it across the reset.
A connection opened between the ``COMMIT`` and the PRAGMA reproduces exactly the
two-connection precondition the bug needs — a corruption class this repository
has already suffered once (`reference_memory_db_walreset_fix`).

So the gate is unconditional and has no caller-supplied escape. These tests are
the falsifier for that: they exist to fail if anyone reintroduces a permit,
whatever it is called, and to prove that the refusal is *stated* to the operator
rather than applied silently.

Everything here drives the real ``run_checkpoint`` / ``normalize_mode``
primitive against a real SQLite connection.
"""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest
import structlog

from trw_memory.storage import _wal_checkpoint
from trw_memory.storage._wal_checkpoint import (
    RESETTING_MODES,
    WAL_RESET_UNSAFE_REMEDY,
    normalize_mode,
    run_checkpoint,
)


def _wal_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO t (v) VALUES ('x')")
    return conn


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(RESETTING_MODES))
def test_resetting_mode_is_refused_on_an_unsafe_engine(mode: str) -> None:
    """Every resetting mode collapses to PASSIVE below 3.51.3."""
    assert normalize_mode(mode, wal_reset_safe=False) == "PASSIVE"


@pytest.mark.parametrize("mode", sorted(RESETTING_MODES))
def test_resetting_mode_is_honoured_on_a_safe_engine(mode: str) -> None:
    """Non-vacuity: the gate is the ENGINE, not a blanket ban on resetting modes."""
    assert normalize_mode(mode, wal_reset_safe=True) == mode


def test_normalize_mode_exposes_no_caller_escape_from_the_gate() -> None:
    """The falsifier: no parameter may let a caller opt out of the downgrade.

    A ``sole_writer=``/``force=``/``permit=`` keyword is exactly what review
    reversed. If one reappears, this fails before any test that would exercise
    it can pass.
    """
    params = inspect.signature(normalize_mode).parameters
    assert set(params) == {"mode", "wal_reset_safe"}, (
        f"normalize_mode grew a parameter; a caller-supplied reset permit is refused by "
        f"PRD-CORE-248 OQ-1 (reversed). Signature: {inspect.signature(normalize_mode)}"
    )
    assert not hasattr(_wal_checkpoint, "hold_exclusive_window"), (
        "the BEGIN EXCLUSIVE probe was deleted: it cannot be held across the checkpoint PRAGMA"
    )


def test_run_checkpoint_takes_no_permit_parameters() -> None:
    """The same falsifier one layer up."""
    params = set(inspect.signature(run_checkpoint).parameters)
    assert params == {"execute_pragma", "requested_mode", "wal_reset_safe", "db_path", "db_error"}


def test_backend_checkpoint_wal_takes_no_permit_parameters() -> None:
    """And at the SQLiteBackend seam trw-mcp actually calls."""
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    assert set(inspect.signature(SQLiteBackend.checkpoint_wal).parameters) == {"self", "mode"}


# ---------------------------------------------------------------------------
# Behaviour on a real connection
# ---------------------------------------------------------------------------


def test_no_resetting_pragma_reaches_an_unsafe_engine(tmp_path: Path) -> None:
    """NFR04: zero TRUNCATE/RESTART executions, and the refusal is stated."""
    conn = _wal_connection(tmp_path / "m.db")
    executed: list[str] = []

    def _record(sql: str) -> object:
        executed.append(sql)
        return conn.execute(sql).fetchone()

    try:
        with structlog.testing.capture_logs() as logs:
            result = run_checkpoint(_record, "TRUNCATE", wal_reset_safe=False, db_path=str(tmp_path / "m.db"))

        assert result["mode"] == "PASSIVE"
        assert not [sql for sql in executed if "TRUNCATE" in sql or "RESTART" in sql]
        refusals = [log for log in logs if log.get("event") == "wal_reset_refused_unsafe_engine"]
        assert refusals, f"the refusal must be visible to an operator; got {logs}"
        assert refusals[0]["requested"] == "TRUNCATE"
        assert refusals[0]["ran"] == "PASSIVE"
        assert "3.51.3" in str(refusals[0]["remedy"]), "the row must name the remedy, not just the symptom"
    finally:
        conn.close()


def test_passive_request_is_not_reported_as_a_refusal(tmp_path: Path) -> None:
    """Non-vacuity: the refusal event fires only when a reset was actually asked for."""
    conn = _wal_connection(tmp_path / "m.db")
    try:
        with structlog.testing.capture_logs() as logs:
            run_checkpoint(
                lambda sql: conn.execute(sql).fetchone(),
                "PASSIVE",
                wal_reset_safe=False,
                db_path=str(tmp_path / "m.db"),
            )
        assert not [log for log in logs if log.get("event") == "wal_reset_refused_unsafe_engine"]
    finally:
        conn.close()


def test_truncate_runs_and_shrinks_the_wal_on_a_safe_engine(tmp_path: Path) -> None:
    """The behaviour the refusal costs us, and what an engine upgrade buys back."""
    db_path = tmp_path / "m.db"
    conn = _wal_connection(db_path)
    try:
        conn.execute("BEGIN")
        for i in range(2000):
            conn.execute("INSERT INTO t (v) VALUES (?)", (f"row-{i}" * 20,))
        conn.execute("COMMIT")
        wal_path = Path(f"{db_path}-wal")
        before = wal_path.stat().st_size
        assert before > 0

        result = run_checkpoint(
            lambda sql: conn.execute(sql).fetchone(),
            "TRUNCATE",
            wal_reset_safe=True,
            db_path=str(db_path),
        )

        assert result["mode"] == "TRUNCATE"
        after = wal_path.stat().st_size if wal_path.exists() else 0
        assert after < before, "on a fixed engine TRUNCATE must reclaim WAL bytes"
    finally:
        conn.close()


def test_backend_refuses_the_reset_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the real SQLiteBackend, with the engine forced unsafe."""
    from trw_memory.storage import _dbapi
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    monkeypatch.setattr(_dbapi, "is_wal_reset_safe", lambda: False)
    backend = SQLiteBackend(tmp_path / "m.db")
    try:
        assert backend.wal_reset_safe is False
        assert backend.checkpoint_wal("TRUNCATE")["mode"] == "PASSIVE"
        assert backend.checkpoint_wal("RESTART")["mode"] == "PASSIVE"
    finally:
        backend.close()


def test_remedy_string_is_the_single_source_for_every_surface() -> None:
    """One sentence, exported, so log/doctor/docstring cannot drift apart."""
    assert "3.51.3" in WAL_RESET_UNSAFE_REMEDY
    assert "pysqlite3" in WAL_RESET_UNSAFE_REMEDY
