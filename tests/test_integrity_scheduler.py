"""Tests for PRD-INFRA-063 periodic integrity scheduler."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from trw_memory.storage._integrity_scheduler import IntegrityScheduler

if TYPE_CHECKING:
    from pytest import MonkeyPatch


# ---------------------------------------------------------------------------
# Test-only mirrors of the removed public ``is_running``/``run_once`` seams.
# ---------------------------------------------------------------------------


def _is_running(sched: IntegrityScheduler) -> bool:
    with sched._lock:
        t = sched._thread
    return t is not None and t.is_alive()


def _run_once(sched: IntegrityScheduler) -> bool:
    ok, detail = sched._probe()
    sched.last_check_at = time.time()
    sched.last_check_ok = ok
    sched._write_sentinel()
    sched._report(ok, detail)
    return ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_healthy_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()


def _corrupt_db(path: Path) -> None:
    """Scramble bytes in the DB header to force quick_check failure."""
    data = bytearray(path.read_bytes())
    # Page 2 header — writing garbage there reliably trips quick_check.
    for offset in range(4096, 4196):
        if offset < len(data):
            data[offset] = 0xFF
    path.write_bytes(bytes(data))


# ---------------------------------------------------------------------------
# Opt-in default (sprint exit criterion guard)
# ---------------------------------------------------------------------------


def test_scheduler_disabled_when_interval_zero(tmp_path: Path) -> None:
    """CRITICAL: interval=0 MUST NOT start any background thread.

    Regression guard for sprint exit criterion `opt-in-defaults`.
    """
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    sched = IntegrityScheduler(db, interval_minutes=0)
    sched.start()
    assert _is_running(sched) is False, "interval=0 must be a hard disable"
    sched.stop()


def test_scheduler_starts_when_interval_positive(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    sched = IntegrityScheduler(db, interval_minutes=1)
    sched.start()
    assert _is_running(sched) is True
    sched.stop(timeout=1.0)
    assert _is_running(sched) is False


# ---------------------------------------------------------------------------
# run_once (synchronous probe)
# ---------------------------------------------------------------------------


def test_run_once_ok_on_healthy_db(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    sched = IntegrityScheduler(db, interval_minutes=0)
    assert _run_once(sched) is True
    assert sched.last_check_ok is True
    assert sched.last_check_at is not None
    assert sched.last_check_at > 0


def test_run_once_failure_on_corrupt_db(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    _corrupt_db(db)
    sched = IntegrityScheduler(db, interval_minutes=0)
    # Corrupted DB → quick_check returns non-ok OR raises; either path sets False.
    assert _run_once(sched) is False
    assert sched.last_check_ok is False


def test_run_once_failure_on_missing_db(tmp_path: Path) -> None:
    missing = tmp_path / "never_created.db"
    sched = IntegrityScheduler(missing, interval_minutes=0)
    assert _run_once(sched) is False
    assert sched.last_check_ok is False


# ---------------------------------------------------------------------------
# Regression callback
# ---------------------------------------------------------------------------


def test_on_regression_fires_when_corrupt(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    _corrupt_db(db)

    captured: list[tuple[Path, str]] = []

    def _capture(p: Path, detail: str, **_kwargs) -> None:
        captured.append((p, detail))

    sched = IntegrityScheduler(db, interval_minutes=0, on_regression=_capture)
    result = _run_once(sched)
    assert result is False
    assert len(captured) == 1
    assert captured[0][0] == db
    # Detail should contain non-"ok" content.
    assert captured[0][1] != "ok"


def test_on_regression_does_not_fire_when_healthy(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _make_healthy_db(db)

    captured: list[tuple[Path, str]] = []
    sched = IntegrityScheduler(db, interval_minutes=0, on_regression=lambda p, d: captured.append((p, d)))
    _run_once(sched)
    assert captured == []


def test_on_regression_callback_exception_swallowed(tmp_path: Path) -> None:
    """A buggy callback must NOT crash the scheduler."""
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    _corrupt_db(db)

    def _raiser(_p: Path, _detail: str) -> None:
        raise RuntimeError("boom")

    sched = IntegrityScheduler(db, interval_minutes=0, on_regression=_raiser)
    escaped: list[str] = []
    try:
        _run_once(sched)
    except Exception as exc:
        escaped.append(repr(exc))
    assert escaped == [], f"a buggy on_regression callback escaped run_once: {escaped}"


# ---------------------------------------------------------------------------
# Thread lifecycle
# ---------------------------------------------------------------------------


def test_start_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    sched = IntegrityScheduler(db, interval_minutes=60)
    sched.start()
    sched.start()  # second start is a no-op
    assert _is_running(sched) is True
    sched.stop(timeout=1.0)


def test_stop_before_start_is_safe(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    sched = IntegrityScheduler(db, interval_minutes=0)
    sched.stop()  # must not raise
    assert sched._thread is None


def test_stop_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    sched = IntegrityScheduler(db, interval_minutes=60)
    sched.start()
    sched.stop(timeout=1.0)
    sched.stop(timeout=1.0)  # second stop is fine
    assert sched._thread is None or not sched._thread.is_alive()


def test_thread_is_daemon(tmp_path: Path) -> None:
    """The background thread MUST be daemon so interpreter exit is not blocked."""
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    sched = IntegrityScheduler(db, interval_minutes=60)
    sched.start()
    assert sched._thread is not None  # type: ignore[attr-defined]
    assert sched._thread.daemon is True  # type: ignore[attr-defined]
    sched.stop(timeout=1.0)


# ---------------------------------------------------------------------------
# Background loop end-to-end
# ---------------------------------------------------------------------------


def test_loop_runs_probe_at_interval(tmp_path: Path) -> None:
    """Shrink the interval-seconds to 0.05s and verify at least one probe runs."""
    db = tmp_path / "memory.db"
    _make_healthy_db(db)

    sched = IntegrityScheduler(db, interval_minutes=1)
    # Override the private _interval_seconds for fast iteration in tests.
    sched._interval_seconds = 0.05  # type: ignore[attr-defined]
    sched.start()
    time.sleep(0.25)
    sched.stop(timeout=1.0)
    assert sched.last_check_at is not None


def test_loop_uses_dedicated_readonly_connection(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Regression: the probe MUST open a fresh read-only uri connection.

    Asserts the scheduler opens its own connection rather than reusing the
    backend's write connection (which the mitigation plan explicitly
    forbids because it serializes writes behind quick_check).
    """
    db = tmp_path / "memory.db"
    _make_healthy_db(db)

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real_connect = sqlite3.connect

    def _recording_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        calls.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _recording_connect)

    sched = IntegrityScheduler(db, interval_minutes=0)
    _run_once(sched)

    assert len(calls) == 1, "expected exactly one connect() in run_once()"
    first_args, first_kwargs = calls[0]
    first_arg = cast("str", first_args[0])
    assert first_arg.startswith("file:"), "probe MUST use a file: URI"
    assert "mode=ro" in first_arg, "probe MUST use read-only mode"
    assert first_kwargs.get("uri") is True


# ---------------------------------------------------------------------------
# last_check_at / last_check_ok accessors (for C3 dashboard)
# ---------------------------------------------------------------------------


def test_last_check_at_initially_none(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    sched = IntegrityScheduler(db, interval_minutes=60)
    assert sched.last_check_at is None
    assert sched.last_check_ok is None


def test_last_check_at_updates_on_run_once(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    sched = IntegrityScheduler(db, interval_minutes=0)
    t0 = time.time()
    _run_once(sched)
    assert sched.last_check_at is not None
    assert sched.last_check_at >= t0 - 0.5


# ---------------------------------------------------------------------------
# PRD-SEC-016 round-5: identity check around the read-only probe connect
# ---------------------------------------------------------------------------


def test_a_db_identity_change_between_stat_and_connect_is_reported_as_a_regression(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A store swapped during the probe's read-only connect (connect_registered refuses it) surfaces
    as a regression, never as a check of whatever now occupies db_path."""
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    sched = IntegrityScheduler(db, interval_minutes=0)

    from tests._swap_after_pin import swap_after_pin

    swap_after_pin(monkeypatch, db)

    ok = _run_once(sched)

    assert ok is False
    assert sched.last_check_ok is False


def test_an_unchanged_db_through_the_probe_still_reports_healthy(tmp_path: Path) -> None:
    """Regression control: the identity check must not false-positive on the ordinary, unraced probe."""
    db = tmp_path / "memory.db"
    _make_healthy_db(db)
    sched = IntegrityScheduler(db, interval_minutes=0)

    ok = _run_once(sched)

    assert ok is True
    assert sched.last_check_ok is True
