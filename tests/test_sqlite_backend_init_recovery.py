"""SQLiteBackend init-time recovery routing tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
import structlog

from trw_memory.exceptions import CorruptDatabaseUnsalvageableError
from trw_memory.storage._init_helpers import open_connection_with_recovery
from trw_memory.storage._recovery import classify_recovery_preflight, recovery_state_path, write_recovery_state


class _FakeBackend:
    def __init__(self, exc: sqlite3.DatabaseError) -> None:
        self.exc = exc
        self.recover_called = False
        self.open_without_called = False

    def _open_and_configure(self, _db_path: Path, **_: object) -> Any:
        raise self.exc

    #: What the (also-locked) probe reports. ``None`` is UNKNOWN — see
    #: ``test_lock_contention_with_an_UNKNOWN_probe_is_not_a_wipe``.
    has_data: bool | None = True

    def _db_has_data(self, _db_path: Path, *, dbapi: Any) -> bool | None:
        return self.has_data

    def _open_without_integrity_check(self, _db_path: Path, *, dbapi: Any) -> Any:
        self.open_without_called = True
        return sqlite3.connect(":memory:")

    def recover_db(
        self,
        _db_path: Path,
        *,
        dbapi: Any,
        recovery_policy: str,
        corrupt_backup_keep: int,
        rebuild_from_cold: bool,
    ) -> Any:
        self.recover_called = True
        return sqlite3.connect(":memory:")


def test_quick_check_failure_with_rows_recovers_instead_of_opening_corrupt_db(tmp_path: Path) -> None:
    """A row-count probe is not a health check; failed quick_check must recover."""
    backend = _FakeBackend(sqlite3.DatabaseError("database disk image is malformed (quick_check failed twice)"))

    conn, integrity_warning, recovered = open_connection_with_recovery(
        backend,  # type: ignore[arg-type]
        tmp_path / "memory.db",
        dbapi=sqlite3,
        recovery_policy="strict",
        corrupt_backup_keep=5,
        rebuild_from_cold=True,
    )

    conn.close()
    assert backend.recover_called is True
    assert backend.open_without_called is False
    assert integrity_warning is False
    assert recovered is True


def test_lock_contention_with_rows_keeps_non_destructive_open_without_probe(tmp_path: Path) -> None:
    """Explicit SQLite lock/busy errors remain the non-destructive fallback case."""
    backend = _FakeBackend(sqlite3.DatabaseError("database is locked"))

    conn, integrity_warning, recovered = open_connection_with_recovery(
        backend,  # type: ignore[arg-type]
        tmp_path / "memory.db",
        dbapi=sqlite3,
        recovery_policy="strict",
        corrupt_backup_keep=5,
        rebuild_from_cold=True,
    )

    conn.close()
    assert backend.recover_called is False
    assert backend.open_without_called is True
    assert integrity_warning is True
    assert recovered is False
    assert recovery_state_path(tmp_path / "memory.db").exists()


def test_lock_contention_with_an_UNKNOWN_probe_is_not_a_wipe(tmp_path: Path) -> None:
    """The defect, end to end: a busy machine used to destroy a populated store.

    ``open_connection_with_recovery`` reaches this branch when the open failed on
    lock/busy, and it asks ``_db_has_data`` whether there is anything to lose. But
    under contention the PROBE is locked too, so it returned ``False`` — "no rows"
    — and the ``else`` branch renamed the live database to ``.corrupt.bak`` and
    initialised a blank schema.

    The probe now returns ``None`` for lock/busy and the caller tests
    ``is not False``, so UNKNOWN takes the non-destructive path. This is the arm
    the previous test could not cover: it hardcoded ``True``, i.e. a probe that
    succeeded, which is the one case that was never broken.
    """
    backend = _FakeBackend(sqlite3.DatabaseError("database is locked"))
    backend.has_data = None

    conn, integrity_warning, recovered = open_connection_with_recovery(
        backend,  # type: ignore[arg-type]
        tmp_path / "memory.db",
        dbapi=sqlite3,
        recovery_policy="strict",
        corrupt_backup_keep=5,
        rebuild_from_cold=True,
    )

    conn.close()
    assert backend.recover_called is False, "an UNKNOWN probe under lock contention triggered destructive recovery"
    assert backend.open_without_called is True
    assert integrity_warning is True
    assert recovered is False


def test_lock_contention_on_an_empty_store_is_not_a_wipe(tmp_path: Path) -> None:
    """A NEW store has no rows while other connections are creating it.

    This arm used to recover ("nothing to lose"), renaming the file and
    unlinking its WAL under the connections still creating it: their commits
    then returned success into files nothing reads again. Lock/busy says
    nothing about corruption; ``test_a_malformed_image_is_still_recovered_not_retried``
    is the partner that proves recovery still runs for a malformed image.
    """
    backend = _FakeBackend(sqlite3.DatabaseError("database is locked"))
    backend.has_data = False

    conn, integrity_warning, recovered = open_connection_with_recovery(
        backend,  # type: ignore[arg-type]
        tmp_path / "memory.db",
        dbapi=sqlite3,
        recovery_policy="strict",
        corrupt_backup_keep=5,
        rebuild_from_cold=True,
    )

    conn.close()
    assert backend.recover_called is False, "lock contention on an empty store triggered destructive recovery"
    assert backend.open_without_called is True
    assert (integrity_warning, recovered) == (True, False)


def test_preflight_classifies_large_db_as_degraded_open(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")

    preflight = classify_recovery_preflight(db_path, inline_max_bytes=4)

    assert preflight.classification == "degraded_open_with_background_recovery"
    assert preflight.reason == "db_exceeds_inline_recovery_budget"
    assert preflight.db_size_bytes == 10


def test_degraded_preflight_blocks_inline_recovery_and_persists_state(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    backend = _FakeBackend(sqlite3.DatabaseError("database disk image is malformed"))

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        open_connection_with_recovery(
            backend,  # type: ignore[arg-type]
            db_path,
            dbapi=sqlite3,
            recovery_policy="strict",
            corrupt_backup_keep=5,
            rebuild_from_cold=True,
            recovery_inline_max_bytes=4,
        )

    assert backend.recover_called is False
    assert "degraded_open_with_background_recovery" in recovery_state_path(db_path).read_text(encoding="utf-8")


def test_recovery_state_write_is_valid_json_and_classifies_hard_fail(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")

    write_recovery_state(db_path, status="hard_fail", reason="inline_recovery_failed", db_size_bytes=10)

    state = recovery_state_path(db_path).read_text(encoding="utf-8")
    assert '"status": "hard_fail"' in state
    assert classify_recovery_preflight(db_path, inline_max_bytes=1024).classification == "hard_fail"


@pytest.mark.parametrize("status", ["pending", "running", "degraded_open_with_background_recovery"])
def test_valid_pending_status_yields_degraded_open(tmp_path: Path, status: str) -> None:
    """Valid in-flight statuses keep the degraded-open-with-background-recovery path."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    write_recovery_state(db_path, status=status, reason="t", db_size_bytes=10)

    # Inline budget large enough that size alone would classify fast_open;
    # the persisted in-flight status is what must force the degraded path.
    preflight = classify_recovery_preflight(db_path, inline_max_bytes=1024)

    assert preflight.classification == "degraded_open_with_background_recovery"
    assert preflight.reason == "recovery_already_pending"
    assert preflight.persisted_status == status


def test_malformed_json_state_does_not_raise_and_falls_through_to_size(tmp_path: Path) -> None:
    """A truncated/garbage sidecar must not crash; classification falls back to size."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    recovery_state_path(db_path).write_text("{not valid json", encoding="utf-8")

    # Oversized DB still degrades on the size budget; no hard_fail, no raise.
    over_budget = classify_recovery_preflight(db_path, inline_max_bytes=4)
    assert over_budget.classification == "degraded_open_with_background_recovery"
    assert over_budget.reason == "db_exceeds_inline_recovery_budget"
    assert over_budget.persisted_status == ""

    # Within budget falls through to fast_open — malformed state is ignored, not hard_fail.
    within_budget = classify_recovery_preflight(db_path, inline_max_bytes=1024)
    assert within_budget.classification == "fast_open"


def test_non_utf8_state_does_not_raise_and_falls_through_to_size(tmp_path: Path) -> None:
    """Non-UTF-8 bytes raise UnicodeDecodeError on read_text — the seam must absorb it."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    # 0xFF is invalid UTF-8; the prior read_text(encoding="utf-8") would crash here.
    recovery_state_path(db_path).write_bytes(b"\xff\xfe\x00\x80garbage")

    within_budget = classify_recovery_preflight(db_path, inline_max_bytes=1024)
    assert within_budget.classification == "fast_open"
    assert within_budget.persisted_status == ""

    over_budget = classify_recovery_preflight(db_path, inline_max_bytes=4)
    assert over_budget.classification == "degraded_open_with_background_recovery"


def test_non_object_json_state_does_not_hard_fail(tmp_path: Path) -> None:
    """A JSON array/scalar is not an object — status is absent, so no hard_fail."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    recovery_state_path(db_path).write_text('["hard_fail"]', encoding="utf-8")

    preflight = classify_recovery_preflight(db_path, inline_max_bytes=1024)

    assert preflight.classification == "fast_open"
    assert preflight.persisted_status == ""


def test_non_string_status_does_not_hard_fail(tmp_path: Path) -> None:
    """A non-string status (e.g. a nested object) must not be coerced into a hard_fail trigger."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    recovery_state_path(db_path).write_text('{"status": {"nested": "hard_fail"}}', encoding="utf-8")

    preflight = classify_recovery_preflight(db_path, inline_max_bytes=1024)

    assert preflight.classification == "fast_open"
    assert preflight.persisted_status == ""


def test_corrupt_state_diagnostics_are_content_free(tmp_path: Path) -> None:
    """Any logs emitted for a corrupt sidecar must leak neither the path nor the raw payload."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    state_path = recovery_state_path(db_path)
    secret_marker = "SECRET-sk-live-DEADBEEF"
    # Malformed JSON carrying a secret marker; the seam must not echo it or the path.
    state_path.write_text(f"{{not json {secret_marker}", encoding="utf-8")

    with structlog.testing.capture_logs() as logs:
        classify_recovery_preflight(db_path, inline_max_bytes=1024)

    for event in logs:
        rendered = repr(event)
        assert secret_marker not in rendered
        assert str(state_path) not in rendered
        assert state_path.name not in rendered


# --- SQLITE_IOERR is not corruption (trw-memory 3.1.0, L-8QV8) ---------------------


def _io_error(message: str = "vtable constructor failed: memories_fts") -> sqlite3.OperationalError:
    """What the 6.0.0 Linux leg's daemon got from ``quick_check``: an I/O error, not a malformed image."""
    exc = sqlite3.OperationalError(message)
    exc.sqlite_errorcode = 10 | (13 << 8)  # type: ignore[attr-defined]  # SQLITE_IOERR_SHORT_READ-style extended code
    return exc


class _FlakyBackend(_FakeBackend):
    """Raises its error for the first *failures* opens, then opens cleanly."""

    def __init__(self, exc: sqlite3.DatabaseError, failures: int) -> None:
        super().__init__(exc)
        self.failures = failures
        self.opens = 0

    def _open_and_configure(self, _db_path: Path, **_: object) -> Any:
        self.opens += 1
        if self.opens <= self.failures:
            raise self.exc
        return sqlite3.connect(":memory:")


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr("trw_memory.storage._init_helpers.time.sleep", sleeps.append)
    return sleeps


def _open(backend: object, db_path: Path) -> tuple[Any, bool, bool]:
    return open_connection_with_recovery(
        backend,  # type: ignore[arg-type]
        db_path,
        dbapi=sqlite3,
        recovery_policy="strict",
        corrupt_backup_keep=5,
        rebuild_from_cold=True,
    )


def test_a_transient_io_error_is_retried_into_a_checked_open(tmp_path: Path, no_backoff: list[float]) -> None:
    backend = _FlakyBackend(_io_error(), failures=1)

    conn, integrity_warning, recovered = _open(backend, tmp_path / "memory.db")

    conn.close()
    assert backend.opens == 2
    assert no_backoff, "the retry waits before opening again"
    assert (integrity_warning, recovered, backend.recover_called, backend.open_without_called) == (
        False,
        False,
        False,
        False,
    )
    assert not recovery_state_path(tmp_path / "memory.db").exists()


@pytest.mark.parametrize("has_data", [True, None, False])
def test_a_persistent_io_error_degrades_and_never_quarantines(
    tmp_path: Path, no_backoff: list[float], has_data: bool | None
) -> None:
    """Even a store whose probe says "no rows" is left in place: an I/O error says nothing about its content."""
    backend = _FlakyBackend(_io_error(), failures=99)
    backend.has_data = has_data

    conn, integrity_warning, recovered = _open(backend, tmp_path / "memory.db")

    conn.close()
    assert backend.recover_called is False, "an I/O error sent a store down the destructive recovery path"
    assert backend.open_without_called is True
    assert (integrity_warning, recovered) == (True, False)
    assert backend.opens == 3
    state = classify_recovery_preflight(tmp_path / "memory.db", inline_max_bytes=1024)
    assert state.classification != "hard_fail"


def test_a_malformed_image_is_still_recovered_not_retried(tmp_path: Path, no_backoff: list[float]) -> None:
    """Non-vacuity partner: only the I/O error class changed; real corruption still recovers at once."""
    backend = _FlakyBackend(sqlite3.DatabaseError("database disk image is malformed (quick_check failed twice)"), 99)

    _open(backend, tmp_path / "memory.db")

    assert backend.recover_called is True
    assert backend.opens == 1
    assert no_backoff == []


def test_a_real_store_that_hits_io_errors_keeps_its_file_and_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_backoff: list[float]
) -> None:
    """End to end on a real file: the group A store stays where it is, readable, with no .corrupt.bak."""
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    db_path = tmp_path / "memory.db"
    seeded = SQLiteBackend(db_path)
    for n in range(3):
        seeded.store(MemoryEntry(id=f"L-{n}", content=f"row {n}"))
    seeded.close()
    inode = db_path.stat().st_ino

    def io_error(*_a: object, **_k: object) -> Any:
        raise _io_error()

    monkeypatch.setattr(SQLiteBackend, "_open_and_configure", staticmethod(io_error))
    reopened = SQLiteBackend(db_path)

    assert reopened.count() == 3
    reopened.close()
    assert db_path.stat().st_ino == inode, "the live store was replaced"
    assert list(tmp_path.glob("memory.db.corrupt*")) == []


# --- lock/busy on a new store's first opens (7.0.0 daemon P0) ----------------------


def test_a_transient_lock_is_retried_into_a_checked_open(tmp_path: Path, no_backoff: list[float]) -> None:
    """Two first opens of a new store both switch it to WAL; SQLite answers one "locked" at once."""
    backend = _FlakyBackend(sqlite3.OperationalError("database is locked"), failures=2)

    conn, integrity_warning, recovered = _open(backend, tmp_path / "memory.db")

    conn.close()
    assert backend.opens == 3
    assert len(no_backoff) == 2 and max(no_backoff) < 0.5, "lock waits are short, not the I/O backoff"
    assert (integrity_warning, recovered, backend.recover_called, backend.open_without_called) == (
        False,
        False,
        False,
        False,
    )
    assert not recovery_state_path(tmp_path / "memory.db").exists()


def test_a_write_through_an_open_connection_survives_a_locked_first_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_backoff: list[float]
) -> None:
    """End to end on a real file, with the daemon's interleaving forced by a hook.

    Writer A holds a connection to a new, empty store and has just passed its
    stale-handle check. Writer B's checked open then fails "database is
    locked" on every attempt. B used to recover the "empty" store -- rename it
    to ``.corrupt.bak`` and unlink its WAL under A -- so A's write, already
    past the check that would have reconnected it, returned success into a
    file nothing reads again (the daemon test counted 199 of 200 rows).
    """
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    db_path = tmp_path / "memory.db"
    writer_a = SQLiteBackend(db_path)
    real_open = SQLiteBackend.__dict__["_open_and_configure"]  # the staticmethod itself, not the unwrapped function
    real_is_stale = writer_a._stale_detector.is_stale
    opened_b: list[SQLiteBackend] = []

    def locked(*_a: object, **_k: object) -> Any:
        raise sqlite3.OperationalError("database is locked")

    def check_then_let_b_open() -> bool:
        stale = real_is_stale()
        monkeypatch.setattr(SQLiteBackend, "_open_and_configure", staticmethod(locked))
        opened_b.append(SQLiteBackend(db_path))
        monkeypatch.setattr(SQLiteBackend, "_open_and_configure", real_open)
        return stale

    monkeypatch.setattr(writer_a._stale_detector, "is_stale", check_then_let_b_open)
    writer_a.store(MemoryEntry(id="L-a", content="written through the first connection"))
    monkeypatch.setattr(writer_a._stale_detector, "is_stale", real_is_stale)
    (writer_b,) = opened_b
    writer_b.store(MemoryEntry(id="L-b", content="written through the degraded open"))
    writer_a.close()
    writer_b.close()

    with SQLiteBackend(db_path) as reader:
        assert reader.count() == 2, "a write that returned success is missing from the live store"
    assert list(tmp_path.glob("memory.db.corrupt*")) == []
