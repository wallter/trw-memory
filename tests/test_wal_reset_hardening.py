"""WAL-reset corruption hardening (2026-05-20 memory.db corruption incident).

Covers the structural fixes:
1. Boot-time warning when the active SQLite engine predates the 3.51.3
   WAL-reset bug fix (is_wal_reset_safe() is False).
2. checkpoint_wal serializes on the single owning connection (the fix that
   removes the two-connection race that detonates the WAL-reset bug).
3. Robust primary salvage walks rowids via a secondary index and skips
   corrupt leaf pages instead of aborting at the first one (the data-loss
   path that salvaged 0 rows on 2026-05-20).
4. The WAL-checkpoint primitive (`_wal_checkpoint.run_checkpoint` /
   `normalize_mode`): the unsafe-engine resetting-mode gate, the busy
   PASSIVE fallback, fail-open on sqlite errors, and the exported
   `CheckpointResult` contract.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import structlog

from trw_memory.storage import _recovery
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# 1. Boot-time WAL-reset safety gate
# ---------------------------------------------------------------------------


def test_boot_warns_when_wal_reset_unsafe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing a backend on an unsafe engine logs sqlite_wal_reset_unsafe."""
    from trw_memory.storage import _dbapi

    monkeypatch.setattr(_dbapi, "is_wal_reset_safe", lambda: False)
    with structlog.testing.capture_logs() as logs:
        backend = SQLiteBackend(tmp_path / "m.db")
    try:
        events = [log for log in logs if log.get("event") == "sqlite_wal_reset_unsafe"]
        assert len(events) == 1, "unsafe engine must emit exactly one boot warning"
        assert events[0]["log_level"] == "warning"
        assert backend.wal_reset_safe is False
    finally:
        backend.close()


def test_boot_silent_when_wal_reset_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A safe engine emits no warning and sets wal_reset_safe True."""
    from trw_memory.storage import _dbapi

    monkeypatch.setattr(_dbapi, "is_wal_reset_safe", lambda: True)
    with structlog.testing.capture_logs() as logs:
        backend = SQLiteBackend(tmp_path / "m.db")
    try:
        events = [log for log in logs if log.get("event") == "sqlite_wal_reset_unsafe"]
        assert events == []
        assert backend.wal_reset_safe is True
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 2. Single-connection checkpoint
# ---------------------------------------------------------------------------


def test_checkpoint_wal_returns_status_dict(tmp_path: Path) -> None:
    """checkpoint_wal returns busy/checkpointed/mode and does not raise."""
    backend = SQLiteBackend(tmp_path / "m.db")
    try:
        result = backend.checkpoint_wal("TRUNCATE")
        assert set(result) >= {"busy", "checkpointed", "mode"}
        assert result["mode"] in {"TRUNCATE", "PASSIVE"}
        assert result["busy"] == 0
    finally:
        backend.close()


def test_checkpoint_wal_invalid_mode_defaults_passive(tmp_path: Path) -> None:
    """An unknown checkpoint mode is coerced to PASSIVE rather than injected."""
    backend = SQLiteBackend(tmp_path / "m.db")
    try:
        result = backend.checkpoint_wal("DROP TABLE memories")
        assert result["mode"] == "PASSIVE"
    finally:
        backend.close()


def test_checkpoint_wal_uses_owning_connection_not_new(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """checkpoint_wal must run on self._conn — never open a competing connection."""
    backend = SQLiteBackend(tmp_path / "m.db")
    try:
        # Any attempt to open a second connection is the corruption trigger.
        monkeypatch.setattr(
            sqlite3,
            "connect",
            lambda *a, **k: pytest.fail("checkpoint_wal opened a competing connection"),
        )
        result = backend.checkpoint_wal("PASSIVE")
        assert result["mode"] == "PASSIVE"
    finally:
        backend.close()


def test_checkpoint_wal_unsafe_engine_coerces_truncate_to_passive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On an unsafe engine, a resetting TRUNCATE checkpoint downgrades to PASSIVE.

    A WAL reset is the corruption trigger; PASSIVE never resets the WAL.
    """
    from trw_memory.storage import _dbapi

    monkeypatch.setattr(_dbapi, "is_wal_reset_safe", lambda: False)
    backend = SQLiteBackend(tmp_path / "m.db")
    try:
        assert backend.wal_reset_safe is False
        assert backend.checkpoint_wal("TRUNCATE")["mode"] == "PASSIVE"
        assert backend.checkpoint_wal("RESTART")["mode"] == "PASSIVE"
    finally:
        backend.close()


def test_checkpoint_wal_safe_engine_allows_truncate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a fixed engine (>=3.51.3) TRUNCATE is allowed (reclaims WAL space)."""
    from trw_memory.storage import _dbapi

    monkeypatch.setattr(_dbapi, "is_wal_reset_safe", lambda: True)
    backend = SQLiteBackend(tmp_path / "m.db")
    try:
        assert backend.wal_reset_safe is True
        assert backend.checkpoint_wal("TRUNCATE")["mode"] == "TRUNCATE"
    finally:
        backend.close()


def test_concurrent_checkpoints_serialize_without_error(tmp_path: Path) -> None:
    """Many threads checkpointing the SAME connection serialize via the lock.

    Validates the locking claim: the single owning connection is never used
    by two threads at once, so checkpoints cannot race (which on an unsafe
    engine would be the corruption condition)."""
    import threading

    from trw_memory.models.memory import MemoryEntry

    backend = SQLiteBackend(tmp_path / "m.db")
    try:
        for i in range(20):
            backend.store(MemoryEntry(id=f"e{i}", content=f"c{i}"))

        results: list[dict[str, object]] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()  # maximize contention
            try:
                results.append(backend.checkpoint_wal("TRUNCATE"))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"concurrent checkpoints raised: {errors}"
        assert len(results) == 8
        assert all("mode" in r and "busy" in r for r in results)
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 3. Robust primary salvage
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows: list[object]) -> None:
        self._rows = list(rows)

    def fetchone(self) -> object:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[object]:
        rows, self._rows = self._rows, []
        return rows


class _FakeConn:
    """Connection whose rowid 2 lives on a 'corrupt page' (raises on fetch)."""

    def __init__(self) -> None:
        self.closed = False

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> object:
        if "INDEXED BY" in sql:
            return _FakeCursor([(1,), (2,), (3,), None])
        if "WHERE rowid=?" in sql:
            assert params is not None
            if params[0] == 2:
                raise sqlite3.DatabaseError("database disk image is malformed")
            return _FakeResult({"id": f"id-{params[0]}", "rowid": params[0]})
        if "SELECT rowid FROM memories" in sql:  # plain-scan fallback
            return iter([(1,), (3,)])
        raise sqlite3.DatabaseError("unexpected query")

    def close(self) -> None:
        self.closed = True


class _FakeResult:
    def __init__(self, row: dict[str, object]) -> None:
        self._row = row

    def fetchone(self) -> dict[str, object]:
        return self._row


def test_robust_salvage_skips_corrupt_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Salvage recovers every readable row and skips the corrupt-page row."""
    fake = _FakeConn()
    monkeypatch.setattr(_recovery, "_connection_connect", lambda *a, **k: fake)

    failed, rows = _recovery._attempt_primary_salvage(
        Path("/nonexistent/backup.db"), dbapi=sqlite3, sqlcipher_key_hex=None
    )

    assert failed is False
    salvaged_ids = sorted(r["id"] for r in rows)
    assert salvaged_ids == ["id-1", "id-3"], "row 2 (corrupt page) skipped; 1 and 3 recovered"
    assert fake.closed is True


def test_robust_salvage_recovers_all_rows_on_healthy_db(tmp_path: Path) -> None:
    """Regression: on a healthy DB salvage returns every row (no false loss)."""
    db = tmp_path / "m.db"
    backend = SQLiteBackend(db)
    from trw_memory.models.memory import MemoryEntry

    for i in range(5):
        backend.store(MemoryEntry(id=f"e{i}", content=f"content {i}"))
    backend.close()

    failed, rows = _recovery._attempt_primary_salvage(db, dbapi=sqlite3, sqlcipher_key_hex=None)
    assert failed is False
    assert len(rows) == 5


class _AllIndexCorruptConn:
    """Every index walk raises; only the plain-scan fallback yields rows."""

    def __init__(self) -> None:
        self.closed = False

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> object:
        if "INDEXED BY" in sql:
            raise sqlite3.DatabaseError("index btree corrupt")
        if "WHERE rowid=?" in sql:
            assert params is not None
            return _FakeResult({"id": f"id-{params[0]}", "rowid": params[0]})
        if "SELECT rowid FROM memories" in sql:  # the no-index direct rowid scan
            return iter([(7,), (9,)])
        if sql == "SELECT * FROM memories":  # last-ditch plain scan
            return _FakeCursor([])
        raise sqlite3.DatabaseError("unexpected query")

    def close(self) -> None:
        self.closed = True


def test_robust_salvage_falls_back_to_plain_rowid_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every secondary index is unusable, salvage still recovers via a
    direct rowid scan (the index-only-corruption / healthy-table case)."""
    fake = _AllIndexCorruptConn()
    monkeypatch.setattr(_recovery, "_connection_connect", lambda *a, **k: fake)

    failed, rows = _recovery._attempt_primary_salvage(
        Path("/nonexistent/backup.db"), dbapi=sqlite3, sqlcipher_key_hex=None
    )

    assert failed is False
    assert sorted(r["id"] for r in rows) == ["id-7", "id-9"]
    assert fake.closed is True


class _EmptyConn:
    """No rowids anywhere — salvage must report total failure (failed=True)."""

    def __init__(self) -> None:
        self.closed = False

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> object:
        if "INDEXED BY" in sql:
            return _FakeCursor([None])
        if "SELECT rowid FROM memories" in sql:
            return iter([])
        if sql == "SELECT * FROM memories":
            return _FakeCursor([])
        raise sqlite3.DatabaseError("unexpected query")

    def close(self) -> None:
        self.closed = True


def test_robust_salvage_reports_failure_when_nothing_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    """primary_failed is True only when zero rows can be read by any path."""
    fake = _EmptyConn()
    monkeypatch.setattr(_recovery, "_connection_connect", lambda *a, **k: fake)

    failed, rows = _recovery._attempt_primary_salvage(
        Path("/nonexistent/backup.db"), dbapi=sqlite3, sqlcipher_key_hex=None
    )

    assert failed is True
    assert rows == []
    assert fake.closed is True


def test_robust_salvage_logs_partial_when_pages_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt-page skip emits db_salvage_partial with the failure count."""
    fake = _FakeConn()  # rowid 2 lives on a corrupt page
    monkeypatch.setattr(_recovery, "_connection_connect", lambda *a, **k: fake)

    with structlog.testing.capture_logs() as logs:
        _recovery._attempt_primary_salvage(
            Path("/nonexistent/backup.db"), dbapi=sqlite3, sqlcipher_key_hex=None
        )

    partials = [log for log in logs if log.get("event") == "db_salvage_partial"]
    assert len(partials) == 1
    assert partials[0]["page_failures"] == 1
    assert partials[0]["salvaged"] == 2


# ---------------------------------------------------------------------------
# 4. WAL-checkpoint primitive (mode coercion, busy fallback, fail-open)
# ---------------------------------------------------------------------------


class TestNormalizeMode:
    """normalize_mode is the single source of truth for the unsafe-engine gate."""

    @pytest.mark.parametrize("mode", ["TRUNCATE", "RESTART", "PASSIVE", "FULL"])
    def test_safe_engine_passes_valid_modes_through(self, mode: str) -> None:
        from trw_memory.storage._wal_checkpoint import normalize_mode

        assert normalize_mode(mode, wal_reset_safe=True) == mode

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [("TRUNCATE", "PASSIVE"), ("RESTART", "PASSIVE"), ("PASSIVE", "PASSIVE"), ("FULL", "FULL")],
    )
    def test_unsafe_engine_downgrades_only_resetting_modes(self, mode: str, expected: str) -> None:
        from trw_memory.storage._wal_checkpoint import normalize_mode

        # The invariant: TRUNCATE/RESTART (the WAL-reset modes) downgrade;
        # PASSIVE/FULL (never reset the WAL) pass through unchanged.
        assert normalize_mode(mode, wal_reset_safe=False) == expected

    @pytest.mark.parametrize("evil", ["DROP TABLE memories", "", "truncate; DELETE", "garbage"])
    def test_unknown_mode_collapses_to_passive(self, evil: str) -> None:
        from trw_memory.storage._wal_checkpoint import normalize_mode

        assert normalize_mode(evil, wal_reset_safe=True) == "PASSIVE"

    def test_lowercase_is_normalized(self) -> None:
        from trw_memory.storage._wal_checkpoint import normalize_mode

        assert normalize_mode("truncate", wal_reset_safe=True) == "TRUNCATE"


def test_run_checkpoint_busy_resetting_falls_back_to_passive() -> None:
    """A TRUNCATE that returns busy=1 retries PASSIVE on the SAME callable."""
    from trw_memory.storage._wal_checkpoint import run_checkpoint

    calls: list[str] = []

    def execute(sql: str) -> object:
        calls.append(sql)
        if "TRUNCATE" in sql:
            return (1, 5, 0)  # busy=1
        return (0, 5, 5)  # PASSIVE succeeds

    result = run_checkpoint(execute, "TRUNCATE", wal_reset_safe=True, db_path=":mem:")

    assert result == {"busy": 0, "checkpointed": 5, "mode": "PASSIVE"}
    assert "TRUNCATE" in calls[0]
    assert "PASSIVE" in calls[1]
    assert len(calls) == 2


def test_run_checkpoint_passive_does_not_retry_on_busy() -> None:
    """A non-resetting checkpoint that is busy does NOT issue a second PRAGMA."""
    from trw_memory.storage._wal_checkpoint import run_checkpoint

    calls: list[str] = []

    def execute(sql: str) -> object:
        calls.append(sql)
        return (1, 0, 0)  # busy=1

    result = run_checkpoint(execute, "PASSIVE", wal_reset_safe=True, db_path=":mem:")

    assert result["mode"] == "PASSIVE"
    assert result["busy"] == 1
    assert len(calls) == 1, "PASSIVE must not trigger a fallback checkpoint"


def test_run_checkpoint_fail_open_on_sqlite_error() -> None:
    """Any sqlite3.Error yields mode='error', busy=1, and is logged not raised."""
    from trw_memory.storage._wal_checkpoint import run_checkpoint

    def execute(sql: str) -> object:
        raise sqlite3.OperationalError("disk I/O error")

    with structlog.testing.capture_logs() as logs:
        result = run_checkpoint(execute, "TRUNCATE", wal_reset_safe=True, db_path="/x.db")

    assert result == {"busy": 1, "checkpointed": 0, "mode": "error"}
    failures = [log for log in logs if log.get("event") == "wal_checkpoint_failed"]
    assert len(failures) == 1


def test_run_checkpoint_missing_row_treated_as_busy() -> None:
    """A None result row (no row returned) is treated as busy, not crash."""
    from trw_memory.storage._wal_checkpoint import run_checkpoint

    result = run_checkpoint(lambda sql: None, "PASSIVE", wal_reset_safe=True, db_path=":mem:")

    assert result["busy"] == 1
    assert result["checkpointed"] == 0


def test_checkpoint_result_is_a_dict_with_public_export() -> None:
    """CheckpointResult is re-exported from trw_memory.storage and is a dict
    at runtime (so .get()-based consumers like trw-mcp keep working)."""
    from trw_memory.storage import CheckpointMode, CheckpointResult  # public import surface

    result: CheckpointResult = {"busy": 0, "checkpointed": 3, "mode": "TRUNCATE"}
    assert isinstance(result, dict)
    assert result.get("mode") == "TRUNCATE"
    # CheckpointMode is the Literal alias consumers use for the mode field.
    assert CheckpointMode is not None
