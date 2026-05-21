"""WAL-reset corruption hardening (2026-05-20 memory.db corruption incident).

Covers the three structural fixes:
1. Boot-time warning when the active SQLite engine predates the 3.51.3
   WAL-reset bug fix (is_wal_reset_safe() is False).
2. checkpoint_wal serializes on the single owning connection (the fix that
   removes the two-connection race that detonates the WAL-reset bug).
3. Robust primary salvage walks rowids via a secondary index and skips
   corrupt leaf pages instead of aborting at the first one (the data-loss
   path that salvaged 0 rows on 2026-05-20).
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
            except BaseException as exc:  # noqa: BLE001 - record any failure
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
