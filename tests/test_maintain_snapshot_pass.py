"""`memory_maintain` takes the rolling snapshot itself (UF-MEM-05).

Snapshot rotation (`take_daily_snapshot` / `take_weekly_snapshot`) ran only from a manual CLI, and PRD-INFRA-065's wire-in
never landed, so no store had an automatic restore point. The maintenance pass now takes one daily snapshot per UTC day
(and a weekly one on Sunday), keeps the configured counts, and reports what it did without ever failing maintenance.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._snapshot import snapshots_base_dir
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools import maintain

_MON = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)  # a Monday
_SUN = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)  # a Sunday


@pytest.fixture
def backend(tmp_path: Path):
    store = SQLiteBackend(tmp_path / "memory.db")
    store.store(MemoryEntry(id="M-1", content="a row worth a restore point", namespace="project:default"))
    yield store
    store.close()


def _files(tmp_path: Path, tier: str) -> list[str]:
    folder = snapshots_base_dir(tmp_path) / tier
    return sorted(p.name for p in folder.glob("*.db")) if folder.is_dir() else []


def test_the_pass_takes_a_daily_snapshot_that_holds_the_rows(backend, tmp_path: Path) -> None:
    result = maintain._run_snapshot(backend, MemoryConfig(), now=_MON)

    assert result["status"] == "ok" and result["daily"] == "taken"
    assert _files(tmp_path, "daily") == ["2026-10-05.db"]
    conn = sqlite3.connect(snapshots_base_dir(tmp_path) / "daily" / "2026-10-05.db")
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    finally:
        conn.close()


def test_a_second_run_on_the_same_utc_day_does_not_snapshot_again(backend, tmp_path: Path) -> None:
    maintain._run_snapshot(backend, MemoryConfig(), now=_MON)
    path = snapshots_base_dir(tmp_path) / "daily" / "2026-10-05.db"
    first_mtime = path.stat().st_mtime_ns

    again = maintain._run_snapshot(backend, MemoryConfig(), now=_MON)

    assert again["status"] == "ok" and again["daily"] == "already_taken"
    assert path.stat().st_mtime_ns == first_mtime  # no VACUUM INTO on every maintain call


def test_sunday_also_takes_the_weekly_snapshot_and_other_days_do_not(backend, tmp_path: Path) -> None:
    maintain._run_snapshot(backend, MemoryConfig(), now=_MON)
    assert _files(tmp_path, "weekly") == []
    maintain._run_snapshot(backend, MemoryConfig(), now=_SUN)
    assert _files(tmp_path, "weekly") == ["2026-W40.db"]


def test_the_configured_retention_is_honoured(backend, tmp_path: Path) -> None:
    config = MemoryConfig(memory_snapshot_daily_keep=2)
    for day in (1, 2, 3):
        maintain._run_snapshot(backend, config, now=datetime(2026, 10, day, 12, tzinfo=timezone.utc))

    assert _files(tmp_path, "daily") == ["2026-10-02.db", "2026-10-03.db"]


def test_a_failing_snapshot_is_reported_and_never_raises(backend, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr("trw_memory.storage._snapshot.take_daily_snapshot", boom)

    result = maintain._run_snapshot(backend, MemoryConfig(), now=_MON)

    assert result["status"] == "error" and result["reason"] == "OSError"


def test_a_store_with_no_file_is_skipped_not_snapshotted() -> None:
    class _Memory:
        db_path = ":memory:"

    assert maintain._run_snapshot(_Memory(), MemoryConfig(), now=_MON)["status"] == "skipped"  # type: ignore[arg-type]


def test_a_second_sunday_run_does_not_copy_the_whole_store_again(backend, tmp_path: Path) -> None:
    maintain._run_snapshot(backend, MemoryConfig(), now=_SUN)
    week = snapshots_base_dir(tmp_path) / "weekly" / "2026-W40.db"
    first = week.stat().st_mtime_ns

    again = maintain._run_snapshot(backend, MemoryConfig(), now=_SUN)

    assert again["weekly"] == "not_due"
    assert week.stat().st_mtime_ns == first
