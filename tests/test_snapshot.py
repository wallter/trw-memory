"""Tests for PRD-INFRA-065 snapshot rotation."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.storage._snapshot import (
    SnapshotError,
    create_snapshot,
    list_snapshots,
    restore_from_snapshot,
    rotate_snapshots,
    snapshots_base_dir,
    take_daily_snapshot,
    take_weekly_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(path: Path, rows: int = 3) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS memories (x INTEGER)")
    for i in range(rows):
        conn.execute("INSERT INTO memories VALUES (?)", (i,))
    conn.commit()
    conn.close()


def _row_count(path: Path) -> int:
    conn = sqlite3.connect(str(path))
    count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    conn.close()
    return int(count)


# ---------------------------------------------------------------------------
# create_snapshot — VACUUM INTO
# ---------------------------------------------------------------------------


def test_create_snapshot_copies_rows(tmp_path: Path) -> None:
    src = tmp_path / "memory.db"
    _make_db(src, rows=5)
    dest = tmp_path / "snap" / "snapshot.db"
    out = create_snapshot(src, dest)
    assert out == dest
    assert dest.exists()
    assert _row_count(dest) == 5


def test_create_snapshot_is_independent_of_source_changes(tmp_path: Path) -> None:
    """VACUUM INTO takes a consistent read-point — subsequent writes do not leak."""
    src = tmp_path / "memory.db"
    _make_db(src, rows=5)
    dest = tmp_path / "snap" / "snapshot.db"
    create_snapshot(src, dest)

    # Mutate source after snapshot.
    conn = sqlite3.connect(str(src))
    conn.execute("INSERT INTO memories VALUES (999)")
    conn.commit()
    conn.close()

    # Snapshot row count must be unchanged.
    assert _row_count(dest) == 5
    assert _row_count(src) == 6


def test_create_snapshot_overwrites_existing_dest(tmp_path: Path) -> None:
    """Second call with same dest must succeed (atomic replace)."""
    src = tmp_path / "memory.db"
    _make_db(src, rows=2)
    dest = tmp_path / "snap" / "snapshot.db"
    create_snapshot(src, dest)
    assert _row_count(dest) == 2

    # Write more to source, snapshot again to same dest.
    conn = sqlite3.connect(str(src))
    conn.execute("INSERT INTO memories VALUES (100)")
    conn.commit()
    conn.close()
    create_snapshot(src, dest)
    assert _row_count(dest) == 3


def test_create_snapshot_raises_when_source_missing(tmp_path: Path) -> None:
    missing = tmp_path / "never_created.db"
    dest = tmp_path / "snap" / "snapshot.db"
    with pytest.raises(SnapshotError, match="source database does not exist"):
        create_snapshot(missing, dest)


def test_create_snapshot_cleans_up_tmpfile_on_failure(tmp_path: Path) -> None:
    """If VACUUM INTO fails the tempfile must not linger."""
    src = tmp_path / "memory.db"
    _make_db(src, rows=2)
    # Dest under a read-only parent to force the atomic replace to fail.
    ro_parent = tmp_path / "ro"
    ro_parent.mkdir()
    # VACUUM INTO succeeds to tmp → replace fails if target path is a directory.
    target_dir = ro_parent / "snapshot.db"
    target_dir.mkdir()  # target is a DIRECTORY, replace will fail

    with pytest.raises(SnapshotError):
        create_snapshot(src, target_dir)
    # No stray .tmp file should remain.
    tmp_artifact = target_dir.with_suffix(target_dir.suffix + ".tmp")
    assert not tmp_artifact.exists() or not tmp_artifact.is_file()


# ---------------------------------------------------------------------------
# take_daily_snapshot
# ---------------------------------------------------------------------------


def test_take_daily_writes_ymd_file(tmp_path: Path) -> None:
    base = tmp_path
    src = tmp_path / "memory.db"
    _make_db(src, rows=3)
    now = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
    dest = take_daily_snapshot(base, src, keep_daily=7, now=now)
    assert dest.name == "2026-04-13.db"
    assert dest.parent == snapshots_base_dir(base) / "daily"
    assert _row_count(dest) == 3


def test_take_daily_is_idempotent_same_day(tmp_path: Path) -> None:
    """Same-day call overwrites the same file (last write of day wins)."""
    base = tmp_path
    src = tmp_path / "memory.db"
    _make_db(src, rows=3)
    now = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
    d1 = take_daily_snapshot(base, src, keep_daily=7, now=now)
    # Add a row and re-snapshot same day.
    conn = sqlite3.connect(str(src))
    conn.execute("INSERT INTO memories VALUES (99)")
    conn.commit()
    conn.close()
    d2 = take_daily_snapshot(base, src, keep_daily=7, now=now)
    assert d1 == d2
    assert _row_count(d2) == 4


# ---------------------------------------------------------------------------
# take_weekly_snapshot
# ---------------------------------------------------------------------------


def test_take_weekly_skips_non_sunday_without_force(tmp_path: Path) -> None:
    base = tmp_path
    src = tmp_path / "memory.db"
    _make_db(src, rows=1)
    # 2026-04-13 is a Monday.
    monday = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
    assert monday.isoweekday() == 1
    result = take_weekly_snapshot(base, src, keep_weekly=4, now=monday)
    assert result is None


def test_take_weekly_runs_on_sunday(tmp_path: Path) -> None:
    base = tmp_path
    src = tmp_path / "memory.db"
    _make_db(src, rows=1)
    sunday = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)
    assert sunday.isoweekday() == 7
    result = take_weekly_snapshot(base, src, keep_weekly=4, now=sunday)
    assert result is not None
    # ISO week of 2026-04-12 is week 15.
    assert result.name == "2026-W15.db"
    assert result.parent == snapshots_base_dir(base) / "weekly"


def test_take_weekly_honors_force(tmp_path: Path) -> None:
    base = tmp_path
    src = tmp_path / "memory.db"
    _make_db(src, rows=1)
    monday = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
    result = take_weekly_snapshot(base, src, keep_weekly=4, now=monday, force=True)
    assert result is not None


# ---------------------------------------------------------------------------
# rotate_snapshots — oldest-first pruning
# ---------------------------------------------------------------------------


def test_rotate_prunes_oldest_daily_first(tmp_path: Path) -> None:
    daily_dir = snapshots_base_dir(tmp_path) / "daily"
    daily_dir.mkdir(parents=True)
    # Seed 5 daily files ordered T1 < T2 < T3 < T4 < T5.
    names = [
        "2026-04-09.db",
        "2026-04-10.db",
        "2026-04-11.db",
        "2026-04-12.db",
        "2026-04-13.db",
    ]
    for n in names:
        (daily_dir / n).write_bytes(b"sqlite-fake")

    result = rotate_snapshots(tmp_path, keep_daily=3, keep_weekly=-1)
    assert result.daily_pruned == 2
    assert result.daily_kept == 3
    remaining = sorted(p.name for p in daily_dir.iterdir())
    assert remaining == ["2026-04-11.db", "2026-04-12.db", "2026-04-13.db"]


def test_rotate_prunes_oldest_weekly_first(tmp_path: Path) -> None:
    weekly_dir = snapshots_base_dir(tmp_path) / "weekly"
    weekly_dir.mkdir(parents=True)
    for n in ["2026-W10.db", "2026-W11.db", "2026-W12.db", "2026-W13.db"]:
        (weekly_dir / n).write_bytes(b"x")
    result = rotate_snapshots(tmp_path, keep_daily=-1, keep_weekly=2)
    assert result.weekly_pruned == 2
    remaining = sorted(p.name for p in weekly_dir.iterdir())
    assert remaining == ["2026-W12.db", "2026-W13.db"]


def test_rotate_ignores_unparseable_filenames(tmp_path: Path) -> None:
    """Files whose names don't match the regex MUST be left alone."""
    daily_dir = snapshots_base_dir(tmp_path) / "daily"
    daily_dir.mkdir(parents=True)
    (daily_dir / "2026-04-10.db").write_bytes(b"x")
    (daily_dir / "junk.db").write_bytes(b"x")
    (daily_dir / "2026-04-10.db-wal").write_bytes(b"x")

    result = rotate_snapshots(tmp_path, keep_daily=0, keep_weekly=-1)
    # Only the parseable file was counted and pruned.
    assert result.daily_pruned == 1
    assert (daily_dir / "junk.db").exists()
    assert (daily_dir / "2026-04-10.db-wal").exists()


def test_rotate_keep_zero_prunes_all(tmp_path: Path) -> None:
    daily_dir = snapshots_base_dir(tmp_path) / "daily"
    daily_dir.mkdir(parents=True)
    (daily_dir / "2026-04-10.db").write_bytes(b"x")
    (daily_dir / "2026-04-11.db").write_bytes(b"x")
    result = rotate_snapshots(tmp_path, keep_daily=0, keep_weekly=-1)
    assert result.daily_pruned == 2
    assert result.daily_kept == 0


def test_rotate_noop_when_under_budget(tmp_path: Path) -> None:
    daily_dir = snapshots_base_dir(tmp_path) / "daily"
    daily_dir.mkdir(parents=True)
    (daily_dir / "2026-04-10.db").write_bytes(b"x")
    result = rotate_snapshots(tmp_path, keep_daily=5, keep_weekly=-1)
    assert result.daily_pruned == 0
    assert result.daily_kept == 1


def test_rotate_missing_dir_is_safe(tmp_path: Path) -> None:
    # No snapshot dir created at all.
    result = rotate_snapshots(tmp_path, keep_daily=3, keep_weekly=3)
    assert result.daily_pruned == 0
    assert result.weekly_pruned == 0


# ---------------------------------------------------------------------------
# Automatic rotation on take_daily / take_weekly
# ---------------------------------------------------------------------------


def test_take_daily_prunes_after_create(tmp_path: Path) -> None:
    """Creating the 4th daily with keep_daily=3 removes the oldest."""
    base = tmp_path
    src = tmp_path / "memory.db"
    _make_db(src, rows=1)
    dates = [
        datetime(2026, 4, 10, tzinfo=timezone.utc),
        datetime(2026, 4, 11, tzinfo=timezone.utc),
        datetime(2026, 4, 12, tzinfo=timezone.utc),
        datetime(2026, 4, 13, tzinfo=timezone.utc),
    ]
    for d in dates:
        take_daily_snapshot(base, src, keep_daily=3, now=d)
    remaining = sorted(p.name for p in (snapshots_base_dir(base) / "daily").iterdir())
    assert remaining == ["2026-04-11.db", "2026-04-12.db", "2026-04-13.db"]


# ---------------------------------------------------------------------------
# list_snapshots — newest-first sorting
# ---------------------------------------------------------------------------


def test_list_snapshots_sorted_newest_first(tmp_path: Path) -> None:
    daily_dir = snapshots_base_dir(tmp_path) / "daily"
    weekly_dir = snapshots_base_dir(tmp_path) / "weekly"
    daily_dir.mkdir(parents=True)
    weekly_dir.mkdir(parents=True)
    (daily_dir / "2026-04-10.db").write_bytes(b"x")
    (daily_dir / "2026-04-12.db").write_bytes(b"x")
    (weekly_dir / "2026-W12.db").write_bytes(b"x")
    (weekly_dir / "2026-W15.db").write_bytes(b"x")

    listing = list_snapshots(tmp_path)
    assert [p.name for p in listing["daily"]] == ["2026-04-12.db", "2026-04-10.db"]
    assert [p.name for p in listing["weekly"]] == ["2026-W15.db", "2026-W12.db"]


# ---------------------------------------------------------------------------
# restore_from_snapshot
# ---------------------------------------------------------------------------


def test_restore_copies_snapshot_over_db(tmp_path: Path) -> None:
    base = tmp_path
    src = tmp_path / "memory.db"
    _make_db(src, rows=5)
    snap = take_daily_snapshot(base, src, keep_daily=7, now=datetime(2026, 4, 13, tzinfo=timezone.utc))

    # Simulate DB corruption / data loss.
    src.unlink()
    restore_from_snapshot(base, snap, src)
    assert src.exists()
    assert _row_count(src) == 5


def test_restore_rejects_path_traversal(tmp_path: Path) -> None:
    """Restore MUST refuse snapshot paths outside the snapshots root."""
    base = tmp_path
    outside = tmp_path / "not_a_snapshot.db"
    outside.write_bytes(b"x")
    dest = tmp_path / "memory.db"
    with pytest.raises(SnapshotError, match="Path traversal guard"):
        restore_from_snapshot(base, outside, dest)


def test_restore_raises_when_snapshot_missing(tmp_path: Path) -> None:
    base = tmp_path
    snapshots_base_dir(base).mkdir(parents=True)
    ghost = snapshots_base_dir(base) / "daily" / "2099-01-01.db"
    dest = tmp_path / "memory.db"
    with pytest.raises(SnapshotError, match="snapshot not found"):
        restore_from_snapshot(base, ghost, dest)
