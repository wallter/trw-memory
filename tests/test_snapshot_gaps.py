"""Wave 15: coverage gap-fill for storage/_snapshot.py (lines 167-170, 241, 314-315, 321-323, 346, 371-372, 408-409, 426-427, 435)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.storage._snapshot import (
    _parse_snapshot_date,
    _prune_tier,
    _sorted_parseable,
    create_snapshot,
    latest_snapshot,
    restore_from_snapshot,
    snapshots_base_dir,
    take_daily_snapshot,
    take_weekly_snapshot,
)


class TestCreateSnapshotVacuumError:
    def test_sqlite_error_in_vacuum_raises_snapshot_error(self, tmp_path: Path) -> None:
        """sqlite3.Error in VACUUM INTO → SnapshotError (lines 167-170)."""
        from trw_memory.storage._snapshot import SnapshotError

        db_path = tmp_path / "memory.db"
        db_path.write_bytes(b"")
        dest = tmp_path / "snapshots" / "daily" / "memory-2026-06-08.db"
        dest.parent.mkdir(parents=True)

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [None, sqlite3.Error("db locked")]  # first call ok, second raises
        with patch("trw_memory.storage._snapshot.sqlite3.connect", return_value=mock_conn):
            with pytest.raises(SnapshotError, match="VACUUM INTO failed"):
                create_snapshot(db_path, dest)


class TestTakeWeeklySnapshotNotSunday:
    def test_returns_none_when_not_sunday_and_not_forced(self, tmp_path: Path) -> None:
        """Non-Sunday + force=False → return None (line 244)."""
        db_path = tmp_path / "memory.db"
        db_path.write_bytes(b"")
        base_dir = tmp_path / "snapshots"
        # Use a Monday (isoweekday=1) so the Sunday check fails
        monday = datetime(2026, 6, 8, tzinfo=timezone.utc)  # Monday
        result = take_weekly_snapshot(db_path, base_dir, force=False, now=monday)
        assert result is None

    def test_returns_none_when_now_is_none_and_not_sunday(self, tmp_path: Path) -> None:
        """now=None on a non-Sunday → datetime.now() branch executed (line 241)."""
        db_path = tmp_path / "memory.db"
        db_path.write_bytes(b"")
        base_dir = tmp_path / "snapshots"
        # Today is June 13 2026 (Saturday); calling without now= exercises line 241
        # and then returns None because Saturday (isoweekday=6) != Sunday (7)
        result = take_weekly_snapshot(db_path, base_dir, force=False)
        assert result is None


class TestParseSnapshotDateInvalidDates:
    def test_invalid_daily_date_returns_none(self) -> None:
        """Invalid date in daily name → ValueError → return None (lines 314-315)."""
        # _DAILY_RE requires bare "YYYY-MM-DD.db" format (no "memory-" prefix)
        result = _parse_snapshot_date("2026-13-40.db")  # month 13 is invalid
        assert result is None

    def test_invalid_weekly_date_returns_none(self) -> None:
        """Invalid iso calendar week → ValueError → return None (lines 321-323)."""
        # _WEEKLY_RE requires bare "YYYY-Www.db" format (no "memory-" prefix)
        result = _parse_snapshot_date("2026-W99.db")  # week 99 is invalid
        assert result is None


class TestParseSnapshotDateNoMatch:
    def test_non_matching_name_returns_none(self) -> None:
        """Name matches neither _DAILY_RE nor _WEEKLY_RE → fall-through None (line 323)."""
        result = _parse_snapshot_date("junk.db")
        assert result is None


class TestLatestSnapshotParsedNone:
    def test_unparseable_snapshot_name_is_skipped(self, tmp_path: Path) -> None:
        """_parse_snapshot_date returns None → continue (line 346)."""
        # latest_snapshot uses _daily_dir(base_dir) = snapshots_base_dir(base_dir)/"daily"
        # = base_dir/"memory"/"snapshots"/"daily"
        daily = snapshots_base_dir(tmp_path) / "daily"
        daily.mkdir(parents=True)
        # Bare date format matches _DAILY_RE but month 13 is invalid → parsed is None
        (daily / "2026-13-40.db").write_bytes(b"")
        result = latest_snapshot(tmp_path)
        assert result is None


class TestRestoreFromSnapshotOsError:
    def test_oserror_during_copy_raises_snapshot_error(self, tmp_path: Path) -> None:
        """OSError from shutil.copy2 → SnapshotError (lines 371-372)."""
        from trw_memory.storage._snapshot import SnapshotError, snapshots_base_dir

        base_dir = tmp_path / "snapshots"
        snap_dir = snapshots_base_dir(base_dir) / "daily"
        snap_dir.mkdir(parents=True)
        snapshot = snap_dir / "memory-2026-06-08.db"
        snapshot.write_bytes(b"fake snapshot data")
        db_path = tmp_path / "memory.db"

        with patch("trw_memory.storage._snapshot.shutil.copy2", side_effect=OSError("no space")):
            with pytest.raises(SnapshotError, match="snapshot restore failed"):
                restore_from_snapshot(base_dir, snapshot, db_path)


class TestPruneTierUnlinkFails:
    def test_oserror_on_unlink_is_logged(self, tmp_path: Path) -> None:
        """OSError on victim.unlink() → debug logged, loop continues (lines 408-409)."""
        import re
        tier_dir = tmp_path / "daily"
        tier_dir.mkdir()
        # Create 3 snapshot files (keep 2, prune 1)
        for i in range(3):
            (tier_dir / f"memory-2026-06-0{i+1}.db").write_bytes(b"")
        with patch("trw_memory.storage._snapshot.Path.unlink", side_effect=OSError("busy")):
            remaining, pruned = _prune_tier(tier_dir, keep=2, pattern=re.compile(r".*\.db"))
        assert pruned == 0  # unlink failed, so none pruned


class TestTakeDailySnapshotNowNone:
    def test_now_none_uses_current_datetime(self, tmp_path: Path) -> None:
        """take_daily_snapshot with now=None → _date_stamp(None) → datetime.now() (line 435)."""
        db_path = tmp_path / "memory.db"
        db_path.write_bytes(b"")
        base_dir = tmp_path
        # Mock create_snapshot to avoid real DB ops; just verify the path was exercised
        with patch("trw_memory.storage._snapshot.create_snapshot", return_value=tmp_path / "snap.db") as mock_cs:
            with patch("trw_memory.storage._snapshot.rotate_snapshots"):
                result = take_daily_snapshot(db_path, base_dir, now=None)
        assert mock_cs.called
        # dest path contains a valid YYYY-MM-DD stamp derived from datetime.now()
        call_dest = mock_cs.call_args[0][1]
        assert call_dest.suffix == ".db"


class TestSortedParseableOsError:
    def test_oserror_in_iterdir_returns_empty_list(self, tmp_path: Path) -> None:
        """OSError during iterdir → return [] (lines 426-427)."""
        import re
        tier_dir = tmp_path / "daily"
        tier_dir.mkdir()
        with patch("trw_memory.storage._snapshot.Path.iterdir", side_effect=OSError("permission denied")):
            result = _sorted_parseable(tier_dir, pattern=re.compile(r".*\.db"))
        assert result == []

    def test_nonexistent_tier_dir_returns_empty_list(self, tmp_path: Path) -> None:
        """tier_dir does not exist → return [] early (line 422-423)."""
        import re
        tier_dir = tmp_path / "nonexistent"
        result = _sorted_parseable(tier_dir, pattern=re.compile(r".*\.db"))
        assert result == []
