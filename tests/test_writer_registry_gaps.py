"""Wave 15: coverage gap-fill for storage/_writer_registry.py (lines 120-121, 142-143, 195-196, 220-221, 240-241, 251, 261-262, 294-303, 308-309)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from trw_memory.storage._writer_registry import WriterRegistry


class TestRegisterPruneFails:
    def test_oserror_in_prune_is_logged_and_ignored(self, tmp_path: Path) -> None:
        """OSError during _prune_stale_peers → debug logged, register continues (lines 120-121)."""
        reg = WriterRegistry(tmp_path / "test.db")
        with patch.object(WriterRegistry, "_prune_stale_peers", side_effect=OSError("no perms")):
            reg.register()  # should not raise
        assert reg.concurrent_writers == 1
        reg.close()

    def test_file_exists_error_triggers_utime_path(self, tmp_path: Path) -> None:
        """FileExistsError from O_EXCL → contextlib.suppress utime path (lines 142-143)."""
        reg = WriterRegistry(tmp_path / "test.db")
        # Create the lock file first so O_EXCL raises FileExistsError
        reg._registry_dir.mkdir(parents=True, exist_ok=True)
        reg._lock_file.write_text("99999\n")
        before = reg._lock_file.stat().st_mtime_ns
        reg.register()  # should hit FileExistsError path, touch mtime
        assert reg._lock_file.stat().st_mtime_ns >= before
        reg.close()

    def test_oserror_on_lock_create_is_logged_and_ignored(self, tmp_path: Path) -> None:
        """Non-FileExistsError OSError from O_EXCL open → logged (lines 144-149)."""
        reg = WriterRegistry(tmp_path / "test.db")
        with patch("trw_memory.storage._writer_registry.os.open", side_effect=PermissionError("permission denied")):
            reg.register()  # should not raise even if lock creation fails
        assert reg.concurrent_writers == 0
        reg.close()


class TestUnregisterSilentOsError:
    def test_oserror_on_unlink_is_logged_and_ignored(self, tmp_path: Path) -> None:
        """OSError on lock_file.unlink → debug logged, close continues (lines 195-196)."""
        reg = WriterRegistry(tmp_path / "test.db")
        reg.register()
        with patch("trw_memory.storage._writer_registry.Path.unlink", side_effect=OSError("busy")):
            reg.close()  # should not raise
        assert reg._lock_file.exists()


class TestEnumerateLivePeersGlobFails:
    def test_oserror_in_glob_returns_empty_list(self, tmp_path: Path) -> None:
        """OSError during glob → return [] (lines 220-221)."""
        reg = WriterRegistry(tmp_path / "test.db")
        with patch("trw_memory.storage._writer_registry.Path.glob", side_effect=OSError("no perms")):
            result = reg._enumerate_live_peers()
        assert result == []


class TestPruneStaleGlobFails:
    def test_oserror_in_glob_returns_early(self, tmp_path: Path) -> None:
        """OSError during glob in _prune_stale_peers → return early (lines 240-241)."""
        reg = WriterRegistry(tmp_path / "test.db")
        with patch("trw_memory.storage._writer_registry.Path.glob", side_effect=OSError("no perms")):
            reg._prune_stale_peers()  # should not raise
        assert reg.peer_pids == []

    def test_unparseable_lockfile_name_is_skipped(self, tmp_path: Path) -> None:
        """Lock file with unparseable name → continue (line 251)."""
        reg = WriterRegistry(tmp_path / "test.db")
        reg._registry_dir.mkdir(parents=True, exist_ok=True)
        bad_lock = reg._registry_dir / "notanumber.lock"
        bad_lock.write_text("")
        reg._prune_stale_peers()  # should not raise; skips unparseable file
        assert bad_lock.exists()

    def test_oserror_on_stale_unlink_is_logged(self, tmp_path: Path) -> None:
        """OSError on f.unlink() for stale peer → logged (lines 261-262)."""
        reg = WriterRegistry(tmp_path / "test.db")
        reg._registry_dir.mkdir(parents=True, exist_ok=True)
        dead_lock = reg._registry_dir / "99999.lock"
        dead_lock.write_text("")
        with (
            patch("trw_memory.storage._writer_registry._pid_is_live", return_value=False),
            patch("trw_memory.storage._writer_registry.Path.glob", return_value=[dead_lock]),
            patch("trw_memory.storage._writer_registry.Path.unlink", side_effect=OSError("busy")),
        ):
            reg._prune_stale_peers()  # should not raise
        assert dead_lock.exists()
