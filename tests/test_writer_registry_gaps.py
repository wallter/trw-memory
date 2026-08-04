"""Wave 15: coverage gap-fill for storage/_writer_registry.py (lines 120-121, 142-143, 195-196, 220-221, 240-241, 251, 261-262, 294-303, 308-309)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

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


class TestInMemoryBackendDoesNotTouchDisk:
    """An in-memory DB must not create a writer-registry sidecar.

    ``WriterRegistry`` places its directory as a sibling of the DB file
    (``<db_path>.writers/``). ``":memory:"`` has no parent, so ``db_path.parent``
    resolved to ``"."`` and the registry created a literal ``./:memory:.writers/``
    directory in whatever the process's CURRENT WORKING DIRECTORY happened to be.

    It surfaced as an untracked ``trw-memory/:memory:.writers/`` in this repo after
    a test run; in a user's project it litters their cwd the same way. The registry
    is also meaningless there — it counts peer PROCESSES sharing one DB file, and
    an in-memory database is private to its connection.
    """

    def test_in_memory_backend_creates_no_writer_registry_in_cwd(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from trw_memory.models.memory import MemoryEntry
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        monkeypatch.chdir(tmp_path)
        backend = SQLiteBackend(Path(":memory:"))
        backend.store(MemoryEntry(id="M-mem", content="in-memory entry"))

        assert backend.get("M-mem") is not None, "the in-memory backend must still work"
        assert not (tmp_path / ":memory:.writers").exists()
        # Scoped to the registry deliberately. Writing this assertion as "the cwd
        # is empty" surfaced a SECOND, unrelated cwd artifact (`_sec001_anchor`,
        # from security-anchor discovery). That is a real observation and is
        # recorded in the run's plan.md, but it is a different subsystem and
        # folding it in here would make this test fail for a reason it does not
        # describe.
        assert not any(p.name.startswith(":memory:") for p in tmp_path.iterdir())

    def test_on_disk_backend_still_registers(self, tmp_path: Path) -> None:
        """Non-vacuity control: the skip must be scoped to ``:memory:`` only.

        Without this, 'stop creating the sidecar' would pass by disabling the
        writer registry everywhere — the concurrent-writer safety net this package
        added in 0.9.5.
        """
        from trw_memory.models.memory import MemoryEntry
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        db_path = tmp_path / "m.db"
        backend = SQLiteBackend(db_path)
        backend.store(MemoryEntry(id="M-disk", content="on-disk entry"))

        assert (tmp_path / "m.db.writers").is_dir()
