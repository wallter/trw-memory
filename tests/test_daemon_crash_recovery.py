"""A crashed daemon must not strand every client (2026-09-25 incident).

The operator's daemon aborted in Metal (two threads encoding on the MPS device),
and its parent, the trw-mcp process that auto-started it, never reaped it. The
zombie's pid answered ``kill(pid, 0)``, so every client read the record as live,
failed to connect, and never started a successor. Each link is pinned here: a
zombie is not live, a record naming one is reapable under the claim lock (while a
live process's record still blocks it), an auto-started daemon is reaped when it
exits, and no model runs on Metal on macOS.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.daemon import DaemonPaths
from trw_memory.daemon._discovery import DaemonInfo, DiscoveryAbsent, read_live_discovery
from trw_memory.daemon._paths import write_secret_file
from trw_memory.storage._pid_liveness import _pid_is_live

pytestmark = pytest.mark.skipif(os.name != "posix", reason="zombies and POSIX process groups")


def _zombie() -> subprocess.Popen[bytes]:
    """A child that has exited and is not yet reaped."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        # waitid(WNOWAIT) reports the exit without reaping, so the child stays a zombie.
        if os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            return child
        time.sleep(0.01)
    raise AssertionError("the child did not exit")


def _plant_record(paths: DaemonPaths, pid: int) -> None:
    paths.user_memory_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = DaemonInfo(
        pid=pid, url="http://127.0.0.1:9/mcp", started_at=datetime.now(timezone.utc).isoformat(), version="0"
    )
    write_secret_file(paths.discovery, info.model_dump_json())


def test_a_zombie_pid_is_not_live_and_a_running_one_is(tmp_path: Path) -> None:
    zombie = _zombie()
    running = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert _pid_is_live(zombie.pid, tmp_path / "lock") is False
        assert _pid_is_live(running.pid, tmp_path / "lock") is True
    finally:
        zombie.wait()
        running.kill()
        running.wait()


def test_a_record_naming_a_zombie_is_absent_so_a_successor_may_claim(tmp_path: Path) -> None:
    from trw_memory.daemon._instance import claim_single_instance, release_single_instance

    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    zombie = _zombie()
    try:
        _plant_record(paths, zombie.pid)
        assert isinstance(read_live_discovery(paths), DiscoveryAbsent)
        claim = claim_single_instance(paths, port=0, version="0")
        try:
            assert claim.info.pid == os.getpid()
        finally:
            claim.sock.close()
            release_single_instance(paths, claimed=claim.info)
    finally:
        zombie.wait()


def test_a_record_naming_another_live_process_still_blocks_the_claim(tmp_path: Path) -> None:
    """Liveness is the pid's, never a connect probe's: a draining daemon has closed its
    socket but may still write, and a successor then would be a second writer."""
    from trw_memory.daemon._instance import claim_single_instance
    from trw_memory.exceptions import DaemonAlreadyRunningError

    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _plant_record(paths, other.pid)  # port 9 listens nowhere
        with pytest.raises(DaemonAlreadyRunningError):
            claim_single_instance(paths, port=0, version="0")
    finally:
        other.kill()
        other.wait()


def _abandoned_import(paths: DaemonPaths) -> Path:
    """A private import copy a killed daemon never cleaned up: the copy plus its schema backup."""
    work = paths.user_memory_dir / "import-tmp" / "import-0123"
    (work / "backups").mkdir(parents=True)
    (work / "source.db").write_bytes(b"copy")
    (work / "backups" / "source.db.pre-schema-8.20260925T000000Z").write_bytes(b"backup")
    return work


@pytest.mark.parametrize("previous", ["none", "crashed"])
def test_a_successor_removes_a_crashed_daemons_import_copies(tmp_path: Path, previous: str) -> None:
    """C12-R: a killed import skips its own cleanup; a copy whose owner is not a live process is removed."""
    from trw_memory.daemon._instance import claim_single_instance, release_single_instance

    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    zombie = _zombie()
    try:
        if previous == "crashed":
            _plant_record(paths, zombie.pid)
        work = _abandoned_import(paths)
        claim = claim_single_instance(paths, port=0, version="0")
        claim.sock.close()
        release_single_instance(paths, claimed=claim.info)
    finally:
        zombie.wait()

    assert not work.exists()


def test_a_live_daemons_import_copies_are_left_alone(tmp_path: Path) -> None:
    from trw_memory.daemon._instance import claim_single_instance
    from trw_memory.exceptions import DaemonAlreadyRunningError

    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _plant_record(paths, other.pid)
        work = _abandoned_import(paths)
        with pytest.raises(DaemonAlreadyRunningError):
            claim_single_instance(paths, port=0, version="0")
    finally:
        other.kill()
        other.wait()

    assert (work / "source.db").exists(), "an import the live daemon may still be running keeps its copy"


def test_a_starting_daemon_keeps_a_live_processes_import_copy(tmp_path: Path) -> None:
    """A stdio server serves memory_import_checkout too, with no daemon record: its in-flight copy is not
    an orphan. Only a copy whose owning process is gone is removed when a daemon claims the store."""
    from trw_memory.daemon._instance import claim_single_instance, release_single_instance

    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    live = paths.user_memory_dir / "import-tmp" / f"{os.getpid()}-{'a' * 32}"
    zombie = _zombie()
    try:
        dead = paths.user_memory_dir / "import-tmp" / f"{zombie.pid}-{'b' * 32}"
        for work in (live, dead):
            work.mkdir(parents=True)
            (work / "source.db").write_bytes(b"copy")
        claim = claim_single_instance(paths, port=0, version="0")
        claim.sock.close()
        release_single_instance(paths, claimed=claim.info)
    finally:
        zombie.wait()

    assert (live / "source.db").exists(), "a live importer's copy is still in use"
    assert not dead.exists(), "a dead importer's copy is an orphan"


def test_a_starting_daemon_never_deletes_through_a_symlinked_import_tmp(tmp_path: Path) -> None:
    from trw_memory.daemon._instance import claim_single_instance, release_single_instance

    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    elsewhere = tmp_path / "elsewhere" / "not-an-import"
    elsewhere.mkdir(parents=True)
    paths.user_memory_dir.mkdir(mode=0o700)
    (paths.user_memory_dir / "import-tmp").symlink_to(elsewhere.parent)

    claim = claim_single_instance(paths, port=0, version="0")
    claim.sock.close()
    release_single_instance(paths, claimed=claim.info)

    assert elsewhere.is_dir(), "a directory outside the store is never cleaned as an orphan import"


def test_an_auto_started_daemon_that_exits_is_reaped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.daemon import client as client_mod

    monkeypatch.setattr(client_mod, "_DAEMON_ARGV", ("-c", "import sys; sys.exit(3)"))
    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    paths.user_memory_dir.mkdir(mode=0o700, parents=True)
    spawned = client_mod.start_daemon_detached(paths)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(spawned.pid, 0)  # succeeds for a zombie; fails once the reaper waited
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        raise AssertionError(f"daemon {spawned.pid} exited but was never reaped")
    assert spawned.returncode == 3


@pytest.mark.skipif(sys.platform != "darwin", reason="the Metal abort is macOS-only")
def test_no_model_runs_on_metal_on_macos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from trw_memory.embeddings import local as local_mod
    from trw_memory.retrieval import reranker

    from ._test_hf_cache_support import build_model_cache, install_fake_sentence_transformers, use_fixture_cache

    use_fixture_cache(monkeypatch, tmp_path)
    build_model_cache(tmp_path)
    captured = install_fake_sentence_transformers(monkeypatch)
    assert local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2").available() is True
    assert captured["device"] == "cpu"

    built: dict[str, object] = {}

    class _FakeCrossEncoder:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            built.update(kwargs)

    monkeypatch.setattr(reranker, "_import_cross_encoder", lambda: True)
    monkeypatch.setattr(reranker, "_cross_encoder_cls", _FakeCrossEncoder)
    monkeypatch.setattr(reranker, "_LOADED_MODELS", {})
    assert reranker._get_model("cross-encoder/ms-marco-MiniLM-L-6-v2") is not None
    assert built["device"] == "cpu"
