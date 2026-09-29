"""A crashed daemon must not strand every client (2026-09-25 incident, PRD-CORE-310).

The operator's daemon aborted in Metal (two threads encoding on the MPS device),
and its parent, the trw-mcp process that auto-started it, never reaped it. The
zombie's pid answered ``kill(pid, 0)``, so every client read the record as live,
failed to connect, and never started a successor. Each link is pinned here: a
zombie is not live, nor is a pid another process reused (FR01), a record naming
either is reapable under the claim lock (while a live daemon's record still
blocks it), an auto-started daemon is nobody's child (FR03), only one site may
start a daemon and it honours the auto-start setting (FR04), and no model runs
on Metal on macOS.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
import structlog

from trw_memory.daemon import DaemonPaths
from trw_memory.daemon import _spawn as spawn_mod
from trw_memory.daemon._discovery import DaemonInfo, DiscoveryAbsent, read_live_discovery, write_discovery
from trw_memory.daemon._paths import write_secret_file
from trw_memory.storage import _pid_liveness
from trw_memory.storage._pid_liveness import _pid_is_live, process_start

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


def _plant_record(paths: DaemonPaths, pid: int, start: str | None = None) -> None:
    paths.user_memory_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = DaemonInfo(
        pid=pid,
        url="http://127.0.0.1:9/mcp",
        started_at=datetime.now(timezone.utc).isoformat(),
        version="0",
        process_start=start,
    )
    write_secret_file(paths.discovery, info.model_dump_json())


@pytest.fixture
def sleeper() -> Iterator[subprocess.Popen[bytes]]:
    """A live process that is not a daemon: what a reused pid names."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield process
    process.kill()
    process.wait()


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


def _start_and_stop(paths: DaemonPaths) -> None:
    from trw_memory.daemon._instance import claim_single_instance, release_single_instance

    claim = claim_single_instance(paths, port=0, version="0")
    claim.sock.close()
    release_single_instance(paths, claimed=claim.info)


def test_a_dead_owners_symlinked_import_entry_is_unlinked_and_its_target_survives(tmp_path: Path) -> None:
    """B71-14: ``rmtree(ignore_errors=True)`` refused a symlinked entry in silence and left it as residue."""
    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.db").write_bytes(b"not an import")
    link = paths.user_memory_dir / "import-tmp" / "import-link"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)

    _start_and_stop(paths)

    assert not link.is_symlink(), "the orphan entry itself is removed"
    assert (outside / "keep.db").read_bytes() == b"not an import", "never what it points at"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the permission that forces the failure")
def test_an_import_copy_that_cannot_be_removed_is_logged(tmp_path: Path) -> None:
    """B71-14: the sweep's removal failures were swallowed; they are one structured warning now."""
    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    work = _abandoned_import(paths)
    (work / "backups").chmod(0o500)
    try:
        with structlog.testing.capture_logs() as events:
            _start_and_stop(paths)
    finally:
        (work / "backups").chmod(0o700)

    assert [e["purpose"] for e in events if e["event"] == "tree_removal_failed" and e["path"] == str(work)]


def test_an_auto_started_daemon_that_exits_leaves_no_zombie(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spawn_mod, "_DAEMON_ARGV", ("-c", "import sys; sys.exit(3)"))
    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    paths.user_memory_dir.mkdir(mode=0o700, parents=True)
    spawned = spawn_mod.start_daemon_detached(paths)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(spawned.pid, 0)  # succeeds for a zombie; fails once its parent (init) reaped it
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        raise AssertionError(f"daemon {spawned.pid} exited but was never reaped")


def test_a_daemon_whose_start_is_unknown_is_never_signalled(tmp_path: Path, sleeper: subprocess.Popen[bytes]) -> None:
    """Without its start, a pid may already name another process: stop() refuses rather than guess."""
    unproven = spawn_mod.SpawnedDaemon(sleeper.pid, None, tmp_path / "lock")

    assert unproven.stop() is False
    assert sleeper.poll() is None, "a process not proven to be the daemon was signalled"


def _parent_of(pid: int) -> int:
    return int(subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout)


def test_an_auto_started_daemon_is_not_the_starters_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PRD-CORE-310 FR03: a process-tree kill of the client cannot reach it, and no client must reap it.

    Pre-change (Popen with ``start_new_session``) its parent was this process.
    """
    monkeypatch.setattr(spawn_mod, "_DAEMON_ARGV", ("-c", "import time; time.sleep(60)"))
    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    paths.user_memory_dir.mkdir(mode=0o700, parents=True)
    spawned = spawn_mod.start_daemon_detached(paths)
    try:
        assert spawned.running()
        assert _parent_of(spawned.pid) != os.getpid()
        assert spawned.start == process_start(spawned.pid) is not None
    finally:
        assert spawned.stop()
    assert not spawned.running()
    assert not spawned.stop(), "a stopped daemon is not stopped twice"


@pytest.mark.skipif(not sys.platform.startswith(("linux", "darwin")), reason="the OS start is read on Linux and macOS")
def test_a_record_naming_a_reused_pid_is_absent_so_a_successor_may_claim(
    tmp_path: Path, sleeper: subprocess.Popen[bytes]
) -> None:
    """PRD-CORE-310 FR01: after a crash or reboot the recorded pid can name an unrelated live process.

    Pre-change the record read as live, so no successor could claim and every call failed.
    """
    from trw_memory.daemon._instance import claim_single_instance, release_single_instance

    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    _plant_record(paths, sleeper.pid, start="another process's start")

    assert isinstance(read_live_discovery(paths), DiscoveryAbsent)
    claim = claim_single_instance(paths, port=0, version="0")
    try:
        assert claim.info.pid == os.getpid()
        assert claim.info.process_start == process_start(os.getpid())
    finally:
        claim.sock.close()
        release_single_instance(paths, claimed=claim.info)


@pytest.mark.parametrize("start", ["own", "none", "unreadable"])
def test_a_live_process_keeps_its_record_unless_its_start_proves_otherwise(
    tmp_path: Path, sleeper: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch, start: str
) -> None:
    """Its own start, a 4.0 record without one, and a start that cannot be read (NFR02) all stay live."""
    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    _plant_record(paths, sleeper.pid, start={"own": process_start(sleeper.pid), "none": None}.get(start, "other"))
    if start == "unreadable":
        monkeypatch.setattr(_pid_liveness, "process_start", lambda _pid: None)

    assert isinstance(read_live_discovery(paths), DaemonInfo)


@pytest.mark.parametrize(("state", "zombie"), [("S", False), ("Z", True)])
def test_the_linux_reader_parses_state_and_start_past_a_command_holding_spaces_and_parens(
    monkeypatch: pytest.MonkeyPatch, state: str, zombie: bool
) -> None:
    """Class E: the Linux branch runs on every Linux host but not on this one, so its parse is pinned here."""
    after_comm = [state, *(str(field) for field in range(4, 22)), "123456", "0"]  # fields 3..23; starttime is 22
    files = {
        "/proc/42/stat": f"42 (trw (memory) d) {' '.join(after_comm)}\n",
        "/proc/sys/kernel/random/boot_id": "boot-a\n",
    }

    class _Proc:
        def __init__(self, path: str) -> None:
            self._path = path

        def read_text(self, **_kwargs: object) -> str:
            if self._path not in files:
                raise FileNotFoundError(self._path)
            return files[self._path]

    monkeypatch.setattr(_pid_liveness.sys, "platform", "linux")
    monkeypatch.setattr(_pid_liveness, "Path", _Proc)

    assert _pid_liveness._is_zombie(42) is zombie
    assert process_start(42) == "linux:boot-a:123456"
    assert process_start(43) is None, "a pid with no /proc entry has no start"


def test_a_published_record_carries_the_daemons_own_start(tmp_path: Path) -> None:
    paths = DaemonPaths(user_memory_dir=tmp_path / "user")
    paths.user_memory_dir.mkdir(mode=0o700, parents=True)
    info = write_discovery(paths, url="http://127.0.0.1:9/mcp", version="0")
    assert info.process_start == process_start(os.getpid())
    if sys.platform.startswith(("linux", "darwin")):
        assert info.process_start is not None


def _calls(package_src: Path, name: str, label: str) -> set[str]:
    """``<label>/src/<path>::enclosing function`` of every call to *name* under *package_src*."""
    sites: set[str] = set()
    for path in package_src.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(scope):
                callee = node.func if isinstance(node, ast.Call) else None
                if getattr(callee, "id", getattr(callee, "attr", None)) == name:
                    sites.add(f"{label}/src/{path.relative_to(package_src).as_posix()}::{scope.name}")
    return sites


def test_census_one_liveness_decision_and_one_spawn_site() -> None:
    """Class before site (PRD-CORE-310 FR01, FR04): a second liveness check or spawn site would skip
    the identity check or the auto-start setting. Keyed by symbol, never by line."""
    # Keyed by package name, not checkout layout: the release check and the public mirror run
    # this from the package tree alone, where trw-mcp is not a sibling to scan.
    package = Path(__file__).resolve().parents[1]
    sources = {"trw-memory": package / "src", "trw-mcp": package.parent / "trw-mcp" / "src"}
    sources = {label: src for label, src in sources.items() if src.is_dir()}
    assert set().union(*(_calls(src, "_pid_is_live", label) for label, src in sources.items())) == {
        "trw-memory/src/trw_memory/storage/_pid_liveness.py::is_process_live",
        # An import copy is owned by a pid named in its directory, not by a daemon record.
        "trw-memory/src/trw_memory/daemon/_instance.py::claim_single_instance",
    }
    assert set().union(*(_calls(src, "start_daemon_detached", label) for label, src in sources.items())) == {
        "trw-memory/src/trw_memory/daemon/client.py::_attach",
    }


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
