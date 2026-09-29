"""PRD-FIX-143 worker-pool edge cases: corruption parity, failures, replaced files, fork."""

from __future__ import annotations

import os
import shutil
import threading
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
from structlog.testing import capture_logs

import trw_memory._graph_worker_pool as pool
import trw_memory.graph as graph
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

from .conftest import make_entry


def _reset_pool(timeout: float = 5.0) -> None:
    """Test-only: stop every worker thread, mirroring the removed ``_reset_graph_worker_pool_for_tests``."""
    survivors = pool._POOL.stop_all(timeout)
    assert not survivors, f"{len(survivors)} graph worker(s) still running after {timeout}s"


def _live(p: pool._GraphWorkerPool) -> int:
    with p._lock:
        return sum(1 for worker in p._live if worker.thread.is_alive())


def _open_backends(p: pool._GraphWorkerPool) -> int:
    with p._lock:
        return sum(1 for worker in p._live if worker.backend is not None)


def _worker_for(p: pool._GraphWorkerPool, config: MemoryConfig, namespace: str) -> object:
    with p._lock:
        return p._workers.get(pool._worker_key(config, namespace))


@pytest.fixture(autouse=True)
def fresh_pool() -> Iterator[None]:
    _reset_pool()
    yield
    graph.wait_for_graph_updates(timeout=5.0)
    _reset_pool()


def _config(root: Path, backend: str = "sqlite") -> MemoryConfig:
    return MemoryConfig(storage_backend=backend, storage_path=str(root))


def _db_path(config: MemoryConfig) -> Path:
    return Path(config.storage_path) / "default" / config.sqlite_db_name


def _open_outcome(open_backend: object) -> str:
    """What opening a store did: the exception class, or recovered / opened."""
    try:
        with open_backend() as backend:  # type: ignore[operator]
            return "recovered" if backend.recovered else "opened"
    except Exception as exc:  # the outcome under comparison is the exception class itself
        return type(exc).__name__


def _corrupt_populated_store(config: MemoryConfig) -> Path:
    with create_backend_from_config(config, "default") as backend:
        for index in range(200):
            backend.store(make_entry(entry_id=f"M-{index}", content=f"row {index} " + "x" * 400))
    db = _db_path(config)
    with db.open("r+b") as handle:  # scribble over interior b-tree pages, keep the header
        handle.seek(4096 * 3)
        handle.write(os.urandom(4096 * 4))
    return db


def test_corrupt_store_on_first_worker_open_recovers_exactly_like_a_direct_open(tmp_path: Path) -> None:
    """FR02: the worker's one open runs the same integrity path as SQLiteBackend.__init__."""
    worker_config = _config(tmp_path / "worker")
    direct_config = _config(tmp_path / "direct")
    _corrupt_populated_store(worker_config)
    direct_db = _db_path(direct_config)
    direct_db.parent.mkdir(parents=True)
    shutil.copy(_db_path(worker_config), direct_db)

    direct = _open_outcome(lambda: create_backend_from_config(direct_config, "default"))

    seen: list[str] = []

    def probe(entry: MemoryEntry, config: MemoryConfig, embedding: list[float] | None) -> None:
        seen.append(_open_outcome(lambda: graph._worker_backend(config, entry.namespace)))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(graph, "_run_scheduled_graph_update", probe)
        owner = object()
        assert graph.schedule_graph_update(make_entry(entry_id="M-new"), owner, config=worker_config)  # type: ignore[arg-type]
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)

    assert direct in {"recovered", "CorruptDatabaseUnsalvageableError", "DatabaseError"}
    assert seen == [direct]


def test_backend_open_failure_is_logged_and_does_not_stop_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path / "store")
    calls = 0
    real_init = SQLiteBackend.__init__

    def failing_first_init(self: SQLiteBackend, *args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected open failure")
        real_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(SQLiteBackend, "__init__", failing_first_init)
    owner = object()
    with capture_logs() as logs:
        assert graph.schedule_graph_update(make_entry(entry_id="M-1"), owner, config=config)  # type: ignore[arg-type]
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)
        worker = _worker_for(pool._POOL, config, "default")
        assert worker is not None
        assert worker.backend is None
        assert graph.schedule_graph_update(make_entry(entry_id="M-2"), owner, config=config)  # type: ignore[arg-type]
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)

    assert [log["event"] for log in logs if log["event"].startswith("graph_update_background")] == [
        "graph_update_background_failed"
    ]
    assert _worker_for(pool._POOL, config, "default") is worker
    assert worker.thread.is_alive()
    assert worker.backend is not None  # the second job's retry opened it
    assert calls == 2


def test_unexpected_job_error_is_logged_and_the_worker_keeps_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[str] = []

    def flaky(entry: MemoryEntry, config: MemoryConfig, embedding: list[float] | None) -> None:
        if entry.id == "M-boom":
            raise RuntimeError("injected job bug")
        ran.append(entry.id)

    monkeypatch.setattr(graph, "_run_scheduled_graph_update", flaky)
    config = _config(tmp_path / "store")
    owner = object()
    with capture_logs() as logs:
        for entry_id in ("M-boom", "M-after"):
            assert graph.schedule_graph_update(make_entry(entry_id=entry_id), owner, config=config)  # type: ignore[arg-type]
        graph.wait_for_graph_updates(timeout=5.0, owner=owner)

    crashed = [log for log in logs if log["event"] == "graph_update_background_crashed"]
    assert len(crashed) == 1
    assert crashed[0]["log_level"] == "error"
    assert ran == ["M-after"]
    worker = _worker_for(pool._POOL, config, "default")
    assert worker is not None
    assert worker.thread.is_alive()


def test_replaced_store_file_is_reopened_not_written_through_the_old_handle(tmp_path: Path) -> None:
    config = _config(tmp_path / "store")
    db = _db_path(config)
    with create_backend_from_config(config, "default") as owner:
        assert graph.schedule_graph_update(make_entry(entry_id="M-1"), owner, config=config)
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)
    worker = _worker_for(pool._POOL, config, "default")
    assert worker is not None
    first_backend = worker.backend

    # Move the old file aside (its inode stays allocated) and create a new store in its place.
    db.rename(db.with_name("old.sqlite"))
    for suffix in ("-wal", "-shm"):
        db.with_name(db.name + suffix).unlink(missing_ok=True)
    with create_backend_from_config(config, "default") as owner:
        assert graph.schedule_graph_update(make_entry(entry_id="M-2"), owner, config=config)
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)

    assert _worker_for(pool._POOL, config, "default") is worker
    assert worker.backend is not None
    assert worker.backend is not first_backend


def test_yaml_store_gets_one_worker_keyed_on_its_entries_directory(tmp_path: Path) -> None:
    config = _config(tmp_path / "yaml", backend="yaml")
    with create_backend_from_config(config, "default") as owner:
        for index in range(3):
            entry = make_entry(entry_id=f"M-y{index}")
            owner.store(entry)
            assert graph.schedule_graph_update(entry, owner, config=config)
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)
    assert _live(pool._POOL) == 1
    worker = _worker_for(pool._POOL, config, "default")
    assert worker is not None
    assert worker.key[0] == str(tmp_path / "yaml" / "default" / "entries")


def test_direct_call_off_a_worker_opens_and_closes_a_one_shot_backend(tmp_path: Path) -> None:
    config = _config(tmp_path / "store")
    with graph._worker_backend(config, "default") as backend:
        assert isinstance(backend, SQLiteBackend)
    assert _live(pool._POOL) == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
def test_forked_child_builds_its_own_worker_and_never_touches_the_parents(tmp_path: Path) -> None:
    config = _config(tmp_path / "store")
    with create_backend_from_config(config, "default") as owner:
        assert graph.schedule_graph_update(make_entry(entry_id="M-parent"), owner, config=config)
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)
        parent_worker = _worker_for(pool._POOL, config, "default")
        assert parent_worker is not None
        parent_backend = parent_worker.backend
        assert parent_backend is not None

        read_fd, write_fd = os.pipe()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # "multi-threaded, fork may deadlock"
            pid = os.fork()
        if pid == 0:  # child: report and _exit, never return into pytest
            code = 1
            try:
                inherited = _live(pool._POOL)
                ok = graph.schedule_graph_update(make_entry(entry_id="M-child"), owner, config=config)
                graph.wait_for_graph_updates(timeout=10.0)
                child_worker = _worker_for(pool._POOL, config, "default")
                fresh = child_worker is not None and child_worker is not parent_worker
                fresh = fresh and child_worker.backend is not None and child_worker.backend is not parent_backend
                os.write(write_fd, f"{inherited} {ok} {fresh}".encode())
                code = 0
            finally:
                os._exit(code)
        os.close(write_fd)
        _pid, status = os.waitpid(pid, 0)
        report = os.read(read_fd, 256).decode()
        os.close(read_fd)

        assert os.waitstatus_to_exitcode(status) == 0
        assert report == "0 True True"
        # The parent's worker and connection are untouched by the child's exit.
        assert _worker_for(pool._POOL, config, "default") is parent_worker
        assert parent_worker.backend is parent_backend
        assert parent_backend.count() == 0
        assert graph.schedule_graph_update(make_entry(entry_id="M-parent-2"), owner, config=config)
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)
        assert parent_worker.thread.is_alive()
        assert parent_worker.backend is parent_backend


def test_a_forked_childs_graph_pool_runs_jobs_for_an_owner_the_parent_abandoned(tmp_path: Path) -> None:
    """B71-133 (d): the child's pool reset cleared the workers but kept the parent's abandonment marks
    (PRD-CORE-331 FR08), so the child's first jobs for that owner were silently skipped."""
    graph_pool = pool._GraphWorkerPool()
    config = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
    owner, running, release, ran = object(), threading.Event(), threading.Event(), threading.Event()

    def blocked() -> None:
        running.set()
        release.wait(10)

    try:
        assert graph_pool.submit(config, "default", owner, "busy", blocked)
        assert graph_pool.submit(config, "default", owner, "queued", lambda: None)
        assert running.wait(10)
        assert graph_pool.abandon_owner(owner) == 1  # the parent gives up on this owner's queued job; one is running

        graph_pool._pid = -1  # what the child sees: another pid, so the fork reset runs
        graph_pool.after_fork_in_child()
        assert graph_pool.submit(config, "default", owner, "child", ran.set)
        assert ran.wait(10), "the child's own job for the owner was skipped by the parent's abandonment mark"
    finally:
        release.set()
        for inherited in graph_pool._abandoned:  # this test's "parent" worker: stop it too
            inherited.queue.put(pool._STOP)
            inherited.thread.join(10)
        graph_pool.stop_all(10)


def _pause_at_start_decision(
    graph_pool: pool._GraphWorkerPool, reached: threading.Event, resume: threading.Event
) -> None:
    """Pause the worker at the last point before it commits to running a job.

    With the fix, the mark check and the start are one locked transition
    (``start_unless_abandoned``), so the only window is before it. Before the fix
    (``is_abandoned`` then ``job.run()``), the window was after the check returned False.
    """
    decide = getattr(graph_pool, "start_unless_abandoned", None)
    if decide is not None:

        def paused_start(owner: object) -> bool:
            reached.set()
            assert resume.wait(10)
            return bool(decide(owner))

        graph_pool.start_unless_abandoned = paused_start  # type: ignore[method-assign]
        return
    check = graph_pool.is_abandoned  # type: ignore[attr-defined]  # pre-fix API

    def paused_check(owner: object) -> bool:
        abandoned = bool(check(owner))
        reached.set()
        assert resume.wait(10)
        return abandoned

    graph_pool.is_abandoned = paused_check  # type: ignore[attr-defined]


def test_an_abandon_that_lands_during_the_start_decision_means_the_job_never_runs(tmp_path: Path) -> None:
    """PRD-CORE-331 FR08 sol r2: an abandon between the worker's mark check and ``job.run()`` let the job start.

    The interleave is forced: the worker pauses at its start decision, the owner is
    abandoned, the worker resumes. The job must be skipped, and the abandon must
    report none of the owner's jobs as running.
    """
    graph_pool = pool._GraphWorkerPool()
    config = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
    owner, reached, resume, ran = object(), threading.Event(), threading.Event(), threading.Event()
    _pause_at_start_decision(graph_pool, reached, resume)
    try:
        assert graph_pool.submit(config, "default", owner, "raced", ran.set)
        assert reached.wait(10)
        running = graph_pool.abandon_owner(owner)
        resume.set()
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)
        assert not ran.is_set(), "job.run started after abandon_owner returned"
        assert running == 0
        with graph_pool._lock:  # the mark self-clears once nothing is pending or running
            assert owner not in graph_pool._abandoned_owners
            assert owner not in graph_pool._owner_pending
            assert owner not in graph_pool._owner_running
    finally:
        resume.set()
        graph_pool.stop_all(10)


def test_an_abandon_between_the_start_and_the_callable_counts_the_job_and_lets_it_run(tmp_path: Path) -> None:
    """PRD-CORE-331 FR08 r2 (sol r1 P2): the window after the locked start, before ``job.run()`` is entered.

    The job is already committed to run there, so the abandon must count it (1),
    and it then runs: no job enters its callable that the abandon did not count.
    """
    graph_pool = pool._GraphWorkerPool()
    config = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
    owner, reached, resume, ran = object(), threading.Event(), threading.Event(), threading.Event()
    decide = graph_pool.start_unless_abandoned

    def paused_after_start(owner_: object) -> bool:
        started = decide(owner_)
        reached.set()
        assert resume.wait(10)
        return started

    graph_pool.start_unless_abandoned = paused_after_start  # type: ignore[method-assign]
    try:
        assert graph_pool.submit(config, "default", owner, "committed", ran.set)
        assert reached.wait(10)
        assert graph_pool.abandon_owner(owner) == 1, "a job committed to run must be in the abandon's count"
        resume.set()
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)
        assert ran.is_set()
        with graph_pool._lock:
            assert owner not in graph_pool._abandoned_owners
            assert owner not in graph_pool._owner_running
    finally:
        resume.set()
        graph_pool.stop_all(10)


def test_an_abandon_after_the_start_counts_the_job_as_running_and_lets_it_finish(tmp_path: Path) -> None:
    """The other side of the one transition: a job already started is reported running and finishes."""
    graph_pool = pool._GraphWorkerPool()
    config = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
    owner, started, release, finished = object(), threading.Event(), threading.Event(), threading.Event()

    def blocked() -> None:
        started.set()
        assert release.wait(10)
        finished.set()

    try:
        assert graph_pool.submit(config, "default", owner, "running", blocked)
        assert started.wait(10)
        assert graph_pool.abandon_owner(owner) == 1
        with graph_pool._lock:  # marked while it runs
            assert owner in graph_pool._abandoned_owners
        release.set()
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)
        assert finished.is_set()
        with graph_pool._lock:
            assert owner not in graph_pool._abandoned_owners
            assert owner not in graph_pool._owner_running
    finally:
        release.set()
        graph_pool.stop_all(10)


def test_interpreter_exit_lets_an_in_flight_job_finish_and_close_its_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job still running at exit (close() stopped waiting for it, PRD-CORE-331 FR08) finishes and
    closes its backend, so SQLite checkpoints memory.db-wal instead of leaving it behind.

    Fails on the former 1 s exit join: a 1.5 s job -- an ordinary enrichment on a loaded host -- was
    killed with its daemon thread and its connection never closed (a 2.2 MB memory.db-wal survived).
    """
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow(entry: MemoryEntry, config: MemoryConfig, embedding: list[float] | None) -> None:
        with graph._worker_backend(config, entry.namespace) as backend:
            backend.store(entry)  # the worker's own connection has written, so it holds the WAL open
            started.set()
            release.wait(30.0)
        finished.set()

    monkeypatch.setattr(graph, "_run_scheduled_graph_update", slow)
    config = _config(tmp_path / "store")
    assert graph.schedule_graph_update(make_entry(entry_id="M-slow"), object(), config=config)  # type: ignore[arg-type]
    assert started.wait(5.0)
    # The job ends 1.5 s after exit starts joining -- measured from the join, not from the job's
    # start, so the former 1.0 s join fails however slowly the host reached this line.
    timer = threading.Timer(1.5, release.set)
    timer.start()
    try:
        pool._stop_workers_at_exit()
        # Read BEFORE the cleanup below releases a still-blocked job, which could then finish on its own.
        finished_by_exit, live_after_exit = finished.is_set(), _live(pool._POOL)
        wal_after_exit = _db_path(config).with_name(config.sqlite_db_name + "-wal").exists()
    finally:
        timer.cancel()
        release.set()

    assert finished_by_exit
    assert live_after_exit == 0
    assert not wal_after_exit
