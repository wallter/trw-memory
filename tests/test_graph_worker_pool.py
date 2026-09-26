"""PRD-FIX-143 worker-pool edge cases: corruption parity, failures, replaced files, fork."""

from __future__ import annotations

import os
import shutil
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
