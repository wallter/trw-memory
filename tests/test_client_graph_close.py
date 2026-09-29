"""Owner-scoped graph teardown cannot race successful client close."""

import asyncio
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import trw_memory._graph_worker_pool as pool
import trw_memory.graph as graph
from trw_memory._graph_threads import _GraphThreadRegistry
from trw_memory.client import MemoryClient
from trw_memory.exceptions import MemoryConnectionError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

from .conftest import make_entry


def _reset_pool(timeout: float = 5.0) -> None:
    """Test-only: stop every worker thread, mirroring the removed ``_reset_graph_worker_pool_for_tests``."""
    survivors = pool._POOL.stop_all(timeout)
    assert not survivors, f"{len(survivors)} graph worker(s) still running after {timeout}s"


def _live(p: "pool._GraphWorkerPool") -> int:
    with p._lock:
        return sum(1 for worker in p._live if worker.thread.is_alive())


def _open_backends(p: "pool._GraphWorkerPool") -> int:
    with p._lock:
        return sum(1 for worker in p._live if worker.backend is not None)


def _worker_for(p: "pool._GraphWorkerPool", config: MemoryConfig, namespace: str) -> object:
    with p._lock:
        return p._workers.get(pool._worker_key(config, namespace))


def test_registered_but_unstarted_worker_is_not_reported_finished() -> None:
    registry = _GraphThreadRegistry()
    owner = object()
    thread = threading.Thread(target=lambda: None)
    registry.track(thread, owner)
    try:
        with pytest.raises(TimeoutError):
            registry.wait(timeout=0.001, owner=owner)
        registry.wait(timeout=0.001, owner=object())
    finally:
        registry.untrack(thread)


@pytest.mark.parametrize("cancel_close", [False, True])
async def test_close_drains_before_directory_removal(
    client: MemoryClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_close: bool,
) -> None:
    entered, release = threading.Event(), threading.Event()
    actual_worker = graph._run_scheduled_graph_update

    def held_worker(entry: MemoryEntry, config: MemoryConfig, embedding: list[float] | None) -> None:
        entered.set()
        assert release.wait(3)
        actual_worker(entry, config, embedding)

    monkeypatch.setattr(graph, "_run_scheduled_graph_update", held_worker)
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    await client.__aenter__()
    close: asyncio.Task[None] | None = None
    concurrent_close: asyncio.Task[None] | None = None
    try:
        await client.store("A local graph lifecycle regression", tags=["lifecycle"])
        assert await asyncio.to_thread(entered.wait, 1)
        close = asyncio.create_task(client.close())
        await asyncio.sleep(0.03)
        assert not close.done()
        concurrent_close = asyncio.create_task(client.close())
        await asyncio.sleep(0.03)
        assert not concurrent_close.done()
        with pytest.raises(MemoryConnectionError, match="close is incomplete"):
            await client.__aenter__()
        if cancel_close:
            close.cancel()
            await asyncio.sleep(0.03)
            assert not close.done()
        release.set()
        if cancel_close:
            with pytest.raises(asyncio.CancelledError):
                await close
        else:
            await close
        await concurrent_close
        root = tmp_path / "storage"
        shutil.rmtree(root)
        await asyncio.to_thread(graph.wait_for_graph_updates)
        assert not root.exists()
    finally:
        release.set()
        if close is not None and not close.done():
            await asyncio.gather(close, return_exceptions=True)
        if concurrent_close is not None and not concurrent_close.done():
            await asyncio.gather(concurrent_close, return_exceptions=True)
        await client.close()
        await asyncio.to_thread(graph.wait_for_graph_updates)


async def test_close_does_not_wait_for_other_backend(
    client: MemoryClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = threading.Event(), threading.Event()

    def held_worker(entry: MemoryEntry, config: MemoryConfig, embedding: list[float] | None) -> None:
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(graph, "_run_scheduled_graph_update", held_worker)
    await client.__aenter__()
    with SQLiteBackend(tmp_path / "other.sqlite") as other:
        try:
            assert graph.schedule_graph_update(MemoryEntry(id="M-other", content="other"), other)
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.wait_for(client.close(), timeout=0.5)
            assert not release.is_set()
        finally:
            release.set()
            await asyncio.to_thread(graph.wait_for_graph_updates)
            await client.close()


async def test_owner_timeout_abandons_pending_jobs_without_raising(
    client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD-CORE-331 FR08 (B71-53): a timed-out drain must not fail close() -- it abandons the wait.

    Pre-fix (79e84147d), the equivalent test asserted the opposite: both calls below
    raised ``TimeoutError`` and the backend was left half torn-down (``client._backend``
    already ``None`` but ``client._pending_close_backend`` never cleared, so a second
    ``close()`` retried the same doomed wait and raised again).
    """
    entered, release = threading.Event(), threading.Event()
    original_wait = graph.wait_for_graph_updates

    def held_worker(entry: MemoryEntry, config: MemoryConfig, embedding: list[float] | None) -> None:
        entered.set()
        assert release.wait(3)

    def short_wait(*, owner: object) -> None:
        original_wait(timeout=0.01, owner=owner)

    monkeypatch.setattr(graph, "_run_scheduled_graph_update", held_worker)
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    await client.__aenter__()
    try:
        await client.store("timeout lifecycle regression")
        assert await asyncio.to_thread(entered.wait, 1)
        monkeypatch.setattr(graph, "wait_for_graph_updates", short_wait)
        await asyncio.wait_for(client.close(), timeout=1.0)
        with pytest.raises(MemoryConnectionError, match="closed"):
            await client.store("must not reuse a closed backend")
        # A second close() is now a plain no-op: the pending handle was cleared, not
        # left dangling on a backend the first call already tore down.
        await client.close()
    finally:
        release.set()
        await asyncio.to_thread(original_wait)
        await client.close()


async def test_close_bounds_a_large_backlog_and_nothing_in_it_writes_after(
    client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD-CORE-331 FR08 (B71-53): close() bounds the drain even with a large pending queue.

    ``held-1`` holds the worker past close()'s bound; the other 24 rows queue behind
    it and never start. Pre-fix, ``close()`` raised past the 5s default; here the bound
    is monkeypatched down so the test stays fast, but the shape -- one held job plus a
    backlog behind it -- is the FR08 scenario ("a bulk write of ~5k+ rows").
    """
    entered, release = threading.Event(), threading.Event()
    ran: list[str] = []
    original_run = graph._run_scheduled_graph_update
    original_wait = graph.wait_for_graph_updates

    def gate_first(entry: MemoryEntry, config: MemoryConfig, embedding: list[float] | None) -> None:
        if entry.id == "M-held-1":
            entered.set()
            assert release.wait(3)
        ran.append(entry.id)
        original_run(entry, config, embedding)

    def bounded_wait(*, owner: object) -> None:
        original_wait(timeout=0.05, owner=owner)

    monkeypatch.setattr(graph, "_run_scheduled_graph_update", gate_first)
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    await client.__aenter__()
    try:
        await client.store("held entry", entry_id="M-held-1")
        assert await asyncio.to_thread(entered.wait, 1)
        for index in range(24):
            await client.store(f"backlog entry {index}", entry_id=f"M-backlog-{index}")

        monkeypatch.setattr(graph, "wait_for_graph_updates", bounded_wait)
        before = time.monotonic()
        await asyncio.wait_for(client.close(), timeout=1.0)
        elapsed = time.monotonic() - before
        assert elapsed < 1.0, f"close() took {elapsed}s -- it must bound the drain, not hang on the backlog"
    finally:
        release.set()
        await asyncio.to_thread(original_wait)
        await client.close()

    # The backlog never ran: abandoned before any of its jobs started.
    assert ran == ["M-held-1"]


async def test_backend_close_failure_keeps_retry_handle(client: MemoryClient, monkeypatch: pytest.MonkeyPatch) -> None:
    await client.__aenter__()
    backend = client._get_backend()
    original_close = backend.close
    attempts = 0

    def fail_first_close() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected backend close failure")
        original_close()

    monkeypatch.setattr(backend, "close", fail_first_close)
    try:
        with pytest.raises(OSError, match="backend close"):
            await client.close()
        assert client._backend is None
        assert client._pending_close_backend is backend
        await client.close()
        assert client._pending_close_backend is None
        assert attempts == 2
    finally:
        original_close()


# --- PRD-FIX-143: one persistent worker (and one backend open) per store ---------


@pytest.fixture
def fresh_pool() -> Iterator[None]:
    _reset_pool()
    yield
    graph.wait_for_graph_updates(timeout=5.0)
    _reset_pool()


@pytest.fixture
def backend_opens(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, str]]:
    """Every SQLiteBackend construction from here on: (db path, constructing thread name)."""
    opened: list[tuple[Path, str]] = []
    real_init = SQLiteBackend.__init__

    def counting_init(self: SQLiteBackend, db_path: Path, *args: Any, **kwargs: Any) -> None:
        opened.append((Path(db_path), threading.current_thread().name))
        real_init(self, db_path, *args, **kwargs)

    monkeypatch.setattr(SQLiteBackend, "__init__", counting_init)
    return opened


def _store_config(root: Path) -> MemoryConfig:
    return MemoryConfig(storage_backend="sqlite", storage_path=str(root))


def _db_path(config: MemoryConfig) -> Path:
    return Path(config.storage_path) / "default" / config.sqlite_db_name


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


@pytest.mark.usefixtures("fresh_pool")
def test_schedule_graph_update_reuses_one_worker_backend_across_rows(
    tmp_path: Path, backend_opens: list[tuple[Path, str]]
) -> None:
    config = _store_config(tmp_path / "store")
    with create_backend_from_config(config, "default") as owner:
        backend_opens.clear()  # the owner's own open is not a graph-worker open
        for index in range(25):
            entry = make_entry(entry_id=f"M-row-{index}", content=f"graph worker row {index}")
            owner.store(entry)
            assert graph.schedule_graph_update(entry, owner, config=config)
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)
        batch: list[tuple[MemoryEntry, list[float] | None]] = [
            (make_entry(entry_id=f"M-batch-{index}", content=f"batch row {index}"), None) for index in range(3)
        ]
        for entry, _embedding in batch:
            owner.store(entry)
        assert graph.schedule_graph_update_many(batch, owner, config=config)
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)

    # 25 single-row jobs plus one batch job for the same file: ONE open, on the worker thread.
    assert [path for path, _thread in backend_opens] == [_db_path(config)]
    assert backend_opens[0][1].startswith("trw-memory-graph-worker-")
    assert _live(pool._POOL) == 1


@pytest.mark.usefixtures("fresh_pool")
def test_idle_worker_self_evicts_and_next_job_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend_opens: list[tuple[Path, str]]
) -> None:
    now = [1000.0]
    monkeypatch.setattr(pool, "_clock", lambda: now[0])
    monkeypatch.setattr(pool, "_IDLE_POLL_SECONDS", 0.005)
    config = _store_config(tmp_path / "store")
    with create_backend_from_config(config, "default") as owner:
        backend_opens.clear()
        assert graph.schedule_graph_update(make_entry(entry_id="M-idle-1"), owner, config=config)
        graph.wait_for_graph_updates(timeout=5.0, owner=owner)
        worker = _worker_for(pool._POOL, config, "default")
        assert worker is not None
        assert _open_backends(pool._POOL) == 1

        # Idle for just under the window: many polls later it is still registered.
        now[0] += pool._GRAPH_WORKER_IDLE_EVICT_SECONDS - 0.5
        time.sleep(0.1)
        assert _worker_for(pool._POOL, config, "default") is worker
        assert worker.thread.is_alive()

        # Past the window it closes its backend and leaves the registry by itself.
        now[0] += 1.0
        assert _wait_until(lambda: not worker.thread.is_alive())
        assert _worker_for(pool._POOL, config, "default") is None
        assert _live(pool._POOL) == 0
        assert _open_backends(pool._POOL) == 0
        assert len(backend_opens) == 1

        # The next job gets a new worker and a real, fresh open (and quick_check).
        assert graph.schedule_graph_update(make_entry(entry_id="M-idle-2"), owner, config=config)
        graph.wait_for_graph_updates(timeout=5.0, owner=owner)
        replacement = _worker_for(pool._POOL, config, "default")
        assert replacement is not None
        assert replacement is not worker
        assert [path for path, _thread in backend_opens] == [_db_path(config)] * 2


@pytest.mark.usefixtures("fresh_pool")
def test_worker_cap_evicts_only_idle_workers_and_never_drops_a_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pool, "_GRAPH_WORKER_MAX_COUNT", 2)
    gates = {name: threading.Event() for name in ("a", "b", "c", "d")}
    started = {name: threading.Event() for name in gates}
    ran: list[str] = []

    def held_worker(entry: MemoryEntry, config: MemoryConfig, embedding: list[float] | None) -> None:
        started[entry.id].set()
        assert gates[entry.id].wait(5)
        ran.append(entry.id)

    monkeypatch.setattr(graph, "_run_scheduled_graph_update", held_worker)
    owner = MagicMock()
    configs = {name: _store_config(tmp_path / name) for name in gates}

    def schedule(name: str) -> None:
        assert graph.schedule_graph_update(make_entry(entry_id=name), owner, config=configs[name])

    def worker_of(name: str) -> pool._GraphWorker | None:
        return _worker_for(pool._POOL, configs[name], "default")

    schedule("a")  # stays busy
    assert started["a"].wait(5)
    gates["b"].set()
    schedule("b")
    assert _wait_until(lambda: (worker := worker_of("b")) is not None and worker.pending == 0)
    worker_a = worker_of("a")
    assert worker_a is not None

    schedule("c")  # at the cap: retires idle b, never busy a
    assert worker_of("b") is None
    assert worker_of("a") is worker_a
    assert started["c"].wait(5)

    schedule("d")  # a and c are both busy: exceed the cap rather than block or drop
    assert worker_of("a") is worker_a
    assert worker_of("c") is not None
    assert worker_of("d") is not None

    for gate in gates.values():
        gate.set()
    graph.wait_for_graph_updates(timeout=5.0, owner=owner)
    assert sorted(ran) == ["a", "b", "c", "d"]


@pytest.mark.usefixtures("fresh_pool")
def test_abandoning_an_owner_skips_its_jobs_without_reordering_another_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD-CORE-331 FR08 sol r1 P2: abandon must not touch the queue -- FIFO holds for every other owner.

    The first fix drained and requeued the whole worker queue to filter one owner,
    which let a later job overtake a requeued one (sol reproduced b2, b1 from
    b1, b2 while abandoning owner a). This version marks the owner instead, and
    a dequeued job for it is skipped, never requeued -- so owner b's jobs, which
    share the SAME worker/queue as owner a's, keep their submitted order exactly.
    """
    config = _store_config(tmp_path / "store")
    owner_a, owner_b = MagicMock(name="owner-a"), MagicMock(name="owner-b")
    gate = threading.Event()
    started_first = threading.Event()
    ran: list[str] = []

    def held_first_then_record(entry: MemoryEntry, cfg: MemoryConfig, embedding: list[float] | None) -> None:
        if entry.id == "M-a0":
            started_first.set()
            assert gate.wait(5)
        ran.append(entry.id)

    monkeypatch.setattr(graph, "_run_scheduled_graph_update", held_first_then_record)

    # a0 is dequeued first and holds the (single, shared) worker; everything else
    # queues behind it in submission order: a1, b0, a2, b1, b2.
    assert graph.schedule_graph_update(make_entry(entry_id="M-a0"), owner_a, config=config)
    assert started_first.wait(5)
    assert graph.schedule_graph_update(make_entry(entry_id="M-a1"), owner_a, config=config)
    assert graph.schedule_graph_update(make_entry(entry_id="M-b0"), owner_b, config=config)
    assert graph.schedule_graph_update(make_entry(entry_id="M-a2"), owner_a, config=config)
    assert graph.schedule_graph_update(make_entry(entry_id="M-b1"), owner_b, config=config)
    assert graph.schedule_graph_update(make_entry(entry_id="M-b2"), owner_b, config=config)

    # Only a0 is already running (and may still write); the queued a1/a2 are skipped.
    assert graph.abandon_graph_jobs(owner_a) == 1

    gate.set()
    graph.wait_for_graph_updates(timeout=5.0)

    # a0 was already running when abandoned, so it still ran; a1/a2 never did.
    # b's three jobs ran in exactly the order they were submitted -- not reordered
    # around the gap a1/a2 leave behind.
    assert ran == ["M-a0", "M-b0", "M-b1", "M-b2"]


@pytest.mark.usefixtures("fresh_pool")
def test_worker_registry_reset_hook_closes_all_workers(tmp_path: Path) -> None:
    owners = []
    for name in ("one", "two", "three"):
        config = _store_config(tmp_path / name)
        owner = create_backend_from_config(config, "default")
        owners.append(owner)
        assert graph.schedule_graph_update(make_entry(entry_id=f"M-{name}"), owner, config=config)
    graph.wait_for_graph_updates(timeout=10.0)
    assert _live(pool._POOL) == 3
    assert _open_backends(pool._POOL) == 3
    worker_threads = [t for t in threading.enumerate() if t.name.startswith("trw-memory-graph-worker-")]
    assert len(worker_threads) == 3

    _reset_pool()

    assert _live(pool._POOL) == 0
    assert _open_backends(pool._POOL) == 0
    assert not any(thread.is_alive() for thread in worker_threads)
    for owner in owners:
        owner.close()


@pytest.mark.usefixtures("fresh_pool")
def test_concurrent_producers_create_exactly_one_worker_per_path(
    tmp_path: Path, backend_opens: list[tuple[Path, str]]
) -> None:
    config = _store_config(tmp_path / "store")
    producers = 16
    barrier = threading.Barrier(producers)
    with create_backend_from_config(config, "default") as owner:
        backend_opens.clear()

        def produce(index: int) -> bool:
            barrier.wait(5)
            return graph.schedule_graph_update(make_entry(entry_id=f"M-race-{index}"), owner, config=config)

        with ThreadPoolExecutor(max_workers=producers) as executor:
            assert all(executor.map(produce, range(producers)))
        graph.wait_for_graph_updates(timeout=10.0, owner=owner)

    assert [path for path, _thread in backend_opens] == [_db_path(config)]
    assert _live(pool._POOL) == 1
