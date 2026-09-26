"""Persistent per-store graph-enrichment workers (PRD-FIX-143).

Background graph enrichment used to start one thread AND open one fresh
``StorageBackend`` per scheduled write. Every open runs ``open_and_configure``'s
unconditional ``PRAGMA quick_check``, whose cost grows with the file, so a burst
of single-row stores paid one full integrity scan per row.

Here each store location (the SQLite file, or the YAML entries directory) gets
ONE :class:`_GraphWorker`: one daemon thread, one job queue, and one backend
that the worker opens lazily ON ITS OWN THREAD through the unchanged
``create_backend_from_config`` path, uses only there, and closes there. The
integrity check therefore still runs on every open, exactly as before; it just
runs once per worker lifetime instead of once per row, the same frequency the
caller's own long-lived connection has always had.

Lifecycle:

* A worker whose queue has been empty for ``_GRAPH_WORKER_IDLE_EVICT_SECONDS``
  closes its backend and leaves the registry; the next job opens a fresh one.
* Creating a worker past ``_GRAPH_WORKER_MAX_COUNT`` first retires the
  least-recently-active IDLE worker. A busy worker is never retired, so the
  cap is soft: it is exceeded rather than blocking a caller or dropping a job.
* A job whose store file was replaced since the open (a different inode, or a
  missing file) reopens the backend instead of writing edges into the old file.
* A forked child never reuses the parent's threads or connections: the pid is
  checked on every submit and the registry is rebuilt empty in the child. The
  inherited backends are kept referenced, never closed, because closing a
  connection a child inherited can checkpoint or unlink the parent's WAL.

Owner-scoped draining is unchanged: every job is tracked on the process
registry in :mod:`trw_memory._graph_threads` with the scheduling backend as its
owner, so ``wait_for_graph_updates(owner=...)`` waits for exactly that owner's
jobs. The tracked handle is a :class:`_GraphJob`, which answers the registry's
``is_alive`` / ``ident`` / ``join`` / ``name`` protocol for work that is queued
rather than running on a thread of its own.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import queue
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import structlog

from trw_memory._graph_threads import _track_graph_thread, _untrack_graph_thread
from trw_memory.exceptions import StorageError

if TYPE_CHECKING:
    from trw_memory.models.config import MemoryConfig
    from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

_GRAPH_WORKER_IDLE_EVICT_SECONDS = 300.0
"""Seconds a worker's queue must stay empty before it closes its backend and exits."""

_GRAPH_WORKER_MAX_COUNT = 16
"""Soft cap on live workers; past it, creation retires the least-recently-active idle one."""

_IDLE_POLL_SECONDS = 1.0
"""How often an idle worker re-reads the clock to decide whether it has been idle long enough."""

_EXIT_JOIN_SECONDS = 1.0
"""Total time interpreter exit spends letting workers close their backends."""

_clock: Callable[[], float] = time.monotonic

# Recoverable job failures, logged as the per-thread dispatcher always did.
_JOB_ERRORS = (StorageError, sqlite3.Error, ValueError, OSError)

_WorkerKey = tuple[str, str, int]


class _GraphJob:
    """Registry handle for one queued job; completes when the worker has run it."""

    def __init__(self, label: str) -> None:
        self.name = f"trw-memory-graph-{label}"
        self._done = threading.Event()
        self._pid = os.getpid()

    @property
    def ident(self) -> int:
        # Non-None so the registry joins this handle instead of polling it as
        # "registered but not started"; a queued job has already been accepted.
        return 0

    def is_alive(self) -> bool:
        # A job queued before a fork never runs in the child.
        return not self._done.is_set() and os.getpid() == self._pid

    def join(self, timeout: float | None = None) -> None:
        if os.getpid() == self._pid:
            self._done.wait(timeout)

    def finish(self) -> None:
        self._done.set()
        _untrack_graph_thread(cast("threading.Thread", self))


@dataclass
class _Job:
    handle: _GraphJob
    run: Callable[[], None]


_STOP = object()
_CURRENT = threading.local()


def _worker_key(config: MemoryConfig, namespace: str) -> _WorkerKey:
    """Identify the store a backend for (*config*, *namespace*) would open."""
    from trw_memory.integrations._backend import resolve_backend_location

    location = os.path.abspath(resolve_backend_location(config, namespace))
    return (location, config.storage_backend, config.embedding_dim)


def _file_identity(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:  # trw-fail-silent-allow: None = no file; never equals an open identity, so it reopens
        return None
    return (st.st_dev, st.st_ino)


class _GraphWorker:
    """One store's worker: a daemon thread draining a queue against one backend."""

    def __init__(self, pool: _GraphWorkerPool, key: _WorkerKey, config: MemoryConfig, namespace: str) -> None:
        self.key = key
        self._pool = pool
        self._config = config
        self._namespace = namespace
        self.queue: queue.SimpleQueue[_Job | object] = queue.SimpleQueue()
        self.pending = 0  # accepted jobs not yet finished; guarded by the pool lock
        self.last_active = _clock()
        self.backend: StorageBackend | None = None
        self._identity: tuple[int, int] | None = None
        self.thread = threading.Thread(target=self._run, name=f"trw-memory-graph-worker-{key[0]}", daemon=True)

    def owns(self, config: MemoryConfig, namespace: str) -> bool:
        return _worker_key(config, namespace) == self.key

    def open_backend(self) -> StorageBackend:
        """Return this worker's backend, reopening when the store file was replaced."""
        if self.backend is not None and self.key[1] == "sqlite" and _file_identity(self.key[0]) != self._identity:
            logger.info("graph_worker_store_replaced", db_path=self.key[0])
            self._close_backend()
        if self.backend is None:
            from trw_memory.integrations._backend import create_backend_from_config

            self.backend = create_backend_from_config(self._config, self._namespace)
            self._identity = _file_identity(self.key[0])
        return self.backend

    def _close_backend(self) -> None:
        backend, self.backend = self.backend, None
        if backend is None:
            return
        try:
            backend.close()
        except _JOB_ERRORS:
            logger.warning("graph_worker_backend_close_failed", db_path=self.key[0], exc_info=True)

    def _run(self) -> None:
        _CURRENT.worker = self
        try:
            while (job := self._next_job()) is not None:
                self._execute(job)
        finally:
            self._close_backend()
            self._pool.worker_exited(self)

    def _next_job(self) -> _Job | None:
        while True:
            try:
                item = self.queue.get(timeout=_IDLE_POLL_SECONDS)
            except queue.Empty:
                if self._pool.retire_if_idle(self):
                    return None
                continue
            if item is _STOP:
                return None
            return cast("_Job", item)

    def _execute(self, job: _Job) -> None:
        try:
            job.run()
        except _JOB_ERRORS:
            logger.warning("graph_update_background_failed", entry_id=job.handle.name, exc_info=True)
        except Exception:  # trw-fail-silent-allow: one worker serves every later job for its store; an unexpected job error is logged with its traceback and must not kill it
            logger.exception("graph_update_background_crashed", entry_id=job.handle.name)
        finally:
            # Idle in the registry BEFORE the owner's wait returns.
            self._pool.job_finished(self)
            job.handle.finish()


class _GraphWorkerPool:
    """Process-wide map from store key to its :class:`_GraphWorker`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workers: dict[_WorkerKey, _GraphWorker] = {}
        self._live: set[_GraphWorker] = set()
        self._pid = os.getpid()
        # Workers inherited across a fork. Never closed in the child (see module docstring).
        self._abandoned: list[_GraphWorker] = []

    def after_fork_in_child(self) -> None:
        if self._pid == os.getpid():
            return
        self._abandoned.extend(self._live)
        self._lock = threading.Lock()
        self._workers = {}
        self._live = set()
        self._pid = os.getpid()

    def submit(self, config: MemoryConfig, namespace: str, owner: object, label: str, run: Callable[[], None]) -> bool:
        """Queue *run* on the worker for (*config*, *namespace*); False if no worker could start."""
        self.after_fork_in_child()
        key = _worker_key(config, namespace)
        handle = _GraphJob(label)
        _track_graph_thread(cast("threading.Thread", handle), owner=owner)
        try:
            with self._lock:
                worker = self._workers.get(key)
                if worker is None:
                    worker = self._create(key, config, namespace)
                worker.pending += 1
                worker.last_active = _clock()
                # Enqueued under the lock: a worker only retires while holding it
                # with nothing pending, so a job can never land on a retired worker.
                worker.queue.put(_Job(handle, run))
        except RuntimeError:  # trw-fail-silent-allow: thread start refused; logged, and False tells the caller nothing was queued (the pre-pool contract)
            _untrack_graph_thread(cast("threading.Thread", handle))
            logger.warning("graph_update_dispatch_failed", entry_id=label, exc_info=True)
            return False
        return True

    def _create(self, key: _WorkerKey, config: MemoryConfig, namespace: str) -> _GraphWorker:
        if len(self._workers) >= _GRAPH_WORKER_MAX_COUNT:
            idle = [worker for worker in self._workers.values() if worker.pending == 0]
            if idle:
                self._retire(min(idle, key=lambda worker: worker.last_active))
        worker = _GraphWorker(self, key, config, namespace)
        worker.thread.start()  # RuntimeError propagates before the worker is registered
        self._workers[key] = worker
        self._live.add(worker)
        return worker

    def _retire(self, worker: _GraphWorker) -> None:
        """Unregister *worker* and queue a stop behind its pending jobs (pool lock held)."""
        if self._workers.get(worker.key) is worker:
            del self._workers[worker.key]
        worker.queue.put(_STOP)

    def retire_if_idle(self, worker: _GraphWorker) -> bool:
        with self._lock:
            if self._workers.get(worker.key) is not worker:
                return False  # already retired; the queued stop ends the loop
            if worker.pending or _clock() - worker.last_active < _GRAPH_WORKER_IDLE_EVICT_SECONDS:
                return False
            del self._workers[worker.key]
            return True

    def job_finished(self, worker: _GraphWorker) -> None:
        with self._lock:
            worker.pending -= 1
            worker.last_active = _clock()

    def worker_exited(self, worker: _GraphWorker) -> None:
        with self._lock:
            if self._workers.get(worker.key) is worker:
                del self._workers[worker.key]  # the thread died without retiring
            self._live.discard(worker)
        # Jobs still queued behind an abnormal exit would never run: finish their
        # handles so no owner waits on them forever, and say so.
        dropped = 0
        while True:
            try:
                item = worker.queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, _Job):
                dropped += 1
                item.handle.finish()
        if dropped:
            logger.error("graph_worker_exited_with_jobs", db_path=worker.key[0], dropped=dropped)

    def stop_all(self, timeout: float) -> list[_GraphWorker]:
        """Retire every worker and join them; returns the ones still alive at the deadline.

        A busy worker finishes its queued jobs before the stop it is sent.
        """
        with self._lock:
            workers = list(self._live)
            for worker in workers:
                self._retire(worker)
        deadline = time.monotonic() + timeout  # a real deadline, not the idle clock tests may freeze
        for worker in workers:
            worker.thread.join(max(0.0, deadline - time.monotonic()))
        return [worker for worker in workers if worker.thread.is_alive()]


_POOL = _GraphWorkerPool()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_POOL.after_fork_in_child)


def _stop_workers_at_exit() -> None:
    _POOL.stop_all(_EXIT_JOIN_SECONDS)


atexit.register(_stop_workers_at_exit)


def submit_graph_job(config: MemoryConfig, namespace: str, owner: object, label: str, run: Callable[[], None]) -> bool:
    """Queue *run* on the persistent worker for the store (*config*, *namespace*) resolves to."""
    return _POOL.submit(config, namespace, owner, label, run)


@contextlib.contextmanager
def worker_backend(config: MemoryConfig, namespace: str) -> Iterator[StorageBackend]:
    """Yield the backend a graph job should write through.

    On the worker that owns the store this is the worker's persistent backend
    (not closed here). Anywhere else, a one-shot backend is opened and closed.
    """
    worker: _GraphWorker | None = getattr(_CURRENT, "worker", None)
    if worker is not None and worker.owns(config, namespace):
        yield worker.open_backend()
        return
    from trw_memory.integrations._backend import create_backend_from_config

    with create_backend_from_config(config, namespace) as backend:
        yield backend
