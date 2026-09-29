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

from trw_memory._graph_threads import GRAPH_DRAIN_SECONDS, GRAPH_THREADS
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

_EXIT_JOIN_SECONDS = GRAPH_DRAIN_SECONDS
"""Total time interpreter exit spends letting workers finish and close their backends.

The same bound as close()'s drain: a job still running when close() stopped waiting
(PRD-CORE-331 FR08) must finish before exit kills its daemon thread, or its connection
never closes and memory.db-wal survives uncheckpointed (1.0 s lost it on a loaded host)."""

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
        GRAPH_THREADS.untrack(cast("threading.Thread", self))


@dataclass
class _Job:
    handle: _GraphJob
    run: Callable[[], None]
    owner: object = None


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

    def _is_stale(self) -> bool:
        """Whether the file this worker's backend has open is no longer the file at its path."""
        return self.backend is not None and self.key[1] == "sqlite" and _file_identity(self.key[0]) != self._identity

    def _evict_if_stale(self) -> None:
        """Close a stale backend with no job to trigger a reopen (idle-poll self-heal).

        A worker's SHARED store-lock ``open`` hold on its path outlives the file it
        was opened against: PRD-CORE-306 S3 gave a fresh, empty file at that path its
        own EXCLUSIVE ``migrate`` hold (nothing to protect otherwise, but ``ensure_schema``'s
        bootstrap storm still runs under it), which self-deadlocks against this worker's
        stale SHARED hold until a job reaches :meth:`open_backend` -- and nothing schedules
        that job while the very open it would trigger is what is blocked. Closing here, on
        the same idle-poll that already runs every ``_IDLE_POLL_SECONDS``, releases the
        hold without waiting for a job (regression:
        test_replaced_store_file_is_reopened_not_written_through_the_old_handle).
        """
        if self._is_stale():
            logger.info("graph_worker_store_replaced", db_path=self.key[0])
            self._close_backend()

    def open_backend(self) -> StorageBackend:
        """Return this worker's backend, reopening when the store file was replaced."""
        self._evict_if_stale()
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
                self._evict_if_stale()
                if self._pool.retire_if_idle(self):
                    return None
                continue
            if item is _STOP:
                return None
            return cast("_Job", item)

    def _execute(self, job: _Job) -> None:
        if not self._pool.start_unless_abandoned(job.owner):
            # Skip, not run: the queue keeps FIFO order for every other owner: this
            # item is simply never handed to job.run() (PRD-CORE-331 FR08, sol r1 P2).
            self._pool.job_finished(self, job.owner, started=False)
            job.handle.finish()
            return
        try:
            job.run()
        except _JOB_ERRORS:
            logger.warning("graph_update_background_failed", entry_id=job.handle.name, exc_info=True)
        except Exception:  # trw-fail-silent-allow: one worker serves every later job for its store; an unexpected job error is logged with its traceback and must not kill it
            logger.exception("graph_update_background_crashed", entry_id=job.handle.name)
        finally:
            # Idle in the registry BEFORE the owner's wait returns.
            self._pool.job_finished(self, job.owner, started=True)
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
        # Owners whose queued-but-not-yet-run jobs a worker must skip (PRD-CORE-331 FR08).
        # Paired with ``_owner_pending`` (queued or running) and ``_owner_running``: the mark is
        # added only while jobs are pending and cleared once none are pending or running, so
        # none of the three can grow without bound.
        self._abandoned_owners: set[object] = set()
        self._owner_pending: dict[object, int] = {}
        self._owner_running: dict[object, int] = {}

    def after_fork_in_child(self) -> None:
        if self._pid == os.getpid():
            return
        self._abandoned.extend(self._live)
        self._lock = threading.Lock()
        self._workers = {}
        self._live = set()
        # The parent's pending counts and abandonment marks describe jobs that never run here; an inherited
        # mark would skip the child's own jobs for that owner (B71-133 (d)).
        self._abandoned_owners = set()
        self._owner_pending = {}
        self._owner_running = {}
        self._pid = os.getpid()

    def submit(self, config: MemoryConfig, namespace: str, owner: object, label: str, run: Callable[[], None]) -> bool:
        """Queue *run* on the worker for (*config*, *namespace*); False if no worker could start."""
        self.after_fork_in_child()
        key = _worker_key(config, namespace)
        handle = _GraphJob(label)
        GRAPH_THREADS.track(cast("threading.Thread", handle), owner)
        try:
            with self._lock:
                worker = self._workers.get(key)
                if worker is None:
                    worker = self._create(key, config, namespace)
                worker.pending += 1
                worker.last_active = _clock()
                self._owner_pending[owner] = self._owner_pending.get(owner, 0) + 1
                # Enqueued under the lock: a worker only retires while holding it
                # with nothing pending, so a job can never land on a retired worker.
                worker.queue.put(_Job(handle, run, owner))
        except RuntimeError:  # trw-fail-silent-allow: thread start refused; logged, and False tells the caller nothing was queued (the pre-pool contract)
            GRAPH_THREADS.untrack(cast("threading.Thread", handle))
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

    def start_unless_abandoned(self, owner: object) -> bool:
        """Record one of *owner*'s jobs as started, unless *owner* is abandoned (PRD-CORE-331 FR08 r2).

        The mark check and the start are one transition under the pool lock that
        :meth:`abandon_owner` also takes: an abandon either lands first (the job is
        skipped) or finds the job already counted as running. Checking the mark and
        then starting as two steps let an abandon between them start a job after
        ``close()`` returned that it had not counted. "Started" means committed to
        run: a job counted here may still enter ``job.run()`` after an abandon
        returns, and that abandon's count includes it.
        """
        with self._lock:
            if owner in self._abandoned_owners:
                return False
            self._owner_running[owner] = self._owner_running.get(owner, 0) + 1
            return True

    def job_finished(self, worker: _GraphWorker, owner: object, *, started: bool) -> None:
        with self._lock:
            worker.pending -= 1
            worker.last_active = _clock()
            if started:
                running = self._owner_running.pop(owner, 0) - 1
                if running > 0:
                    self._owner_running[owner] = running
            self._owner_done_locked(owner)

    def _owner_done_locked(self, owner: object) -> None:
        """One less job pending for *owner*; clear its abandon mark once none are pending or running (lock held)."""
        remaining = self._owner_pending.get(owner, 0) - 1
        if remaining <= 0 and owner not in self._owner_running:
            self._owner_pending.pop(owner, None)
            self._abandoned_owners.discard(owner)
        else:
            self._owner_pending[owner] = remaining

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
                with self._lock:
                    self._owner_done_locked(item.owner)
                item.handle.finish()
        if dropped:
            logger.error("graph_worker_exited_with_jobs", db_path=worker.key[0], dropped=dropped)

    def abandon_owner(self, owner: object) -> int:
        """Mark *owner*'s still-pending jobs to be skipped, not run (PRD-CORE-331 FR08).

        No queue surgery: each worker keeps draining its queue in the order jobs
        were submitted, for every owner. A job for *owner* is simply never handed
        to ``job.run()`` once dequeued (see :meth:`start_unless_abandoned`, which
        takes the same lock) -- a job already running when this is called keeps
        going on the worker's own backend (a separate connection to the same store,
        by PRD-FIX-143 design), since a Python thread cannot be preempted mid-write.
        Returns how many of *owner*'s jobs were already running: exactly the ones
        that may still write after this returns.
        """
        with self._lock:
            if self._owner_pending.get(owner, 0):
                self._abandoned_owners.add(owner)
            return self._owner_running.get(owner, 0)

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


def abandon_graph_jobs(owner: object) -> int:
    """Mark *owner*'s pending graph jobs to be skipped, not run; return how many were already running."""
    return _POOL.abandon_owner(owner)


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
