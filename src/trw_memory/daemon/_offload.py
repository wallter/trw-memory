"""Bounded off-loop execution for the served tool bodies -- PRD-CORE-279 FR04.

The registered tool functions are ``async def`` that contain no ``await`` over
their work: SQLite reads, BM25, dense search and model inference all ran inline
on the event loop. Measured consequence on the loopback daemon (2026-09-17, 65
facts, 40 recalls): p95 279.9 ms at one caller, 1438.1 ms at five, 3898.6 ms at
twenty, with the wall clock for the whole run flat at 8-10 s. A flat wall clock
under rising concurrency is the signature of a queue, not of work.

This module is the whole fix: one process-wide ``ThreadPoolExecutor``, and a
helper that runs a synchronous callable on it. Three deliberate choices:

**The executor's ``max_workers`` IS the bound.** Requests beyond it queue in the
executor rather than in a second admission mechanism; that queue is unbounded,
exactly as the event loop's was, so nothing about overload behaviour is claimed
to have improved -- only that up to :data:`OFFLOAD_MAX_WORKERS` requests now
make progress at once.

**Each call opens its own backend.** The callables handed here create and close
their storage backend inside the worker, so no SQLite connection crosses a
thread boundary. That is a property of the call sites, restated here because it
is what makes this helper safe to use.

**Context travels with the work.** The callable runs inside a copy of the
caller's ``contextvars`` context, so structlog's bound request context reaches
the worker's log lines and a reused worker does not inherit the previous
request's context.

Cancellation is not offered: a future handed to ``run_in_executor`` cannot stop
a callable that has already started. An MCP client that disconnects mid-recall
leaves the work to finish and be discarded.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, TypeVar

import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

logger = structlog.get_logger(__name__)

__all__ = ["OFFLOAD_MAX_WORKERS", "refuse_offload_submissions", "run_offloaded", "shutdown_offload_pool"]

T = TypeVar("T")

#: How many served tool bodies may run at once. A fixed number rather than a
#: config field: it is the daemon's documented concurrency contract (README,
#: "Loopback daemon"), and a number an operator can reason about beats one that
#: varies with the machine. Four keeps a laptop's cores available to the
#: embedding model while removing the serial ceiling; every worker holds its own
#: SQLite connection, so this is also the bound on concurrent writers.
OFFLOAD_MAX_WORKERS = 4

#: How long shutdown waits for in-flight calls before giving up on them. A
#: BOUND, not a promise: a worker stuck in a model load must not be able to hold
#: the daemon past its shutdown, because the code that runs after this is what
#: withdraws the discovery record and re-delivers the signal.
OFFLOAD_SHUTDOWN_GRACE_SECONDS = 5.0

_EXECUTOR_LOCK = threading.Lock()
#: By worker count: the shared pool, and the one-thread write lane (``daemon._lane``, which
#: dispatches through this same helper at ``workers=1``).
_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_EXECUTOR_PID: int | None = None
#: One-way (PRD-CORE-331 B71-135 e, audit finding F5's other half): set only by
#: :func:`refuse_offload_submissions` (real daemon shutdown), never by
#: :func:`shutdown_offload_pool` (also the per-test pool reset -- see its docstring). Without this,
#: a consolidation body still running when shutdown starts (off-lane read/embed/cluster) could
#: finish after the pool drained and submit its next call -- either its own ``run_offloaded`` or,
#: via the lane's ``_dispatch()``, a cluster write -- which would silently recreate a pool here.
_closed = False


def _executor(workers: int = OFFLOAD_MAX_WORKERS) -> ThreadPoolExecutor:
    """Return the process's executor with *workers* threads, creating it on first use.

    Recreated after a fork: a child inherits the parent's executor object, but
    not the parent's worker threads, so submitting to it would hang forever.

    Raises:
        RuntimeError: After :func:`refuse_offload_submissions` -- the daemon has shut down and no
            call (bare ``run_offloaded`` or a lane job dispatched through this same helper) may
            recreate the pool.
    """
    global _EXECUTOR_PID
    with _EXECUTOR_LOCK:
        if _closed:
            raise RuntimeError("offload pool closed: the daemon has shut down and refuses new work")
        if os.getpid() != _EXECUTOR_PID:
            _EXECUTORS.clear()
            _EXECUTOR_PID = os.getpid()
        if workers not in _EXECUTORS:
            _EXECUTORS[workers] = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix=f"trw-memory-tool-{workers}"
            )
        return _EXECUTORS[workers]


def refuse_offload_submissions() -> None:
    """One-way: every later :func:`_executor` call (a bare ``run_offloaded`` or the write lane's
    own dispatch, ``daemon._lane._dispatch`` -> ``_executor(1)``) raises instead of lazily
    recreating a pool.

    Call this only from the real daemon shutdown path (``_serve.py``), before
    :func:`shutdown_offload_pool` drains the pool -- closing the door first removes the window
    where an already-running body finishes after the drain and would otherwise land on a pool
    silently recreated here.
    """
    global _closed
    with _EXECUTOR_LOCK:
        _closed = True


def shutdown_offload_pool(*, timeout: float = OFFLOAD_SHUTDOWN_GRACE_SECONDS) -> bool:
    """Stop accepting work and drain in-flight calls within *timeout*.

    The daemon calls this BEFORE releasing its discovery record, so a worker
    that finishes in time cannot still be writing to the store after the
    endpoint has been withdrawn. The wait is BOUNDED on purpose: an unbounded
    one would let a single stuck call keep the daemon alive through a SIGTERM,
    which is the failure this PRD exists to remove.

    Args:
        timeout: Seconds to wait for in-flight calls. 0 does not wait.

    Returns:
        True when the pool drained, False when the grace period expired with
        work still running (the caller continues either way).
    """
    global _EXECUTOR_PID
    from trw_memory.daemon._lane import cancel_queued

    cancel_queued()  # the write lane's queue is its own, not the executor's
    with _EXECUTOR_LOCK:
        executors, _EXECUTOR_PID = list(_EXECUTORS.values()), None
        _EXECUTORS.clear()
    # Queued-but-unstarted work is dropped; only started calls are waited for.
    for executor in executors:
        executor.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + max(timeout, 0.0)
    # ``_threads`` is the only handle the stdlib exposes on the running workers,
    # and shutdown(wait=True) has no timeout parameter -- the bound is the point.
    threads = [thread for executor in executors for thread in getattr(executor, "_threads", ()) or ()]
    for thread in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(remaining)
    drained = not any(thread.is_alive() for thread in threads)
    if not drained:
        logger.warning("offload_pool_shutdown_timed_out", grace_seconds=timeout)
    return drained


async def run_offloaded(fn: Callable[..., T], /, *args: object, **kwargs: object) -> T:
    """Run *fn* on the bounded worker pool and return its result.

    Args:
        fn: A synchronous callable. It must own every resource it opens.
        *args: Positional arguments for *fn*.
        **kwargs: Keyword arguments for *fn*.

    Returns:
        Whatever *fn* returns; exceptions propagate unchanged.
    """
    call = functools.partial(contextvars.copy_context().run, fn, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(_executor(), call)
