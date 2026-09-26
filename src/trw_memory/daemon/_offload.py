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

__all__ = ["OFFLOAD_MAX_WORKERS", "run_offloaded", "run_serialized", "shutdown_offload_pool"]

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
#: By worker count: the shared pool, and the one-thread lane of :func:`run_serialized`.
_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_EXECUTOR_PID: int | None = None


def _executor(workers: int = OFFLOAD_MAX_WORKERS) -> ThreadPoolExecutor:
    """Return the process's executor with *workers* threads, creating it on first use.

    Recreated after a fork: a child inherits the parent's executor object, but
    not the parent's worker threads, so submitting to it would hang forever.
    """
    global _EXECUTOR_PID
    with _EXECUTOR_LOCK:
        if os.getpid() != _EXECUTOR_PID:
            _EXECUTORS.clear()
            _EXECUTOR_PID = os.getpid()
        if workers not in _EXECUTORS:
            _EXECUTORS[workers] = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix=f"trw-memory-tool-{workers}"
            )
        return _EXECUTORS[workers]


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
    return await _submit(_executor(), fn, *args, **kwargs)


async def _submit(executor: ThreadPoolExecutor, fn: Callable[..., T], /, *args: object, **kwargs: object) -> T:
    call = functools.partial(contextvars.copy_context().run, fn, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(executor, call)


async def run_serialized(fn: Callable[..., T], /, *args: object, **kwargs: object) -> T:
    """:func:`run_offloaded`, but on a one-thread lane of its own, so these bodies run one at a time (C12 rc4).

    Every body that changes or removes an existing learning row runs here -- forget, update and
    correction, consolidate, maintain, rename and merge, review, import -- so none interleaves with
    another: a rename's emptiness check and its move, a consolidation's cluster and its archival
    against a forget (rc7: the rollback re-stored a forgotten row). A cancelled caller cannot free
    the lane: its body finishes first. Bodies that only add rows, vectors or edges, or only read
    (store, recall, similar, vectors, reembed, graph backfill), keep ``run_offloaded``.
    """
    return await _submit(_executor(1), fn, *args, **kwargs)
