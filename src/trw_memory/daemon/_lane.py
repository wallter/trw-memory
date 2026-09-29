"""The daemon's one write lane, and what each job on it may spend -- PRD-CORE-307 (class module M2).

Every body that changes or removes an existing learning row runs on one thread, one job at a time
(C12 rc4/rc7): a rename's emptiness check and its move, a consolidation cluster's re-read and its
archival, a review's read and its promotion never interleave with a forget. The price is that a long
job holds every tenant's writes, so this module is the only way onto the lane, and every job declares
a :class:`LaneBudget`:

- **Seconds.** Where a job's own loop stops starting new work (the job reads its budget). SQLite
  interrupts any statement a connection opened in the job still runs ``STATEMENT_DEADLINE_FACTOR``
  budgets after the job started (``sqlite3.OperationalError: interrupted``), and a job still running
  then (work outside SQLite: a model load, a file walk) is logged as ``lane_job_overran``.
- **Rows.** The most rows one job reads or writes; long work is a sequence of such jobs
  (:func:`run_slices`) that gives the lane back between them and resumes from a cursor.
- **Class.** An interactive job (forget, update, status) starts before any queued background job
  (maintenance slices, cluster writes, imports); a background job that has waited ``AGING_SECONDS``
  competes as interactive, so a busy daemon still maintains. Within a class the tenant served least
  recently goes first, then the job queued first, so no namespace starves another.

A started body holds the lane until it returns, even when its caller is cancelled; a job cancelled
before it starts never runs. ``tests/test_daemon_lane.py``'s census fails on any lane submission that
does not name a budget from :data:`BUDGETS`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import itertools
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

import structlog

from trw_memory.daemon._offload import _executor
from trw_memory.storage._connection import LANE_DEADLINE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Awaitable, Callable

logger = structlog.get_logger(__name__)

__all__ = [
    "BUDGETS",
    "IMPORT",
    "INTERACTIVE",
    "MAINTENANCE",
    "LaneBudget",
    "cancel_queued",
    "refuse_lane_submissions",
    "run_on_lane",
    "run_slices",
    "submit",
]

T = TypeVar("T")


@dataclass(frozen=True)
class LaneBudget:
    """What one job on the lane may spend before it gives the lane back."""

    name: str
    seconds: float
    rows: int
    background: bool = False


#: A lane job's statements are interrupted this many budgets after it started: the budget is where the
#: job's own loop stops, and the gap lets a statement that began just before that finish.
STATEMENT_DEADLINE_FACTOR = 4.0
#: No statement deadline is shorter than this: the deadline is a backstop, not the slice's clock.
STATEMENT_DEADLINE_FLOOR = 5.0
#: A background job queued this long competes as interactive.
AGING_SECONDS = 5.0
#: What one tool call may spend over all its slices (:func:`run_slices`) before it returns to resume later.
CALL_SECONDS = 60.0

#: One request's body: a forget, an update, a review, a status read.
INTERACTIVE = LaneBudget("interactive", seconds=10.0, rows=10_000)
#: One slice of maintenance (decay, a verify slice, one consolidation cluster's writes, a namespace move batch).
MAINTENANCE = LaneBudget("maintenance", seconds=0.25, rows=1_000, background=True)
#: One checkout import's write step, which already stops at its own deadline (``tools/_checkout_merge``).
IMPORT = LaneBudget("import", seconds=30.0, rows=100_000, background=True)
BUDGETS = (INTERACTIVE, MAINTENANCE, IMPORT)


@dataclass(eq=False)
class _Job:
    budget: LaneBudget
    tenant: str
    call: Callable[[float], object]
    future: concurrent.futures.Future[Any] = field(default_factory=concurrent.futures.Future)
    queued_at: float = field(default_factory=time.monotonic)
    seq: int = field(default_factory=itertools.count().__next__)


#: Tenants with the time their last job started; cleared past this many, so it stays bounded.
_SERVED_MAX = 4096
_lock = threading.Lock()
_queue: list[_Job] = []
_served: dict[str, float] = {}
_running: list[_Job] = []  # zero or one
#: One-way: set only by :func:`refuse_lane_submissions` (real daemon shutdown), never by
#: :func:`cancel_queued` (also used as a per-test backlog reset -- see its docstring).
_shutdown = False


def _after_fork() -> None:  # a child inherits the queue but not the lane's thread
    global _lock, _shutdown
    _lock = threading.Lock()
    _queue.clear()
    _served.clear()
    _running.clear()
    _shutdown = False


os.register_at_fork(after_in_child=_after_fork)


def _rank(job: _Job, now: float) -> tuple[bool, float, int]:
    waiting_class = job.budget.background and now - job.queued_at < AGING_SECONDS
    return (waiting_class, _served.get(job.tenant, 0.0), job.seq)


def _dispatch() -> None:
    """Start the best queued job if the lane is free."""
    with _lock:
        _queue[:] = [job for job in _queue if not job.future.cancelled()]
        if _running or not _queue:
            return
        now = time.monotonic()
        job = min(_queue, key=lambda queued: _rank(queued, now))
        _queue.remove(job)
        _running.append(job)
        if len(_served) >= _SERVED_MAX:
            _served.clear()
        _served[job.tenant] = now
    try:
        _executor(1).submit(_run, job).add_done_callback(lambda ran: _drop(job) if ran.cancelled() else None)
    except RuntimeError:  # the pool shut down between its lookup and this submit
        _drop(job)


def _drop(job: _Job) -> None:
    """The pool shut down before the lane's thread took *job*: it never runs, and the lane is free again."""
    job.future.cancel()
    _release()


def _release() -> None:
    with _lock:
        _running.clear()
    _dispatch()


def _run(job: _Job) -> None:
    try:
        if not job.future.set_running_or_notify_cancel():
            return
        started = time.monotonic()
        allowed = max(job.budget.seconds * STATEMENT_DEADLINE_FACTOR, STATEMENT_DEADLINE_FLOOR)
        try:
            job.future.set_result(job.call(started + allowed))
        except BaseException as exc:  # handed to the caller unchanged, as the executor would
            job.future.set_exception(exc)
        if (elapsed := time.monotonic() - started) > allowed:
            logger.warning("lane_job_overran", budget=job.budget.name, tenant=job.tenant, elapsed=round(elapsed, 3))
    finally:
        _release()


def submit(
    budget: LaneBudget, tenant: str, fn: Callable[..., T], /, *args: Any, **kwargs: Any
) -> concurrent.futures.Future[T]:
    """Queue *fn* as one job on the lane under *budget*, scheduled fairly against *tenant*'s neighbours.

    *fn* runs in a copy of the submitter's context, with the job's statement deadline set in it, and
    owns every resource it opens (its backend and connections). Callable from any thread except the
    lane's own; ``cancel()`` on the returned future succeeds only while the job has not started.
    """
    context = contextvars.copy_context()
    job = _Job(budget, tenant, lambda deadline: context.run(_deadlined, deadline, fn, args, kwargs))
    with _lock:
        if _shutdown:
            job.future.cancel()  # refused: the daemon already withdrew its endpoint (audit finding F5)
            return cast("concurrent.futures.Future[T]", job.future)
        _queue.append(job)
    _dispatch()
    return cast("concurrent.futures.Future[T]", job.future)


def _deadlined(deadline: float, fn: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]) -> T:
    LANE_DEADLINE.set(deadline)
    return fn(*args, **kwargs)


async def run_on_lane(budget: LaneBudget, tenant: str, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """:func:`submit`, awaited; its result or exception comes back unchanged."""
    return await asyncio.wrap_future(submit(budget, tenant, fn, *args, **kwargs))


def cancel_queued() -> None:
    """Drop every job not yet started, and the schedule's memory.

    Does NOT stop future submissions: the lane still serves a ``submit`` call made after this
    returns (the offload module's per-test reset calls this repeatedly in one process, and
    tests rely on the lane staying usable after each call). Pair this with
    :func:`refuse_lane_submissions` at real daemon shutdown to also refuse what comes next.
    """
    with _lock:
        dropped = list(_queue)
        _queue.clear()
        _served.clear()
    for job in dropped:
        job.future.cancel()


def refuse_lane_submissions() -> None:
    """One-way: every :func:`submit` call from here on is refused (cancelled immediately).

    Call this only from the real daemon shutdown path (``_serve.py``, after the offload pool is
    drained), never from a test's per-process reset. Without it, a lane job still in flight when
    the executor pool is torn down (e.g. FR04's per-cluster archive write, submitted after an
    off-lane read/embed/cluster step) can submit its next job after the endpoint was withdrawn;
    the offload module recreates its pool lazily on next use, so that submission would otherwise
    silently land on a fresh pool instead of being refused (audit finding F5).
    """
    global _shutdown
    with _lock:
        _shutdown = True


async def run_slices(
    job: Callable[[Callable[[Any], dict[str, object]]], Awaitable[dict[str, object]]],
    step: Callable[[Any], dict[str, object]],
    unfinished: Callable[[], bool],
    rows: Callable[[], int],
    *,
    max_rows: int,
) -> dict[str, object]:
    """Run *step* as one lane *job* at a time until *unfinished* says done or this call has spent
    ``CALL_SECONDS`` or *max_rows* (``rows()`` so far); the caller resumes from its cursor. A step's
    non-empty reply (a refusal) stops the loop and is returned."""
    ends = time.monotonic() + CALL_SECONDS
    while not (refused := await job(step)):
        if not unfinished() or rows() >= max_rows or time.monotonic() >= ends:
            return {}
    return refused
