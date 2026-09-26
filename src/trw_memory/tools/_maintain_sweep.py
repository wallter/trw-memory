"""One ``memory_maintain`` run on the daemon's serialized lane, and its resumable verification sweep (rc9).

Belongs to ``tools/maintain.py``, which keeps the passes and the stamp file.

The daemon runs every row-changing body on one lane (``run_serialized``), so a maintain that ran its
passes and its whole verification sweep as one job held every other tenant's writes for as long as the
sweep took. Here each pass is its own job and the sweep runs in slices of about ``SLICE_SECONDS``. One
call stops after about ``BUDGET_SECONDS`` or ``BUDGET_ROWS``, stamps where the sweep stopped (for this
store and root only), and the next call resumes there. A failure in any call of a sweep is carried to
the call that completes it, so ``last_maintained_at`` advances only for a sweep with none.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from trw_memory.lifecycle.verification_pass import MaintainVerifySummary, VerifySettings
from trw_memory.models.config import MemoryConfig
from trw_memory.security.rbac import Permission
from trw_memory.tools import maintain

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

__all__ = ["add_sweep_counts", "serve_maintain", "serve_verify"]

#: One serialized job verifies for about this long (plus one row's own checks), then gives the lane back.
SLICE_SECONDS = 2.0
#: What one daemon ``memory_maintain`` call verifies at most, over all its slices.
BUDGET_SECONDS = 60.0
BUDGET_ROWS = 10_000

#: Namespaces with a daemon maintain or verify in progress: a second one would interleave its slices.
#: Touched only on the event loop, never in a job (see :func:`serve_maintain`).
_RUNNING: set[str] = set()

_Step = Callable[["StorageBackend"], dict[str, object]]
_Job = Callable[[_Step], Awaitable[dict[str, object]]]


def lane_job(namespace: str, operation: str) -> _Job:
    """Run a step as one job on the daemon's serialized lane, authorized again for every job: a
    refusal (a revoked grant) comes back as the step's reply and ends the sweep with it."""
    from trw_memory.daemon._offload import run_serialized
    from trw_memory.tools.entry import in_namespace

    async def job(step: _Step) -> dict[str, object]:
        return await run_serialized(in_namespace, namespace, Permission.WRITE, operation, lambda b, _c: step(b))

    return job


def claim(namespace: str) -> dict[str, object]:
    """Claim *namespace* for one maintain or verify (``{}``), or say it is busy. Call on the event loop only,
    after the caller is authorized, and release in a ``finally`` (:func:`release`): a job outlives
    a cancelled caller, so a claim taken inside one could outlive its release."""
    if namespace in _RUNNING:
        return {"status": "busy", "error": f"a memory_maintain or memory_verify of {namespace} is already running"}
    _RUNNING.add(namespace)
    return {}


def release(namespace: str) -> None:
    """End :func:`claim`'s hold on *namespace* (event loop only)."""
    _RUNNING.discard(namespace)


async def run_slices(
    job: _Job, step: _Step, unfinished: Callable[[], bool], rows: Callable[[], int]
) -> dict[str, object]:
    """Run *step* one lane job at a time until the sweep is finished or this call's daemon-owned
    budget (``BUDGET_SECONDS``, ``BUDGET_ROWS``) is spent; a step's non-empty reply (a refusal)
    stops it and is returned."""
    ends = time.monotonic() + BUDGET_SECONDS
    while not (refused := await job(step)):
        if not unfinished() or rows() >= BUDGET_ROWS or time.monotonic() >= ends:
            return {}
    return refused


def add_sweep_counts(done: dict[str, object], more: dict[str, object]) -> dict[str, object]:
    """Two slices of one sweep as one: *more*'s other keys, the counts added (``root_unwalkable``,
    a flag, is the larger of the two)."""
    merged = {**done, **more}
    for key in MaintainVerifySummary().as_dict():
        first, second = (value if isinstance(value := side.get(key), int) else 0 for side in (done, more))
        merged[key] = max(first, second) if key == "root_unwalkable" else first + second
    return merged


def rows_verified(counts: object) -> int:
    """The rows a sweep's *counts* (a summary, or ``None`` before its first slice) say it verified."""
    found = counts if isinstance(counts, dict) else {}
    return sum(value for key in ("entries_processed", "entry_failures") if isinstance(value := found.get(key), int))


@dataclass
class MaintainRun:
    """One maintain call: where its sweep resumes, a failure an earlier call of that sweep had, and
    what its passes have reported so far."""

    namespace: str
    config: MemoryConfig
    previous: object
    after: tuple[str, str] | None
    carried: str | None
    attempted_at: str = field(default_factory=maintain._now)
    passes: dict[str, object] = field(default_factory=dict)
    knobs: dict[str, Any] = field(default_factory=dict)
    key: str = ""

    def rows_verified(self) -> int:
        return rows_verified(self.passes.get("verification"))


def _identity(backend: StorageBackend) -> list[int] | None:
    """The store file's (device, inode): a store swapped in under the same path is another store."""
    try:
        stat = os.stat(getattr(backend, "db_path", None))  # type: ignore[arg-type]
    except (OSError, TypeError):  # trw-fail-silent-allow: no identity resumes nothing; the sweep starts over
        return None
    return [stat.st_dev, stat.st_ino]


def begin(
    namespace: str, backend: StorageBackend, config: MemoryConfig, knobs: dict[str, Any] | None = None
) -> MaintainRun:
    """Read *namespace*'s stamp: its last completion, and the unfinished sweep to resume. Sweeps are
    kept per root and settings, so no call resumes one checked against other files or thresholds;
    one recorded for another store, or at a position outside *namespace*, starts over."""
    knobs = asdict(VerifySettings()) if knobs is None else knobs
    key = json.dumps([config.project_root, knobs], sort_keys=True)
    stamp = maintain._namespace_stamp(backend, namespace)
    sweeps = stamp.get("verify_sweeps")
    sweep = sweeps.get(key) if isinstance(sweeps, dict) else None
    after, carried = None, None
    if isinstance(sweep, dict) and (identity := _identity(backend)) is not None:
        resume, failed = sweep.get("next"), sweep.get("failed")
        if (
            sweep.get("store") == identity
            and isinstance(resume, list)
            and len(resume) == 2
            and resume[0] == namespace
            and isinstance(resume[1], str)
        ):
            after, carried = (namespace, resume[1]), failed if isinstance(failed, str) else None
    return MaintainRun(namespace, config, stamp.get("last_maintained_at", ""), after, carried, knobs=knobs, key=key)


def verify_slice(
    run: MaintainRun, backend: StorageBackend, seconds: float | None, rows: int | None = None
) -> dict[str, object]:
    """Verify on from where *run*'s sweep stands for about *seconds* and at most *rows* rows
    (``None``: to the end)."""
    more = maintain._run_verification(
        run.namespace, backend, run.config, after=run.after, seconds=seconds, rows=rows, **run.knobs
    )
    done = run.passes.get("verification")
    merged = add_sweep_counts(done, more) if isinstance(done, dict) else dict(more)
    if isinstance(done, dict) and done.get("status") == maintain._ERROR:
        merged["status"], merged["reason"] = maintain._ERROR, done.get("reason", merged.get("reason"))
    if "complete" not in more:  # the slice raised: where the sweep stands is unknown, so it starts over
        merged.pop("complete", None)
        merged.pop("next", None)
    if run.carried and merged.get("status") != maintain._ERROR:
        merged["status"], merged["reason"] = maintain._ERROR, run.carried
    run.passes["verification"] = merged
    resume = merged.get("next")
    run.after = (
        (run.namespace, str(resume[1])) if merged.get("complete") is False and isinstance(resume, list) else None
    )
    return {}


def _sweep_record(run: MaintainRun, backend: StorageBackend) -> dict[str, object] | None:
    """What the stamp keeps of *run*'s unfinished sweep, for the next maintain or verify to resume."""
    verification = run.passes.get("verification")
    if run.after is None or not isinstance(verification, dict):
        return None
    failed = str(verification.get("reason", "error")) if verification.get("status") == maintain._ERROR else None
    return {"next": list(run.after), "store": _identity(backend), "failed": failed}


def _save_sweep(run: MaintainRun, backend: StorageBackend) -> dict[str, object]:
    """Stamp where *run*'s sweep stands, and nothing else: a ``memory_verify`` is no maintenance attempt."""
    maintain._record_stamp(
        backend, run.namespace, attempted_at=None, sweep=_sweep_record(run, backend), sweep_key=run.key
    )
    return {}


def finish(run: MaintainRun, backend: StorageBackend) -> dict[str, object]:
    """Stamp *run* (``last_maintained_at`` only when every pass succeeded and the sweep completed) and reply."""
    passes = run.passes
    succeeded = all(str(p.get("status")) != maintain._ERROR for p in passes.values() if isinstance(p, dict))
    sweep = _sweep_record(run, backend)
    record = maintain._record_stamp(
        backend,
        run.namespace,
        attempted_at=run.attempted_at,
        succeeded=succeeded and sweep is None,
        passes=passes,
        sweep=sweep,
        sweep_key=run.key,
    )
    logger.info(
        "memory_maintain",
        namespace=run.namespace,
        status=maintain._OK if succeeded else maintain._ERROR,
        decay=passes.get("decay"),
        consolidation=passes.get("consolidation"),
        verification_complete=sweep is None,
    )
    return {
        "namespace": run.namespace,
        "status": maintain._OK if succeeded else maintain._ERROR,
        "passes": passes,
        "last_attempted_at": run.attempted_at,
        "last_maintained_at": record.get("last_maintained_at", ""),
        "previous_maintained_at": run.previous,
    }


def _passed(run: MaintainRun, name: str, result: dict[str, object]) -> dict[str, object]:
    run.passes[name] = result
    return {}


async def serve_maintain(namespace: str, consolidation: dict[str, object] | None) -> dict[str, object]:
    """The registered ``memory_maintain``: every pass, and every verification slice, a job of its own."""
    runs: list[MaintainRun] = []
    configs: list[MemoryConfig] = []
    job = lane_job(namespace, "maintain")

    def authorize(_backend: StorageBackend) -> dict[str, object]:
        config = maintain.maintain_config(consolidation)
        if isinstance(config, dict):
            return config
        configs.append(config)
        return {}

    def start(backend: StorageBackend) -> dict[str, object]:
        runs.append(begin(namespace, backend, configs[0]))
        return {}

    if (refused := await job(authorize)) or (refused := claim(namespace)):
        return refused
    try:
        if refused := await job(start):
            return refused
        run = runs[0]
        for step in (
            lambda b: _passed(run, "decay", maintain._run_decay(b)),
            lambda b: _passed(run, "consolidation", maintain._run_consolidation(namespace, b, run.config)),
        ):
            if refused := await job(step):
                return refused
        verify = lambda b: verify_slice(run, b, SLICE_SECONDS, BUDGET_ROWS - run.rows_verified())  # noqa: E731
        if refused := await run_slices(job, verify, lambda: run.after is not None, run.rows_verified):
            return refused
        return await job(lambda b: (_passed(run, "wal_checkpoint", maintain._run_checkpoint(b)), finish(run, b))[1])
    finally:
        release(namespace)


async def serve_verify(
    namespace: str, project_root: str | None, settings: dict[str, object] | None
) -> dict[str, object]:
    """The registered ``memory_verify``: maintain's verification sweep alone, under the same per-call
    budget and claim, resuming and stamping the same position (so no caller can skip rows or restart
    another's sweep), and recording no maintenance attempt."""
    from trw_memory.tools.entry import checkout_path

    runs: list[MaintainRun] = []
    job = lane_job(namespace, "verify")

    def start(backend: StorageBackend) -> dict[str, object]:
        try:
            knobs = asdict(VerifySettings(**(settings or {})))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            return {"error": f"invalid verify settings: {exc}", "status": "invalid"}
        # on the pool, after the namespace is authorized: the checkout path and its files (C12 rc4)
        root = checkout_path(project_root, "memory_verify", within=False)
        if isinstance(root, dict):
            return root
        runs.append(begin(namespace, backend, MemoryConfig(project_root=root or ""), knobs))
        return {}

    if (refused := await job(lambda _b: {})) or (refused := claim(namespace)):
        return refused
    try:
        if refused := await job(start):
            return refused
        run = runs[0]
        step = lambda b: verify_slice(run, b, SLICE_SECONDS, BUDGET_ROWS - run.rows_verified())  # noqa: E731
        if (refused := await run_slices(job, step, lambda: run.after is not None, run.rows_verified)) or (
            refused := await job(lambda b: _save_sweep(run, b))
        ):
            return refused
    finally:
        release(namespace)
    done = run.passes.get("verification")
    if not isinstance(done, dict) or "complete" not in done:  # the sweep raised; it starts over next time
        return {"status": "error", "error": str(done.get("reason") if isinstance(done, dict) else "no result")}
    summary = {key: done.get(key, 0) for key in MaintainVerifySummary().as_dict()}
    reply: dict[str, object] = {"status": "ok", "summary": summary, **({"next": list(run.after)} if run.after else {})}
    # A sweep that ended in error (entry or persist failures, an unusable root, a failure carried from an
    # earlier call of it), or that checked nothing (no project root), says so with its counts (B71-109).
    if (status := done.get("status")) in (maintain._ERROR, maintain._SKIPPED):
        reply["status"], reply["error" if status == maintain._ERROR else "reason"] = status, str(done.get("reason"))
    return reply
