"""PRD-CORE-307: the daemon's write lane -- budgets, scheduling, statement deadlines, and the census.

Every lane job names a :class:`LaneBudget`; interactive jobs start before queued background ones;
the tenant served least recently goes first; SQLite interrupts a lane job's statement past its
deadline. The census fails on any lane submission that names no registry budget.
"""

from __future__ import annotations

import ast
import asyncio
import functools
import math
import sqlite3
import statistics
import threading
import time
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from tests._timing import assert_budget
from trw_memory.daemon import _lane
from trw_memory.daemon._lane import BUDGETS, INTERACTIVE, MAINTENANCE, LaneBudget, run_on_lane
from trw_memory.daemon._offload import run_offloaded, shutdown_offload_pool
from trw_memory.storage._connection import LANE_DEADLINE, connect

_DEADLINE = 15.0
_SRC = Path(__file__).resolve().parents[1] / "src" / "trw_memory"


@pytest.fixture(autouse=True)
def _fresh_lane():
    shutdown_offload_pool()
    yield
    shutdown_offload_pool()


async def _order_after_a_blocker(queued: list[tuple[LaneBudget, str, str]], blocker_tenant: str = "a") -> list[str]:
    """Hold the lane, queue *queued* (budget, tenant, label) in order, release; the order they ran in."""
    started, release, order = threading.Event(), threading.Event(), []
    blocker = asyncio.ensure_future(run_on_lane(INTERACTIVE, blocker_tenant, lambda: (started.set(), release.wait(5))))
    assert await asyncio.to_thread(started.wait, _DEADLINE)
    jobs = []
    for budget, tenant, label in queued:
        jobs.append(asyncio.ensure_future(run_on_lane(budget, tenant, order.append, label)))
        await asyncio.sleep(0.01)  # queued in this order
    release.set()
    await asyncio.wait_for(asyncio.gather(blocker, *jobs), timeout=_DEADLINE)
    return order


async def test_an_interactive_job_starts_before_a_queued_background_job():
    """B71-79: a forget queued behind a maintenance slice ran after it (the lane was FIFO)."""
    order = await _order_after_a_blocker([(MAINTENANCE, "b", "maintain"), (INTERACTIVE, "c", "forget")])
    assert order == ["forget", "maintain"]


async def test_a_background_job_that_waited_past_the_aging_bound_competes_as_interactive(monkeypatch):
    monkeypatch.setattr(_lane, "AGING_SECONDS", 0.0)
    order = await _order_after_a_blocker([(MAINTENANCE, "b", "maintain"), (INTERACTIVE, "c", "forget")])
    assert order == ["maintain", "forget"]


async def test_the_tenant_served_least_recently_goes_first():
    """One namespace queueing many jobs cannot hold another's single job behind all of them."""
    queued = [(INTERACTIVE, "a", "a1"), (INTERACTIVE, "a", "a2"), (INTERACTIVE, "a", "a3"), (INTERACTIVE, "b", "b1")]
    assert await _order_after_a_blocker(queued, blocker_tenant="a") == ["b1", "a1", "a2", "a3"]


async def test_a_cancelled_waiter_never_runs_and_queued_jobs_are_dropped_at_shutdown():
    ran: list[str] = []
    started, release = threading.Event(), threading.Event()
    blocker = asyncio.ensure_future(run_on_lane(INTERACTIVE, "a", lambda: (started.set(), release.wait(5))))
    assert await asyncio.to_thread(started.wait, _DEADLINE)
    cancelled = asyncio.ensure_future(run_on_lane(INTERACTIVE, "b", ran.append, "cancelled"))
    dropped = asyncio.ensure_future(run_on_lane(INTERACTIVE, "c", ran.append, "dropped"))
    await asyncio.sleep(0.01)
    cancelled.cancel()
    _lane.cancel_queued()
    release.set()
    await asyncio.wait_for(blocker, timeout=_DEADLINE)
    for job in (cancelled, dropped):
        with pytest.raises(asyncio.CancelledError):
            await job
    await run_on_lane(INTERACTIVE, "a", lambda: None)  # the lane still serves
    assert ran == []


async def test_refuse_lane_submissions_refuses_a_submission_made_after_it():
    """CORE-307 row (e), audit finding F5: cancel_queued() alone lets the lane keep serving (proved
    above); a job still in flight when the daemon shuts down (FR04's per-cluster write, submitted
    after an off-lane read/embed/cluster step) must not have its NEXT submission silently land on
    a pool the offload module recreates lazily. refuse_lane_submissions() is the one-way switch."""
    assert await asyncio.wait_for(run_on_lane(INTERACTIVE, "a", lambda: "ran before shutdown"), timeout=_DEADLINE) == (
        "ran before shutdown"
    )
    try:
        _lane.refuse_lane_submissions()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_on_lane(INTERACTIVE, "a", lambda: "should never run"), timeout=_DEADLINE)
        # cancel_queued() alone (the pre-existing per-test reset call) does not clear this switch:
        # it stays tripped for every later submission in this process, by design.
        _lane.cancel_queued()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_on_lane(INTERACTIVE, "a", lambda: "still refused"), timeout=_DEADLINE)
    finally:
        _lane._shutdown = False  # restore: a one-way switch in production, reset for later tests


async def test_a_job_the_pool_never_started_is_dropped_and_the_lane_stays_free(monkeypatch):
    """Shutdown cancels the lane thread's pending work item: that job must end cancelled and free the lane,
    or every later job waits forever behind it (the pool is recreated after a shutdown)."""
    import concurrent.futures

    from trw_memory.daemon import _offload

    class _ShutDownPool:
        def submit(self, *_args, **_kwargs):
            never_started = concurrent.futures.Future()
            never_started.cancel()
            return never_started

    monkeypatch.setattr(_lane, "_executor", lambda _workers: _ShutDownPool())
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_on_lane(INTERACTIVE, "a", lambda: "never"), timeout=_DEADLINE)
    monkeypatch.setattr(_lane, "_executor", _offload._executor)
    assert await asyncio.wait_for(run_on_lane(INTERACTIVE, "a", lambda: "ran"), timeout=_DEADLINE) == "ran"


async def test_a_statement_past_the_jobs_deadline_is_interrupted_and_the_connection_outlives_it(tmp_path, monkeypatch):
    """A lane job's connection interrupts its own long statement, and never work done outside the lane."""
    monkeypatch.setattr(_lane, "STATEMENT_DEADLINE_FLOOR", 0.0)
    budget = LaneBudget("tiny", seconds=0.01, rows=1)
    endless = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n"
    kept: list[sqlite3.Connection] = []

    def job() -> None:
        assert LANE_DEADLINE.get() is not None
        conn = connect(tmp_path / "s.db", dbapi=sqlite3, timeout=1.0, check_same_thread=False)
        kept.append(conn)
        conn.execute(endless)

    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        await asyncio.wait_for(run_on_lane(budget, "a", job), timeout=_DEADLINE)
    assert LANE_DEADLINE.get() is None
    assert (
        kept[0]
        .execute(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 200000) SELECT count(*) FROM n"
        )
        .fetchone()[0]
        == 200000
    )
    kept[0].close()


async def test_a_lane_jobs_store_open_is_not_cut_by_its_deadline_and_verifies_once(tmp_path, monkeypatch):
    """sol r2 P1: every lane job opens its store, and the open's full-file quick_check (~320 ms at 20k rows) ran
    under the job's statement deadline, so a large healthy store failed every maintenance slice at open."""
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage import _connection
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    with SQLiteBackend(tmp_path / "memory.db") as store:
        for index in range(300):
            store.store(MemoryEntry(id=f"r{index}", content=f"note {index} " * 20, namespace="default"))
    _connection._VERIFIED_STORES.clear()
    monkeypatch.setattr(_lane, "STATEMENT_DEADLINE_FLOOR", 0.0)
    no_time = LaneBudget("no-time", seconds=0.0, rows=1)

    def opened() -> int:
        with SQLiteBackend(tmp_path / "memory.db") as backend:
            return backend.count()

    assert await asyncio.wait_for(run_on_lane(no_time, "a", opened), timeout=_DEADLINE) == 300
    assert _connection._store_identity(tmp_path / "memory.db") in _connection._VERIFIED_STORES


async def test_a_backend_opened_in_a_lane_job_still_enforces_the_deadline(tmp_path, monkeypatch):
    """sol r3 P1: the open ran with the deadline cleared, so connect() installed no progress handler and the
    backend's own statements were never interrupted (a 200,000-row query completed past the deadline)."""
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    monkeypatch.setattr(_lane, "STATEMENT_DEADLINE_FLOOR", 0.0)
    endless = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n"

    def job() -> None:
        with SQLiteBackend(tmp_path / "memory.db") as backend:
            backend._conn.execute(endless)

    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        await asyncio.wait_for(run_on_lane(LaneBudget("tiny", seconds=0.01, rows=1), "a", job), timeout=_DEADLINE)


async def test_a_job_past_its_budget_is_logged(monkeypatch):
    monkeypatch.setattr(_lane, "STATEMENT_DEADLINE_FLOOR", 0.0)
    with capture_logs() as events:
        await run_on_lane(LaneBudget("tiny", seconds=0.0, rows=1), "a", time.sleep, 0.01)
    overran = [event for event in events if event["event"] == "lane_job_overran"]
    assert overran and overran[0]["budget"] == "tiny" and overran[0]["tenant"] == "a"


@pytest.mark.requires_local_timing
async def test_a_forget_while_maintain_runs_answers_within_a_second(tmp_path, monkeypatch):
    """PRD-CORE-307 DoD (B71-79): p95 of memory_forget during memory_maintain <= 1 s. Each row's
    verification is slowed to 20 ms so one maintain's sweep holds the lane for seconds; before the
    budget module a verify slice held it for 2 s, so a forget waited up to that long."""
    from trw_memory.integrations._backend import create_backend_from_config
    from trw_memory.lifecycle import verification_pass
    from trw_memory.models.config import MemoryConfig
    from trw_memory.models.memory import Anchor, MemoryEntry

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "mod.py").write_text("def present_symbol():\n    return 1\n")
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_PROJECT_ROOT", str(checkout.resolve()))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)
    anchors = [Anchor(file="mod.py", symbol_name="present_symbol", symbol_type="function")]
    for namespace in ("project:a", "project:b"):
        with create_backend_from_config(MemoryConfig(), namespace) as backend:
            for index in range(150):
                backend.store(
                    MemoryEntry(id=f"r{index}", content=f"note {index}", namespace=namespace, anchors=anchors)
                )
    check = verification_pass.run_verification_pass

    def slow(*args, **kwargs):
        time.sleep(0.02)
        return check(*args, **kwargs)

    monkeypatch.setattr(verification_pass, "run_verification_pass", slow)
    from trw_memory.server import mcp

    maintain, forget = await mcp.get_tool("memory_maintain"), await mcp.get_tool("memory_forget")
    latencies: list[float] = []

    async def forgets() -> None:
        await asyncio.sleep(0.3)
        for index in range(20):
            began = time.monotonic()
            await forget.run({"memory_id": f"gone-{index}", "namespace": "project:c"})
            latencies.append(time.monotonic() - began)
            await asyncio.sleep(0.05)

    await asyncio.wait_for(
        asyncio.gather(maintain.run({"namespace": "project:a"}), maintain.run({"namespace": "project:b"}), forgets()),
        timeout=60,
    )
    p95 = statistics.quantiles(latencies, n=20)[-1]
    assert_budget("memory_forget_p95_during_maintain", p95, 1.0, "s")


async def test_consolidation_clusters_off_the_lane_and_writes_each_cluster_on_it(tmp_path, monkeypatch):
    """B71-89: memory_consolidate read, embedded and clustered on the lane, holding every tenant's writes
    for the whole cycle (seconds on a first model load). Now only each cluster's writes are lane jobs,
    and they still never interleave with each other."""
    from trw_memory.tools import consolidate as tool_mod

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)
    clustering, writing, overlap, running = [], [], [0], [0]
    guard = threading.Lock()

    def write(_backend) -> str:
        with guard:
            running[0] += 1
            overlap[0] = max(overlap[0], running[0])
        writing.append(threading.current_thread().name)
        time.sleep(0.02)
        with guard:
            running[0] -= 1
        return "written"

    def cycle(_storage, _embedder=None, *, lane, **_kwargs):
        clustering.append(threading.current_thread().name)
        return {
            "status": "completed",
            "clusters_found": 2,
            "consolidated_count": [lane(write), lane(write)].count("written"),
        }

    monkeypatch.setattr(tool_mod, "consolidate_cycle", cycle)
    monkeypatch.setattr(tool_mod, "keyword_only_on_refusal", lambda *_args, **_kwargs: (None, None))  # no model load
    from trw_memory.server import mcp

    tool = await mcp.get_tool("memory_consolidate")
    namespaces = ("project:a", "project:b", "project:c")
    await asyncio.wait_for(asyncio.gather(*(tool.run({"namespace": ns}) for ns in namespaces)), timeout=_DEADLINE)

    assert len(clustering) == 3 and not any(name.startswith("trw-memory-tool-1_") for name in clustering), clustering
    assert len(writing) == 6 and all(name.startswith("trw-memory-tool-1_") for name in writing), writing
    assert overlap[0] == 1


async def test_a_failed_cluster_writes_rollback_runs_outside_the_jobs_deadline(tmp_path, monkeypatch):
    """Audit F1: the compensating rollback ran under the job's statement deadline, so a cluster whose archive
    failed late could keep its consolidated entry beside its still-active originals."""
    from trw_memory.exceptions import StorageError
    from trw_memory.lifecycle import consolidation
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    seen: list[float | None] = []
    undo = consolidation._rollback_consolidation

    def archive_fails(*_args, **_kwargs):
        raise StorageError("late")

    def rollback(*args):
        seen.append(LANE_DEADLINE.get())
        undo(*args)

    monkeypatch.setattr(consolidation, "_archive_originals", archive_fails)
    monkeypatch.setattr(consolidation, "_rollback_consolidation", rollback)
    with SQLiteBackend(tmp_path / "memory.db") as store:
        for index in range(3):
            store.store(MemoryEntry(id=f"r{index}", content=f"note {index}", namespace="default"))

    def write() -> str:
        with SQLiteBackend(tmp_path / "memory.db") as backend:
            cluster = [backend.get(f"r{i}", namespace="default") for i in range(3)]
            return consolidation._write_cluster(
                cluster, {"summary": "note 0", "detail": ""}, None, None, "default", backend
            )  # type: ignore[arg-type]

    outcome = await asyncio.wait_for(run_on_lane(MAINTENANCE, "a", write), _DEADLINE)

    assert "late" in outcome and seen == [math.inf], (outcome, seen)
    with SQLiteBackend(tmp_path / "memory.db") as store:
        assert [e for e in store.list_entries(namespace="default") if e.source == "consolidated"] == []


def test_a_cluster_changed_after_it_was_clustered_is_skipped(tmp_path):
    """B71-89: clustered off the lane, a cluster can change before its writes run; they re-read it and skip."""
    from trw_memory.lifecycle import consolidation
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    with SQLiteBackend(tmp_path / "memory.db") as storage:
        cluster = [MemoryEntry(id=f"e{i}", content=f"content {i}", namespace="default") for i in range(3)]
        for entry in cluster:
            storage.store(entry)
        cluster = [storage.get(e.id, namespace="default") for e in cluster]
        storage.update("e1", namespace="default", content="corrected meanwhile")
        chosen = {"summary": "content 0", "detail": ""}

        outcome = consolidation._write_cluster(cluster, chosen, None, None, "default", storage)  # type: ignore[arg-type]

        assert outcome == "skipped"
        assert [e for e in storage.list_entries(namespace="default") if e.source == "consolidated"] == []
        assert all(storage.get(f"e{i}", namespace="default").consolidated_into is None for i in range(3))


def test_a_cluster_content_changed_with_updated_at_unchanged_is_not_missed(tmp_path):
    """PRD-CORE-331 (B71-135 d), audit F7: the re-read compared (status, consolidated_into,
    updated_at) -- a sync-apply that writes new content but keeps updated_at (store() writes
    verbatim, it does not auto-touch updated_at) slips through that tuple compare unchanged,
    so _write_cluster can consolidate stale content it never actually re-read. A content-digest
    compare (PRD-CORE-308's revision_of) catches it."""
    from trw_memory.lifecycle import consolidation
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    with SQLiteBackend(tmp_path / "memory.db") as storage:
        cluster = [MemoryEntry(id=f"e{i}", content=f"content {i}", namespace="default") for i in range(3)]
        for entry in cluster:
            storage.store(entry)
        cluster = [storage.get(e.id, namespace="default") for e in cluster]
        # Sync-apply race: content changes, updated_at stays exactly as clustered.
        storage.store(cluster[1].model_copy(update={"content": "changed by a concurrent sync-apply"}))
        chosen = {"summary": "content 0", "detail": ""}

        outcome = consolidation._write_cluster(cluster, chosen, None, None, "default", storage)  # type: ignore[arg-type]

        assert outcome == "skipped", (
            "the race changed e1's content between clustering and write, with updated_at held "
            "constant; the write must detect it via content digest and skip, not consolidate stale data"
        )
        assert [e for e in storage.list_entries(namespace="default") if e.source == "consolidated"] == []


async def test_a_submission_after_offload_shutdown_is_refused_and_never_recreates_the_pool():
    """PRD-CORE-331 (B71-135 e), audit F5's other half: shutdown_offload_pool() clears the pool
    but nothing stops _executor() from lazily recreating it on the next call -- and the lane's own
    dispatch thread shares that same lazy-recreate helper (_lane._dispatch -> _offload._executor(1)).
    A consolidation body still running when shutdown starts can finish after the drain and submit
    its next call, which must be refused with a clear error rather than silently landing on a
    freshly recreated pool."""
    from trw_memory.daemon import _offload

    try:
        _offload.refuse_offload_submissions()

        with pytest.raises(RuntimeError):
            await run_offloaded(lambda: "should never run after shutdown")
        assert _offload._EXECUTORS == {}, "a bare run_offloaded call after shutdown recreated the offload pool"

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_on_lane(INTERACTIVE, "t", lambda: "should never run"), timeout=_DEADLINE)
        assert _offload._EXECUTORS == {}, "a lane job dispatched after shutdown recreated the offload pool"
    finally:
        _offload._closed = False  # restore: a one-way switch in production, reset for later tests
        shutdown_offload_pool()


# --- census --------------------------------------------------------------------------------------------


@functools.cache
def _modules() -> tuple[tuple[str, ast.Module], ...]:
    return tuple(
        (str(p.relative_to(_SRC)), ast.parse(p.read_text(), filename=str(p))) for p in sorted(_SRC.rglob("*.py"))
    )


def _calls(*names: str, bare: bool = False) -> list[tuple[str, ast.Call]]:
    """Every call of *names* in the package (*bare*: only as a plain name, so ``pool.submit`` is not one)."""
    found = []
    for where, module in _modules():
        for node in ast.walk(module):
            if isinstance(node, ast.Call):
                func = node.func
                via_lane = isinstance(func, ast.Attribute) and getattr(func.value, "id", "") == "_lane"
                called = (
                    func.id if isinstance(func, ast.Name) else getattr(func, "attr", "") if via_lane or not bare else ""
                )
                if called in names:
                    found.append((where, node))
    return found


def test_every_lane_job_names_a_registry_budget_and_a_tenant():
    """Q2 census: a job put on the lane without a declared budget (or through any other door) fails here."""
    registry = {name for name, value in vars(_lane).items() if isinstance(value, LaneBudget)}
    assert {budget.name.upper() for budget in BUDGETS} == registry
    sites = _calls("run_on_lane", "submit", bare=True)
    assert len(sites) >= 12, f"the census found only {len(sites)} lane jobs: is the scan broken?"
    for where, call in sites:
        if where == "daemon/_lane.py":
            continue
        budget = call.args[0] if call.args else None
        named = budget.id if isinstance(budget, ast.Name) else budget.attr if isinstance(budget, ast.Attribute) else ""
        assert named in registry, f"{where}:{call.lineno} runs a lane job without a registry budget"
        assert len(call.args) >= 3, f"{where}:{call.lineno} names no tenant"


def test_nothing_else_reaches_the_lane_thread():
    assert not _calls("run_serialized"), "run_serialized is gone: use daemon._lane.run_on_lane"
    assert [where for where, call in _calls("_executor") if call.args] == ["daemon/_lane.py"]
    assert [where for where, _call in _calls("ThreadPoolExecutor")] == ["daemon/_offload.py"]


# --- CORE-331 FR04: the off-lane writer census -----------------------------------------------------
# A lane submission that names a budget (the census above) is not the only door onto a store: a body
# that runs on the daemon's OFFLOAD POOL (daemon._offload.run_offloaded, or serve_namespace(...,
# exclusive=False)) can write directly to its backend too, racing every other writer -- the lane's
# whole-thread guarantee never applied to it. This census requires every such write-capable site to be
# named, with a reason, in one of the two allowlists below, or migrated so its write goes through the
# lane instead (a re-read-before-write guard, like consolidation's and graph_backfill's).

#: serve_namespace(namespace, Permission.WRITE, "<operation>", ..., exclusive=False) sites: keyed
#: "path:operation" (the operation string is the tool's own stable name, not a line number).
_OFF_LANE_WRITE_ALLOWLIST = {
    "tools/maintain.py:graph_backfill": (
        "each row's edge write is re-submitted onto the lane and re-read first "
        "(_graph_backfill_lane_write, CORE-331 FR04); a row changed since the page was listed "
        "comes back 'stale' and is skipped, never written from stale data."
    ),
    "tools/reembed.py:reembed": (
        "idempotent and resumable (reembed_rows: keyset pages, re-encodes from the row's current "
        "text); a stale-read re-embed is redundant work, never corruption, and a later resumed "
        "pass re-visits the same row."
    ),
}

#: Bare run_offloaded(fn, ...) sites outside its own definition and serve_namespace's generic
#: dispatch (which the allowlist above already covers): keyed "path:enclosing_function".
_BARE_RUN_OFFLOADED_ALLOWLIST = {
    "tools/store.py:memory_store": (
        "revision-conditional write: re-reads and compares revision_of() inside backend.transaction(), "
        "refusing (status=conflict) rather than overwriting a row changed since it was read."
    ),
    "tools/recall.py:memory_recall": "read-only.",
    "tools/checkout_import.py:memory_import_checkout": (
        "the untrusted read/copy/compare run off-lane; the actual write is submitted onto the lane "
        "via _serialized_lane's lane(step)."
    ),
    "tools/consolidate.py:memory_consolidate": (
        "read/embed/cluster run off-lane; each cluster's write is submitted onto the lane via "
        "lane_writes (CORE-307 FR04). Team promotion instead runs as one INTERACTIVE lane job "
        "(CORE-307 follow-up row (c): batching that one job is a separate, tracked fix)."
    ),
    "tools/_maintain_sweep.py:serve_maintain": (
        "the same off-lane read/embed/cluster + on-lane write split as consolidate.py, for the "
        "maintain-triggered consolidation pass."
    ),
}


def _enclosing_function_name(module: ast.Module, lineno: int) -> str:
    """The innermost def whose body spans *lineno* -- the census's site key, not the line number."""
    best: tuple[int, int, str] | None = None
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            if node.lineno <= lineno <= end and (best is None or (end - node.lineno) < (best[1] - best[0])):
                best = (node.lineno, end, node.name)
    return best[2] if best else "<module>"


def _off_lane_write_sites() -> tuple[list[str], list[str]]:
    """Every ``serve_namespace(..., exclusive=False)`` WRITE site, and every bare ``run_offloaded``
    site outside its own definition and ``entry.py``'s generic dispatch, each as its allowlist key."""
    writes, bare = [], []
    for where, module in _modules():
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "serve_namespace":
                exclusive_false = any(
                    kw.arg == "exclusive" and isinstance(kw.value, ast.Constant) and kw.value.value is False
                    for kw in node.keywords
                )
                permission = node.args[1] if len(node.args) > 1 else None
                is_write = isinstance(permission, ast.Attribute) and permission.attr == "WRITE"
                if exclusive_false and is_write:
                    operation = node.args[2] if len(node.args) > 2 else None
                    op_name = operation.value if isinstance(operation, ast.Constant) else f"line {node.lineno}"
                    writes.append(f"{where}:{op_name}")
            elif isinstance(func, ast.Name) and func.id == "run_offloaded" and where != "daemon/_offload.py":
                if where == "tools/entry.py" and _enclosing_function_name(module, node.lineno) == "serve_namespace":
                    continue  # the generic dispatch itself; its callers are covered by the census above
                bare.append(f"{where}:{_enclosing_function_name(module, node.lineno)}")
    return writes, bare


def test_every_off_lane_write_site_is_allowlisted_or_migrated():
    """Q2 census (CORE-331 FR04): a new serve_namespace(exclusive=False) WRITE site, or a new bare
    run_offloaded site, must be reasoned about here -- allowlisted with why its write is safe off the
    lane, or migrated to submit its write onto the lane instead."""
    writes, bare = _off_lane_write_sites()
    assert len(writes) >= 2, f"the census found only {len(writes)} off-lane write sites: is the scan broken?"
    assert len(bare) >= 5, f"the census found only {len(bare)} bare run_offloaded sites: is the scan broken?"
    assert set(writes) == set(_OFF_LANE_WRITE_ALLOWLIST), (
        f"unlisted off-lane write site(s): {set(writes) - set(_OFF_LANE_WRITE_ALLOWLIST)}; "
        f"stale allowlist entries: {set(_OFF_LANE_WRITE_ALLOWLIST) - set(writes)}"
    )
    assert set(bare) == set(_BARE_RUN_OFFLOADED_ALLOWLIST), (
        f"unlisted bare run_offloaded site(s): {set(bare) - set(_BARE_RUN_OFFLOADED_ALLOWLIST)}; "
        f"stale allowlist entries: {set(_BARE_RUN_OFFLOADED_ALLOWLIST) - set(bare)}"
    )


def test_the_off_lane_census_catches_an_unlisted_write_site():
    """Proves the scan itself, not just the allowlist: a fixture module with an unlisted
    serve_namespace(exclusive=False) WRITE site, and an unlisted bare run_offloaded site, are both
    found -- so a real new site with neither a fix nor an allowlist entry would fail the test above."""
    fixture = ast.parse(
        "async def memory_new_tool(namespace):\n"
        "    return await serve_namespace(namespace, Permission.WRITE, 'new_op', run, exclusive=False)\n"
        "\n"
        "async def memory_other_new_tool(namespace):\n"
        "    return await run_offloaded(_write_body)\n",
        filename="tools/fixture_unlisted.py",
    )
    writes, bare = [], []
    for node in ast.walk(fixture):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "serve_namespace":
            writes.append(f"tools/fixture_unlisted.py:{node.args[2].value}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_offloaded":
            bare.append(f"tools/fixture_unlisted.py:{_enclosing_function_name(fixture, node.lineno)}")
    assert writes == ["tools/fixture_unlisted.py:new_op"]
    assert bare == ["tools/fixture_unlisted.py:memory_other_new_tool"]
    assert not set(writes) & set(_OFF_LANE_WRITE_ALLOWLIST)
    assert not set(bare) & set(_BARE_RUN_OFFLOADED_ALLOWLIST)
