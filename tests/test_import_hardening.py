"""New import-hardening contract (int-700): per-path untrusted registry, an ids budget, a lane seam.

Written against a contract, not yet against code: ``trw_memory.storage._connection.OPEN_DEADLINE``
is removed in favour of ``untrusted_store``/``untrusted_deadline`` (a per-path registry any thread can
see, not a copied contextvar), ``memory_import_checkout_impl`` gains an optional ``lane`` seam that
runs the untrusted phases before the destination-owning step, and the registered tool moves those
untrusted phases onto :func:`trw_memory.daemon._offload.run_offloaded` while only the lane step keeps
:func:`trw_memory.daemon._offload.run_serialized`. ``IMPORT_MAX_IDS`` no longer gates inside the impl:
the served tool's ``ids`` argument is bounded by ``daemon._arg_bounds.ArgumentBounds`` before the tool
body runs at all (``tests/test_arg_bounds.py`` covers that contract directly). Every test here is
expected to fail to import or to fail its assertions against the pre-contract code; each docstring
says why.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from fastmcp import Client

from tests.test_tools_checkout_import import _DIM, _NS, _identical_copy, _project_store, _row
from trw_memory.server import mcp
from trw_memory.storage._connection import UNTRUSTED_LENGTH_LIMIT, connect, untrusted_deadline, untrusted_store
from trw_memory.storage._resilient_fetch import fetch_rows_via_bytes_fallback
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools import _checkout_merge, checkout_import
from trw_memory.tools.checkout_import import (
    IMPORT_COPY_DEADLINE_SECONDS,
    IMPORT_COPY_MAX_BYTES,
    IMPORT_MAX_IDS,
    memory_import_checkout_impl,
    register_checkout_import_tools,
)


@pytest.fixture
def user_store(tmp_path: Path) -> SQLiteBackend:
    store = SQLiteBackend(tmp_path / "user.db", dim=_DIM)
    if not store.vec_available:
        pytest.skip("sqlite-vec unavailable")
    yield store
    store.close()


# ---------------------------------------------------------------------------
# A. connect() capping via a per-path registry, reachable from any thread
# ---------------------------------------------------------------------------


def test_connect_caps_a_registered_path_from_a_worker_thread_and_uncaps_after(tmp_path: Path) -> None:
    """``OPEN_DEADLINE`` was a contextvar: a plain ``threading.Thread`` never copies it, so the cap
    was invisible off the caller's own (copied) context. ``untrusted_store`` registers by realpath
    instead, so a worker thread that never touched the context manager still sees the cap."""
    db = tmp_path / "any.db"
    sqlite3.connect(db).close()

    def build(size: int) -> int:
        conn = connect(db, dbapi=sqlite3, timeout=0.0, check_same_thread=False)
        try:
            return int(conn.execute("SELECT length(zeroblob(?))", (size,)).fetchone()[0])
        finally:
            conn.close()

    outcome: dict[str, object] = {}

    def capped_from_worker() -> None:
        with untrusted_store(db, time.monotonic() + 30.0):
            outcome["under_limit"] = build(UNTRUSTED_LENGTH_LIMIT)
            try:
                build(UNTRUSTED_LENGTH_LIMIT + 1)
                outcome["over_limit"] = "no_raise"
            except sqlite3.DatabaseError as exc:
                outcome["over_limit"] = str(exc)

    worker = threading.Thread(target=capped_from_worker)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()

    assert outcome["under_limit"] == UNTRUSTED_LENGTH_LIMIT
    assert "too big" in str(outcome["over_limit"])

    # An unregistered path never caps, and this same path is uncapped once the block has exited.
    other = tmp_path / "unregistered.db"
    sqlite3.connect(other).close()
    assert build(UNTRUSTED_LENGTH_LIMIT + 1) == UNTRUSTED_LENGTH_LIMIT + 1  # never registered
    assert build(UNTRUSTED_LENGTH_LIMIT + 1) == UNTRUSTED_LENGTH_LIMIT + 1  # `db`, after the block


def test_untrusted_deadline_reports_the_registered_deadline_or_none(tmp_path: Path) -> None:
    """``untrusted_deadline`` is the registry's read side; nothing registers a path outside the block."""
    db = tmp_path / "any.db"
    sqlite3.connect(db).close()
    assert untrusted_deadline(db) is None
    deadline = time.monotonic() + 30.0
    with untrusted_store(db, deadline):
        assert untrusted_deadline(db) == deadline
    assert untrusted_deadline(db) is None


def test_the_bytes_fallback_inherits_the_cap_inside_untrusted_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bytes-mode fallback opens its OWN secondary connection via ``storage._connection.connect``
    (PRD-SEC-016 round-4 finding 3); it must inherit the per-path cap the same way the primary
    connection does, since it is exactly the path an untrusted copy's decode failure routes through."""
    db = tmp_path / "copy.db"
    store = SQLiteBackend(db, dim=_DIM)
    try:
        store.store(_row("L-bad", "default"))
        store.store(_row("L-big", "default"))
        query = store._fetch_query(limit=None)
    finally:
        store.close()

    big = "x" * (UNTRUSTED_LENGTH_LIMIT + 1)
    with contextlib.closing(sqlite3.connect(db)) as conn:
        conn.execute("UPDATE memories SET content = CAST(x'ff' AS TEXT) WHERE id = 'L-bad'")
        conn.execute("UPDATE memories SET content = ? WHERE id = 'L-big'", (big,))
        conn.commit()

    outside = fetch_rows_via_bytes_fallback(db_path=db, dbapi=sqlite3, query=query)
    assert outside[1] == 1, "the bad-UTF-8 row is quarantined, not raised, with no cap in force"
    assert any(entry.id == "L-big" for entry in outside[0]), "the oversized row reads fine uncapped"

    # Capped, the fallback's own connection refuses the oversized value; the fallback answers empty
    # (logged ``fallback_failed``), never the value, and the import then fails closed on the short list.
    with untrusted_store(db, time.monotonic() + 30.0):
        rows, _quarantined = fetch_rows_via_bytes_fallback(db_path=db, dbapi=sqlite3, query=query)
    assert not any(entry.id == "L-big" for entry in rows), "the cap kept the oversized value out"


# ---------------------------------------------------------------------------
# B. memory_import_checkout_impl: IMPORT_MAX_IDS, dedup, the lane seam
# ---------------------------------------------------------------------------


async def test_too_many_ids_is_invalid_before_any_untrusted_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap must reject before the copy is ever verified or opened -- a caller cannot spend the
    daemon's I/O budget on an untrusted file merely by naming more ids than it holds.

    ``_ids_refused`` (the impl's own ids-count check) is gone: the served tool's ``ids`` argument
    is now bounded by ``daemon._arg_bounds.ArgumentBounds`` (``OVERRIDES["memory_import_checkout"]``),
    which refuses over IMPORT_MAX_IDS ids before the tool body -- ``memory_import_checkout_impl``,
    ``_checkout_merge.verify_untrusted_store`` and ``checkout_import._private_checkout_copy`` -- ever
    runs. Driven through the real served tool (not the impl directly) so this asserts the actual
    middleware-refusal contract, not a since-removed impl-level check.
    """
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    verify_calls: list[Path] = []
    copy_calls: list[object] = []
    impl_calls: list[object] = []
    monkeypatch.setattr(_checkout_merge, "verify_untrusted_store", lambda path: verify_calls.append(path))
    monkeypatch.setattr(checkout_import, "_private_checkout_copy", lambda *a: copy_calls.append(a))
    monkeypatch.setattr(
        checkout_import,
        "memory_import_checkout_impl",
        lambda *a, **k: (impl_calls.append((a, k)), {"status": "ok"})[1],
    )

    ids = [f"L-{i}" for i in range(IMPORT_MAX_IDS + 1)]
    async with Client(mcp) as client:
        result = await client.call_tool(
            "memory_import_checkout",
            {"namespace": _NS, "source_path": str(work), "ids": ids},
            raise_on_error=False,
        )

    data = result.data
    assert isinstance(data, dict)
    assert data.get("status") == "invalid", data
    assert data.get("error") == "argument_too_large", data
    assert data.get("argument") == "ids", data
    assert data.get("limit") == IMPORT_MAX_IDS, data
    assert verify_calls == []
    assert copy_calls == [], "the impl never reaches _private_checkout_copy either"
    assert impl_calls == [], "memory_import_checkout_impl never ran either -- the refusal is the middleware's"


def test_duplicate_ids_are_deduped_in_the_held_row_count(tmp_path: Path, user_store: SQLiteBackend) -> None:
    """A caller retrying with a longer, overlapping id list must not double-count a held row: the
    pre-contract ``sum(1 for entry_id in ids if ...)`` counts a repeated id twice."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1", "L-1"], backend=user_store)

    assert answer["status"] == "ok"
    assert answer["held"]["rows"] == 1


def test_the_lane_runs_once_after_the_untrusted_work_and_never_sees_a_hostile_copy(
    tmp_path: Path, user_store: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``lane`` is the seam the served tool uses to keep the destination-owning step on the
    serialized lane while the untrusted verify/open/compare run elsewhere; it must run exactly once,
    strictly after ``verify_untrusted_store``, and never at all when the copy itself is refused."""
    order: list[str] = []
    real_verify = _checkout_merge.verify_untrusted_store

    def _spy_verify(path: Path) -> None:
        order.append("verify")
        real_verify(path)

    monkeypatch.setattr(_checkout_merge, "verify_untrusted_store", _spy_verify)

    calls = 0

    def lane(step: Callable[[SQLiteBackend], dict[str, object]]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        order.append("lane")
        return step(user_store)

    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    answer = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store, lane=lane)

    assert answer["status"] == "ok"
    assert calls == 1
    assert order == ["verify", "lane"]

    # A copy trw-memory never writes is refused before ANY backend read; the lane is never reached.
    hostile = tmp_path / "hostile.db"
    with contextlib.closing(sqlite3.connect(hostile)) as conn:
        conn.executescript("CREATE TABLE t(x, CONSTRAINT c check(x > 0)); INSERT INTO t VALUES (1);")
        conn.commit()

    hostile_answer = memory_import_checkout_impl(_NS, str(hostile), [], backend=user_store, lane=lane)

    assert hostile_answer["status"] == "invalid"
    assert calls == 1, "the lane was not called a second time for the hostile copy"


def test_the_collision_compare_reads_the_copy_a_page_at_a_time(
    tmp_path: Path, user_store: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-contract compare listed the whole copy in ONE ``list_entries(limit=total+1)`` call. With
    the page shrunk to 2, a 3-row copy whose every id collides must be read in pages of at most 2."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1", "L-2", "L-3"])
    for entry_id, index in (("L-1", 0), ("L-2", 1), ("L-3", 2)):
        _identical_copy(user_store, entry_id, index)  # every id collides, identically
    monkeypatch.setattr(_checkout_merge, "_READ_PAGE", 2)

    real_list_entries = SQLiteBackend.list_entries
    real_plan = checkout_import.plan_import
    limits: list[int] = []
    planning = [False]

    def spy_list_entries(self: SQLiteBackend, **kwargs: object) -> list[object]:
        if planning[0] and kwargs.get("namespace") == "default":
            limits.append(int(kwargs.get("limit", 100)))  # type: ignore[arg-type]
        return real_list_entries(self, **kwargs)

    def spy_plan(*args: object, **kwargs: object) -> object:
        planning[0] = True
        try:
            return real_plan(*args, **kwargs)  # type: ignore[arg-type]
        finally:
            planning[0] = False

    monkeypatch.setattr(SQLiteBackend, "list_entries", spy_list_entries)
    monkeypatch.setattr(checkout_import, "plan_import", spy_plan)

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1", "L-2", "L-3"], backend=user_store)

    assert answer["status"] == "ok"
    assert (answer["moved"], answer["skipped"]) == (0, 3)
    assert len(limits) >= 2 and max(limits) <= 2, f"the copy was listed with limits {limits}"


def test_a_destination_change_between_compare_and_lane_answers_busy_and_merges_nothing(
    tmp_path: Path, user_store: SQLiteBackend
) -> None:
    """The compare runs BEFORE the lane; if the lane's own step is handed a destination that changed
    in between (a second writer, or -- as here -- the lane's own wrapper), the merge must re-check
    the colliding rows and refuse rather than merge stale-compared content."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1", "L-2"])
    _identical_copy(user_store, "L-1")  # L-1 collides, identically; L-2 is a plain new row

    def lane(step: Callable[[SQLiteBackend], dict[str, object]]) -> dict[str, object]:
        user_store.update("L-1", namespace=_NS, content="changed after the compare")
        return step(user_store)

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1", "L-2"], backend=user_store, lane=lane)

    assert answer["status"] == "busy"
    assert user_store.get("L-2", namespace=_NS) is None, "nothing merged once the recheck found drift"


# ---------------------------------------------------------------------------
# C. Registered tool: untrusted phases offloaded, only the lane step serialized
# ---------------------------------------------------------------------------


class _Captured:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self) -> object:
        return lambda fn: self.tools.setdefault(fn.__name__, fn)


def test_the_served_tool_offloads_the_untrusted_phases_and_serializes_only_the_lane_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-contract, the WHOLE served body (private copy included) ran on ``run_serialized``'s
    one-thread lane -- a large untrusted copy could stall every other writer. The contract moves the
    private copy, verify, open, reads and compare onto ``run_offloaded``'s pool and keeps only the
    destination-owning lane step on the one-thread ``run_serialized`` lane."""
    from mcp.server.auth.middleware.auth_context import auth_context_var
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from mcp.server.auth.provider import AccessToken

    import trw_memory.daemon._offload as offload_mod

    real_offloaded, real_serialized = offload_mod.run_offloaded, offload_mod.run_serialized
    calls: list[tuple[str, str]] = []

    async def _spy_offloaded(fn: Callable[..., object], /, *a: object, **kw: object) -> object:
        calls.append(("offloaded", getattr(fn, "func", fn).__name__))
        return await real_offloaded(fn, *a, **kw)

    async def _spy_serialized(fn: Callable[..., object], /, *a: object, **kw: object) -> object:
        calls.append(("serialized", getattr(fn, "func", fn).__name__))
        return await real_serialized(fn, *a, **kw)

    monkeypatch.setattr(offload_mod, "run_offloaded", _spy_offloaded)
    monkeypatch.setattr(offload_mod, "run_serialized", _spy_serialized)

    copy_thread_names: list[str] = []
    real_copy = checkout_import._private_checkout_copy

    def _spy_copy(*a: object, **kw: object) -> object:
        copy_thread_names.append(threading.current_thread().name)
        return real_copy(*a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(checkout_import, "_private_checkout_copy", _spy_copy)
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "dest-storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    root = tmp_path / "repo"
    root.mkdir()
    _project_store(root / "memory.db", ["L-1"])

    server = _Captured()
    register_checkout_import_tools(server)  # type: ignore[arg-type]
    token = AccessToken(token="t", client_id="c", scopes=[f"ns:{_NS}"], claims={"root": str(root)})
    reset = auth_context_var.set(AuthenticatedUser(token))
    try:
        answer = asyncio.run(
            server.tools["memory_import_checkout"](namespace=_NS, source_path=str(root / "memory.db"), ids=["L-1"])  # type: ignore[operator]
        )
    finally:
        auth_context_var.reset(reset)

    assert isinstance(answer, dict)
    assert copy_thread_names, "the private copy never ran"
    assert not copy_thread_names[0].startswith("trw-memory-tool-1"), (
        f"the private copy ran on the serialized lane thread ({copy_thread_names[0]})"
    )
    lanes = {kind for kind, _name in calls}
    assert lanes == {"offloaded", "serialized"}, calls
    assert calls[0][0] == "offloaded", f"the untrusted phase must be submitted first, got {calls}"


# ---------------------------------------------------------------------------
# D. Budgets
# ---------------------------------------------------------------------------


def test_the_copy_budgets_are_512mib_and_30_seconds() -> None:
    """The contract halves the byte budget (2 GiB -> 512 MiB) and the deadline (120s -> 30s); a
    silent drift here would widen or narrow the daemon's exposure without any test noticing."""
    assert IMPORT_COPY_MAX_BYTES == 512 * 1024**2
    assert IMPORT_COPY_DEADLINE_SECONDS == 30.0


def test_overlapping_registrations_of_one_file_keep_it_capped_until_the_last_ends(tmp_path: Path) -> None:
    """sol rc9 round 1: one registration's exit popped the file for every other, so a second import
    of the same path lost its cap and deadline for any later reopen."""
    db = tmp_path / "shared.db"
    sqlite3.connect(db).close()
    first, second = time.monotonic() + 60.0, time.monotonic() + 30.0
    with untrusted_store(db, first):
        with untrusted_store(db, second):
            assert untrusted_deadline(db) == second  # the earliest deadline binds
        assert untrusted_deadline(db) == first, "the inner exit must not unregister the outer import"
    assert untrusted_deadline(db) is None


def test_a_copy_with_more_graph_edges_than_an_import_takes_is_invalid(
    tmp_path: Path, user_store: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sol rc9 round 1: every source edge was materialized in Python, with no page or cap bounding it."""
    work = tmp_path / "work.db"
    _project_store(work, ["L-1", "L-2"], edge=("L-1", "L-2"))
    monkeypatch.setattr(_checkout_merge, "IMPORT_MAX_EDGES", 0)

    answer = memory_import_checkout_impl(_NS, str(work), ["L-1", "L-2"], backend=user_store)

    assert answer["status"] == "invalid"
    assert "graph edges" in str(answer["error"])
    assert user_store.get("L-1", namespace=_NS) is None


def test_a_write_the_lane_reaches_after_the_queue_budget_writes_nothing_and_answers_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sol rc9 round 2: the caller waited for the write lane with no bound, and the write's own deadline
    starts only once the lane runs it. Past the queue budget the step must not run at all."""
    from trw_memory.daemon._offload import run_serialized

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "dest-storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    ran: list[object] = []

    async def scenario() -> dict[str, object]:
        lane = checkout_import._serialized_lane(_NS, asyncio.get_running_loop(), queue_seconds=0.2)
        blocker = asyncio.ensure_future(run_serialized(time.sleep, 0.6))  # holds the one-thread lane
        await asyncio.sleep(0.05)
        answer = await asyncio.to_thread(lane, lambda backend: ran.append(backend) or {"status": "ok"})
        await blocker
        return answer

    answer = asyncio.run(scenario())

    assert answer["status"] == "busy"
    assert ran == [], "the step ran after the caller's queue budget"


def test_a_write_already_running_at_the_queue_budget_is_waited_for_not_abandoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sol rc9 round 3: the caller answered busy (and removed the copy) while a started write still ran."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "dest-storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    finished: list[bool] = []

    def slow_write(_backend: SQLiteBackend) -> dict[str, object]:
        time.sleep(0.5)  # starts at once, outlives the 0.1 s queue budget
        finished.append(True)
        return {"status": "ok", "moved": 1, "skipped": 0}

    async def scenario() -> dict[str, object]:
        lane = checkout_import._serialized_lane(_NS, asyncio.get_running_loop(), queue_seconds=0.1)
        return await asyncio.to_thread(lane, slow_write)

    answer = asyncio.run(scenario())

    assert finished == [True]
    assert answer["status"] == "ok", "a running write is the caller's answer, never an abandoned busy"
