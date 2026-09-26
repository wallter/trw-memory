"""PRD-CORE-279 FR04/NFR02/NFR04: served tool bodies run off the event loop.

Every test here carries a deadline. Without one, a regression that puts the
work back on the event loop would not fail these tests -- it would hang them,
and a hung suite reads as "still running" rather than "broken".
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time

import pytest

from trw_memory.daemon._offload import (
    OFFLOAD_MAX_WORKERS,
    run_offloaded,
    shutdown_offload_pool,
)

_DEADLINE = 15.0


@pytest.fixture(autouse=True)
def _fresh_pool():
    shutdown_offload_pool()
    yield
    shutdown_offload_pool()


async def test_offload_returns_the_callables_result():
    """NFR02: the helper is transport-neutral -- same callable, same payload."""
    payload = {"memories": [{"id": "a"}], "total_matches": 1}

    async def _call() -> dict[str, object]:
        return await run_offloaded(lambda: payload)

    assert await asyncio.wait_for(_call(), timeout=_DEADLINE) is payload


async def test_offload_propagates_exceptions():
    """A failure in the worker reaches the caller unchanged."""

    def _boom() -> None:
        raise ValueError("no")

    with pytest.raises(ValueError, match="no"):
        await asyncio.wait_for(run_offloaded(_boom), timeout=_DEADLINE)


async def test_offload_runs_on_a_worker_thread():
    """FR04: the body does not execute on the loop's thread."""
    loop_thread = threading.current_thread().name

    name = await asyncio.wait_for(
        run_offloaded(lambda: threading.current_thread().name),
        timeout=_DEADLINE,
    )
    assert name != loop_thread
    assert name.startswith("trw-memory-tool")


async def test_recall_runs_off_the_event_loop():
    """FR04: two calls overlap. On the old serial path the second never starts."""
    started = threading.Event()
    release = threading.Event()

    def _blocking() -> str:
        started.set()
        assert release.wait(timeout=_DEADLINE), "the second call never reached the worker pool"
        return "first"

    first = asyncio.ensure_future(run_offloaded(_blocking))
    # Yield to the loop so the first call is submitted before the second.
    await asyncio.sleep(0)
    assert started.wait(timeout=_DEADLINE), "the first call never started: it is still on the loop"

    second = await asyncio.wait_for(run_offloaded(lambda: "second"), timeout=_DEADLINE)
    assert second == "second", "a second call could not run while the first was busy"
    release.set()
    assert await asyncio.wait_for(first, timeout=_DEADLINE) == "first"


async def test_offload_is_bounded_by_the_worker_count():
    """NFR04: never more than OFFLOAD_MAX_WORKERS callables run at once."""
    lock = threading.Lock()
    in_flight = 0
    peak = 0
    release = threading.Event()

    def _occupy() -> None:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        release.wait(timeout=_DEADLINE)
        with lock:
            in_flight -= 1

    tasks = [asyncio.ensure_future(run_offloaded(_occupy)) for _ in range(OFFLOAD_MAX_WORKERS + 3)]
    deadline = time.monotonic() + _DEADLINE
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        with lock:
            if in_flight >= OFFLOAD_MAX_WORKERS:
                break
    with lock:
        assert in_flight == OFFLOAD_MAX_WORKERS, f"expected the pool to be saturated, saw {in_flight}"
    release.set()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=_DEADLINE)
    assert peak <= OFFLOAD_MAX_WORKERS, f"pool ran {peak} callables at once"


async def test_offload_carries_the_caller_context():
    """Log context bound by the caller must reach the worker, and not leak."""
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="unset")

    marker.set("request-a")
    seen_a = await asyncio.wait_for(run_offloaded(marker.get), timeout=_DEADLINE)
    assert seen_a == "request-a"

    async def _other_request() -> str:
        marker.set("request-b")
        return await run_offloaded(marker.get)

    seen_b = await asyncio.wait_for(asyncio.ensure_future(_other_request()), timeout=_DEADLINE)
    assert seen_b == "request-b"
    # The worker that ran request-b must not still be carrying it.
    assert await asyncio.wait_for(run_offloaded(marker.get), timeout=_DEADLINE) == "request-a"


async def test_pool_is_recreated_after_shutdown():
    """Shutdown is not terminal: the next call gets a working pool."""
    assert await asyncio.wait_for(run_offloaded(lambda: 1), timeout=_DEADLINE) == 1
    shutdown_offload_pool()
    assert await asyncio.wait_for(run_offloaded(lambda: 2), timeout=_DEADLINE) == 2


async def test_registered_recall_tool_offloads(tmp_path, monkeypatch):
    """FR04 integration: the registered memory_recall body leaves the loop."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)

    from trw_memory.server import mcp

    tool = await mcp.get_tool("memory_recall")
    loop_thread = threading.current_thread().name
    observed: list[str] = []

    import trw_memory.tools.recall as recall_mod

    original = recall_mod.memory_recall_impl

    def _spy(*args, **kwargs):
        observed.append(threading.current_thread().name)
        return original(*args, **kwargs)

    monkeypatch.setattr(recall_mod, "memory_recall_impl", _spy)

    await asyncio.wait_for(
        tool.run({"query": "anything", "namespace": "project:default", "include_org_memories": False}),
        timeout=60.0,
    )
    assert observed, "the impl never ran"
    assert observed[0] != loop_thread
    assert observed[0].startswith("trw-memory-tool")


async def test_registered_store_tool_offloads(tmp_path, monkeypatch):
    """FR04 integration: the registered memory_store body leaves the loop too."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)

    from trw_memory.server import mcp

    tool = await mcp.get_tool("memory_store")
    loop_thread = threading.current_thread().name
    observed: list[str] = []

    import trw_memory.tools.store as store_mod

    original = store_mod.memory_store_impl

    def _spy(*args, **kwargs):
        observed.append(threading.current_thread().name)
        return original(*args, **kwargs)

    monkeypatch.setattr(store_mod, "memory_store_impl", _spy)

    await asyncio.wait_for(
        tool.run({"content": "a fact stored off the loop", "namespace": "project:default"}),
        timeout=60.0,
    )
    assert observed, "the impl never ran"
    assert observed[0] != loop_thread
    assert observed[0].startswith("trw-memory-tool")


async def test_shutdown_is_bounded_when_a_worker_will_not_stop():
    """NFR04: a stuck call must not hold shutdown open indefinitely."""
    release = threading.Event()
    started = threading.Event()

    def _stuck() -> None:
        started.set()
        release.wait(timeout=30)

    task = asyncio.ensure_future(run_offloaded(_stuck))
    await asyncio.sleep(0)
    assert started.wait(timeout=_DEADLINE)

    began = time.monotonic()
    drained = await asyncio.to_thread(shutdown_offload_pool, timeout=0.5)
    elapsed = time.monotonic() - began

    assert drained is False, "shutdown claimed a clean drain while a call was stuck"
    assert elapsed < 5.0, f"shutdown waited {elapsed:.1f}s despite a 0.5s grace"
    release.set()
    await asyncio.wait_for(task, timeout=_DEADLINE)


@pytest.mark.parametrize(
    ("name", "module", "impl", "args"),
    [
        (
            "memory_update",
            "trw_memory.tools.update",
            "memory_update_impl",
            {"entry_id": "L-x", "patch": {"summary": "re-encoded"}, "namespace": "project:default"},
        ),
        (
            "memory_vectors",
            "trw_memory.tools.recall_support",
            "memory_vectors_impl",
            {"namespace": "project:default", "ids": ["L-x"]},
        ),
        # C12 rc4: verify reads every anchored and asserted checkout file, and every other served body
        # (SQLite, the model, checkout files) now runs on the pool: one of each handler shape
        ("memory_verify", "trw_memory.tools._maintain_sweep", "verify_slice", {"namespace": "project:default"}),
        (
            "memory_get",
            "trw_memory.tools.entry",
            "memory_get_impl",
            {"memory_id": "L-x", "namespace": "project:default"},
        ),
        ("memory_consolidate", "trw_memory.tools.consolidate", "memory_consolidate_impl", {"dry_run": True}),
        ("memory_search", "trw_memory.tools.search", "memory_search_impl", {}),
        ("memory_forget", "trw_memory.tools.forget", "memory_forget_impl", {"memory_id": "L-x"}),
        ("memory_status", "trw_memory.tools.status", "memory_status_impl", {}),
        ("memory_audit", "trw_memory.tools.audit", "memory_audit_impl", {"learning_id": "L-x"}),
        ("memory_namespace_diagnose", "trw_memory.tools.namespace_admin", "memory_namespace_diagnose_impl", {}),
        ("memory_quarantine_list", "trw_memory.tools.review", "memory_quarantine_list_impl", {}),
    ],
)
async def test_a_model_load_in_one_request_does_not_stall_another(tmp_path, monkeypatch, name, module, impl, args):
    """C12 (7.0.0 rc2): update re-encodes and vectors resolves the space, both of which load the model.

    With the body on the event loop, one caller's model load froze every other tenant's request.
    """
    import importlib

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)
    from trw_memory.server import mcp

    release = threading.Event()

    def _loading_the_model(*_args, **_kwargs):
        release.wait(timeout=3.0)  # bounded: on the loop this is how long everything else waits
        return {"status": "ok"}

    monkeypatch.setattr(importlib.import_module(module), impl, _loading_the_model)
    tool = await mcp.get_tool(name)

    slow = asyncio.ensure_future(tool.run(args))
    try:
        await asyncio.sleep(0.05)  # an unrelated request, served while the model is still loading
        unrelated_finished_first = not slow.done()
    finally:
        release.set()
    await asyncio.wait_for(slow, timeout=_DEADLINE)
    assert unrelated_finished_first, f"{name} held the event loop while its model loaded"


@pytest.mark.parametrize(
    ("name", "module", "args"),
    [
        ("memory_status", "trw_memory.tools.status", {"security_settings_only": True}),
        ("memory_search", "trw_memory.tools.search", {}),
        ("memory_forget", "trw_memory.tools.forget", {"memory_id": "L-x"}),
        ("memory_consolidate", "trw_memory.tools.consolidate", {"dry_run": True}),
        ("memory_verify", "trw_memory.tools.entry", {"namespace": "project:default"}),  # in_namespace reads it
    ],
)
async def test_reading_config_does_not_stall_another_request(tmp_path, monkeypatch, name, module, args):
    """C12 rc4 sweep: MemoryConfig() reads .trw/config.yaml; built on the loop, a slow read stalled every tenant."""
    import importlib

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)
    from trw_memory.server import mcp

    release = threading.Event()

    class _SlowConfig:
        def __new__(cls, *_args, **_kwargs):
            release.wait(timeout=3.0)
            raise RuntimeError("config read")  # the body's answer does not matter, only where it waited

    monkeypatch.setattr(importlib.import_module(module), "MemoryConfig", _SlowConfig)
    tool = await mcp.get_tool(name)

    slow = asyncio.ensure_future(tool.run(args))
    try:
        await asyncio.sleep(0.05)
        unrelated_finished_first = not slow.done()
    finally:
        release.set()
    with pytest.raises(Exception, match="config read"):
        await asyncio.wait_for(slow, timeout=_DEADLINE)
    assert unrelated_finished_first, f"{name} read its config on the event loop"


@pytest.mark.parametrize(
    ("name", "module", "impl", "args"),
    [
        ("memory_namespace_rename", "trw_memory.tools.namespace_admin", "memory_namespace_rename_impl", {}),
        ("memory_namespace_merge", "trw_memory.tools.namespace_admin", "memory_namespace_merge_impl", {}),
        ("memory_consolidate", "trw_memory.tools.consolidate", "memory_consolidate_impl", {"dry_run": True}),
    ],
)
async def test_check_then_write_curation_never_interleaves(tmp_path, monkeypatch, name, module, impl, args):
    """C12 rc4: on the loop these ran one at a time; on the pool two renames could both see an empty destination."""
    import importlib

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)
    from trw_memory.server import mcp

    running, overlap = [0], [0]
    guard = threading.Lock()

    def _curating(*_args, **_kwargs):
        with guard:
            running[0] += 1
            overlap[0] = max(overlap[0], running[0])
        time.sleep(0.1)
        with guard:
            running[0] -= 1
        return {"status": "ok"}

    monkeypatch.setattr(importlib.import_module(module), impl, _curating)
    tool = await mcp.get_tool(name)
    if "source" in tool.parameters.get("properties", {}):
        args = {"source": "project:a", "destination": "project:b"}

    await asyncio.wait_for(asyncio.gather(*(tool.run(args) for _ in range(3))), timeout=_DEADLINE)

    assert overlap[0] == 1, f"{name}: {overlap[0]} bodies ran at once"


async def test_token_verification_reads_the_grants_file_off_the_loop(tmp_path, monkeypatch):
    """C12 rc4 sweep: every request re-reads the grants file; on the loop a slow read stalled every tenant."""
    from trw_memory.daemon import _verifier
    from trw_memory.daemon._paths import DaemonPaths

    release = threading.Event()
    monkeypatch.setattr(_verifier, "read_grant", lambda *_a: release.wait(timeout=3.0) and None)
    verifier = _verifier.LoopbackTokenVerifier(DaemonPaths(user_memory_dir=tmp_path))

    slow = asyncio.ensure_future(verifier.verify_token("t"))
    try:
        await asyncio.sleep(0.05)
        unrelated_finished_first = not slow.done()
    finally:
        release.set()
    assert await asyncio.wait_for(slow, timeout=_DEADLINE) is None
    assert unrelated_finished_first, "verify_token read the grants file on the event loop"


async def test_a_cancelled_curation_caller_holds_the_lock_until_its_body_finishes():
    """C12 rc4: cancelling the caller released the lock while its body kept running, so a second rename interleaved."""
    from trw_memory.daemon._offload import run_serialized

    started, release, order = threading.Event(), threading.Event(), []

    def _first():
        started.set()
        release.wait(timeout=3.0)
        order.append("first")

    caller = asyncio.ensure_future(run_serialized(_first))
    assert await asyncio.to_thread(started.wait, _DEADLINE)
    caller.cancel()
    follower = asyncio.ensure_future(run_serialized(order.append, "second"))
    await asyncio.sleep(0.1)
    ran_early = list(order)
    release.set()
    await asyncio.wait_for(follower, timeout=_DEADLINE)

    assert ran_early == [] and order == ["first", "second"], order


async def test_the_bodies_the_loop_ran_one_at_a_time_still_never_interleave(tmp_path, monkeypatch):
    """C12 rc4 round 3: offloaded, a forget could delete a quarantined row between a review's read and its promotion,
    or a source row between a consolidation's clustering and its write (rc7: maintain's, on the pool)."""
    import importlib

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)
    from trw_memory.server import mcp

    running, overlap = [0], [0]
    guard = threading.Lock()

    def _body(*_args, **_kwargs):
        with guard:
            running[0] += 1
            overlap[0] = max(overlap[0], running[0])
        time.sleep(0.05)
        with guard:
            running[0] -= 1
        return {"status": "ok"}

    calls = [
        ("memory_forget", "trw_memory.tools.forget", "memory_forget_impl", {"memory_id": "L-x"}),
        (
            "memory_review",
            "trw_memory.tools.review",
            "memory_review_impl",
            {"learning_id": "L-x", "decision": "approve"},
        ),
        ("memory_consolidate", "trw_memory.tools.consolidate", "memory_consolidate_impl", {"dry_run": True}),
        (
            "memory_namespace_rename",
            "trw_memory.tools.namespace_admin",
            "memory_namespace_rename_impl",
            {"source": "project:a", "destination": "project:b"},
        ),
        ("memory_search", "trw_memory.tools.search", "memory_search_impl", {}),
        (
            "memory_get",
            "trw_memory.tools.entry",
            "memory_get_impl",
            {"memory_id": "L-x", "namespace": "project:default"},
        ),
        (
            "memory_sync_find",
            "trw_memory.tools.sync",
            "memory_sync_find_impl",
            {"namespace": "project:default", "remote_id": "r", "ids": []},
        ),
        # rc7: every learning-row writer shares the lane -- maintain's consolidation raced a forget, an
        # update landed between a sync apply's write and its ack or a verify's check and its write-back, and a
        # forget between a correction's read and write
        ("memory_verify", "trw_memory.tools._maintain_sweep", "verify_slice", {"namespace": "project:default"}),
        (
            "memory_sync_apply",
            "trw_memory.tools.sync",
            "memory_sync_apply_impl",
            {"namespace": "project:default", "entry": {"id": "L-x"}},
        ),
        (
            "memory_update",
            "trw_memory.tools.update",
            "memory_update_impl",
            {"entry_id": "L-x", "patch": {"summary": "s"}, "namespace": "project:default"},
        ),
        ("memory_maintain", "trw_memory.tools.maintain", "memory_maintain_impl", {}),
        # rc9: only the import's write step takes the lane; the copy, its checks and reads run off it
        (
            "memory_import_checkout",
            "trw_memory.tools.checkout_import",
            "_on",
            {"namespace": "project:default", "source_path": "x", "ids": []},
        ),
    ]
    for _name, module, impl, _args in calls:
        monkeypatch.setattr(importlib.import_module(module), impl, _body)
    checkout_import = importlib.import_module("trw_memory.tools.checkout_import")
    monkeypatch.setattr(checkout_import, "_on", lambda _step: _body)  # the lane's body is the write step
    monkeypatch.setattr(checkout_import, "checkout_path", lambda *_args, **_kwargs: "x")
    monkeypatch.setattr(checkout_import, "transport_root", lambda: (False, None))
    monkeypatch.setattr(checkout_import, "memory_import_checkout_impl", lambda *_args, lane, **_kwargs: lane(None))
    tools = [(await mcp.get_tool(name), args) for name, _module, _impl, args in calls]

    await asyncio.wait_for(asyncio.gather(*(tool.run(args) for tool, args in tools)), timeout=_DEADLINE)

    assert overlap[0] == 1, f"{overlap[0]} formerly loop-serialized bodies ran at once"


async def test_a_saturated_pool_never_delays_the_serialized_bodies(tmp_path, monkeypatch):
    """C12 rc4 round 4: sharing the pool, four long reembeds starved memory_status and memory_forget, which the loop
    used to answer without any pool capacity; their one-thread lane is separate."""
    from trw_memory.daemon._offload import OFFLOAD_MAX_WORKERS, run_offloaded

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)
    from trw_memory.server import mcp

    release = threading.Event()
    hogs = [asyncio.ensure_future(run_offloaded(release.wait, 5.0)) for _ in range(OFFLOAD_MAX_WORKERS)]
    try:
        status = await mcp.get_tool("memory_status")
        answer = await asyncio.wait_for(status.run({"security_settings_only": True}), timeout=2.0)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(*hogs), timeout=_DEADLINE)
    assert answer is not None
