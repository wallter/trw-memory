"""PRD-CORE-279 FR08: the daemon-side maintenance trigger.

What these pin is not "the passes ran" but "the report is true": a pass that
returns an error payload without raising must not advance
``last_maintained_at``, and the decay pass must refuse a store where its
namespace-blind UPDATE would touch rows that did not qualify.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.maintain import MAINTENANCE_STATE_FILE, memory_maintain_impl


@pytest.fixture
def backend(tmp_path):
    store = SQLiteBackend(tmp_path / "memory.db")
    yield store
    store.close()


def _stale_entry(entry_id: str, namespace: str) -> MemoryEntry:
    old = datetime.now(timezone.utc) - timedelta(days=400)
    return MemoryEntry(
        id=entry_id,
        content=f"an old note in {namespace}",
        namespace=namespace,
        status=MemoryStatus.ACTIVE,
        importance=0.8,
        created_at=old,
        updated_at=old,
        last_accessed_at=old,
    )


def test_maintain_runs_the_three_passes_and_records_the_stamp(backend, tmp_path):
    """FR08: decay, consolidation and checkpoint run; the stamp is written."""
    backend.store(_stale_entry("old-1", "project:default"))

    result = memory_maintain_impl("project:default", backend=backend)

    assert result["status"] == "ok", result
    passes = result["passes"]
    assert set(passes) == {"decay", "consolidation", "verification", "wal_checkpoint"}
    assert (passes["verification"]["status"], passes["verification"]["reason"]) == ("skipped", "no project_root")
    assert passes["decay"]["status"] == "ok", passes["decay"]
    assert passes["decay"]["scope"] == "store"
    assert passes["decay"]["processed"] == 1
    assert passes["consolidation"]["scope"] == "namespace"
    assert passes["wal_checkpoint"]["status"] == "ok"

    # The decay pass really lowered the importance of the unused entry.
    assert backend.get("old-1", namespace="project:default").importance < 0.8

    state = json.loads((tmp_path / MAINTENANCE_STATE_FILE).read_text())
    record = state["project:default"]
    assert record["last_maintained_at"] == result["last_attempted_at"]
    assert record["store"].endswith("memory.db")
    assert result["previous_maintained_at"] == ""


def test_a_second_call_reads_the_first_stamp_back(backend):
    """FR08: the recorded time is what the next caller sees as previous."""
    first = memory_maintain_impl("project:default", backend=backend)
    second = memory_maintain_impl("project:default", backend=backend)

    assert second["previous_maintained_at"] == first["last_maintained_at"]
    assert second["last_maintained_at"] >= first["last_maintained_at"]


def test_decay_keys_on_namespace_when_one_id_spans_namespaces(backend):
    """FR08: the decay pass runs on a store where one id lives in two
    namespaces, and only the row that qualified is decayed."""
    backend.store(_stale_entry("shared", "project:default"))
    fresh = _stale_entry("shared", "project:other")
    fresh.last_accessed_at = datetime.now(timezone.utc)
    backend.store(fresh)

    result = memory_maintain_impl("project:default", backend=backend)

    assert result["passes"]["decay"]["status"] == "ok"
    assert result["passes"]["decay"]["processed"] == 1
    assert backend.get("shared", namespace="project:default").importance < 0.8
    # The row that did not qualify keeps its importance.
    assert backend.get("shared", namespace="project:other").importance == pytest.approx(0.8)
    assert result["status"] == "ok"


def test_a_consolidation_with_cluster_errors_is_not_reported_as_success(backend, monkeypatch):
    """FR08: consolidate_cycle reports per-cluster failures in a payload, and a
    payload with errors is a failed pass, not a completed one."""
    import trw_memory.tools.maintain as maintain_mod

    monkeypatch.setattr(
        "trw_memory.tools.consolidate.memory_consolidate_impl",
        lambda *args, **kwargs: {
            "clusters_found": 2,
            "entries_consolidated": 0,
            "status": "completed",
            "errors": [{"cluster": "c1", "error": "merge failed"}],
        },
    )

    outcome = maintain_mod._run_consolidation("project:default", backend, MemoryConfig())
    assert outcome["status"] == "error", outcome
    assert outcome["errors"]

    result = memory_maintain_impl("project:default", backend=backend)
    assert result["status"] == "error"
    assert result["last_maintained_at"] == ""


def test_maintain_reports_a_failed_pass_without_aborting_the_rest(backend, monkeypatch):
    """FR08 negative: an error payload blocks last_maintained_at, not the run."""
    import trw_memory.tools.maintain as maintain_mod

    monkeypatch.setattr(
        maintain_mod,
        "_run_checkpoint",
        lambda _backend: {"status": "error", "reason": "busy", "scope": "store"},
    )

    result = memory_maintain_impl("project:default", backend=backend)

    assert result["status"] == "error"
    assert result["passes"]["decay"]["status"] in {"ok", "skipped"}
    assert result["passes"]["consolidation"]["status"] == "ok"
    assert result["last_maintained_at"] == "", "a failed pass still advanced the success stamp"
    assert result["last_attempted_at"], "the attempt was not recorded"


def test_a_busy_checkpoint_is_not_reported_as_success(backend, monkeypatch):
    """A payload-reported failure is a failure, even though nothing raised."""
    import trw_memory.tools.maintain as maintain_mod

    monkeypatch.setattr(
        type(backend),
        "checkpoint_wal",
        lambda self, mode="TRUNCATE": {"busy": 1, "checkpointed": 0, "log_frames": 3, "mode": "PASSIVE"},
    )

    assert maintain_mod._run_checkpoint(backend)["status"] == "error"


def test_invalid_namespace_is_reported_not_raised(backend):
    """A bad namespace is a caller error with a readable status."""
    result = memory_maintain_impl("not a namespace!", backend=backend)
    assert result["status"] == "invalid"
    assert "error" in result


async def test_maintain_is_registered_on_the_server():
    """FR08: the tool is on the LIVE registry, not only in the declared tuple."""
    from trw_memory.server import REGISTERED_TOOL_NAMES, mcp

    assert "memory_maintain" in REGISTERED_TOOL_NAMES
    assert await mcp.get_tool("memory_maintain") is not None


async def test_registered_maintain_tool_reports_scope_in_its_description():
    """FR08: the description tells the caller which passes are store-wide."""
    from trw_memory.server import mcp

    tool = await mcp.get_tool("memory_maintain")
    description = (tool.description or "").lower()
    assert "whole store" in description
    assert "consolidation applies to" in description


def test_a_corrupt_stamp_file_is_refused_not_overwritten(backend, tmp_path):
    """FR08 negative: unreadable is not the same answer as never-maintained.

    A corrupt sidecar that read as "{}" would be silently rewritten, taking
    every other namespace's stamp with it.
    """
    from trw_memory.exceptions import StorageError

    memory_maintain_impl("project:default", backend=backend)
    state_file = tmp_path / MAINTENANCE_STATE_FILE
    before = state_file.read_text()
    state_file.write_text("{not json")

    with pytest.raises(StorageError, match="cannot be read"):
        memory_maintain_impl("project:default", backend=backend)

    assert state_file.read_text() == "{not json", "the corrupt file was overwritten"
    assert "project:default" in before


def _registered_maintain(backend, monkeypatch):
    """The registered ``memory_maintain`` over *backend*, and the configs its consolidation pass saw."""
    from contextlib import nullcontext

    from trw_memory.tools import maintain as maintain_mod

    seen: list[MemoryConfig] = []

    def _consolidation(_namespace, _backend, config):
        seen.append(config)
        return {"status": "ok"}

    monkeypatch.setattr(maintain_mod, "_run_consolidation", _consolidation)
    monkeypatch.setattr(
        "trw_memory.integrations._backend.create_backend_from_config", lambda _cfg, namespace: nullcontext(backend)
    )
    tools: dict[str, object] = {}

    class _Server:
        def tool(self):
            def _register(fn):
                tools[fn.__name__] = fn
                return fn

            return _register

    maintain_mod.register_maintain_tool(_Server())  # type: ignore[arg-type]
    return tools["memory_maintain"], seen


def test_the_callers_consolidation_policy_governs_the_pass(backend, monkeypatch):
    """PRD-CORE-302 FR03: the daemon config is process-wide, so the project's policy travels per request."""
    import asyncio

    maintain, seen = _registered_maintain(backend, monkeypatch)
    policy = {"enabled": False, "similarity_threshold": 0.9, "min_cluster": 4, "max_per_cycle": 7}

    asyncio.run(maintain(namespace="project:default", consolidation=policy))
    asyncio.run(maintain(namespace="project:default"))

    assert [
        (
            c.consolidation_enabled,
            c.consolidation_similarity_threshold,
            c.consolidation_min_cluster,
            c.consolidation_max_per_cycle,
        )
        for c in seen
    ] == [(False, 0.9, 4, 7), (True, 0.75, 3, 50)]


@pytest.mark.parametrize(
    "policy",
    [
        {"enabled": True, "similarity_threshold": 1.5, "min_cluster": 3, "max_per_cycle": 50},
        {"enabled": True, "similarity_threshold": 0.75, "min_cluster": 1, "max_per_cycle": 50},
        {"enabled": True, "similarity_threshold": 0.75, "min_cluster": 3},
        {"enabled": True, "similarity_threshold": 0.75, "min_cluster": 3, "max_per_cycle": 50, "extra": 1},
    ],
    ids=["threshold-range", "min-cluster", "missing-field", "unknown-field"],
)
def test_an_invalid_consolidation_policy_is_refused_before_any_pass(backend, monkeypatch, policy):
    import asyncio

    maintain, seen = _registered_maintain(backend, monkeypatch)

    result = asyncio.run(maintain(namespace="project:default", consolidation=policy))

    assert result["status"] == "invalid"
    assert seen == []


def _anchored_checkout(tmp_path, backend):
    """A checkout reached through a symlink (macOS ``/tmp``), holding one entry anchored to a present symbol."""
    from trw_memory.models.memory import Anchor

    real = tmp_path / "real-checkout"
    real.mkdir()
    (real / "mod.py").write_text("def present_symbol():\n    return 1\n")
    link = tmp_path / "linked-checkout"
    link.symlink_to(real, target_is_directory=True)
    entry = _stale_entry("anchored-1", "project:default").model_copy(
        update={
            "anchors": [Anchor(file="mod.py", symbol_name="present_symbol", symbol_type="function")],
            "anchor_validity": 0.5,
        }
    )
    backend.store(entry)
    return link


def test_a_symlinked_project_root_still_scores_its_anchors(backend, tmp_path):
    """Release-verify F4: the configured root is resolved once, so a symlinked checkout path scores 1.0, not 0.0."""
    link = _anchored_checkout(tmp_path, backend)

    result = memory_maintain_impl("project:default", backend=backend, config=MemoryConfig(project_root=str(link)))

    assert result["passes"]["verification"]["status"] == "ok", result["passes"]["verification"]
    assert backend.get("anchored-1", namespace="project:default").anchor_validity == 1.0


def test_a_root_the_anchor_walk_refuses_leaves_the_persisted_score_untouched(backend, tmp_path):
    """Release-verify F4: "could not look" is unknown, never "every anchor is stale"."""
    from trw_memory.lifecycle import verification_pass

    link = _anchored_checkout(tmp_path, backend)

    verification_pass.run_maintain_verify(backend, project_root=link, namespace="project:default")

    assert backend.get("anchored-1", namespace="project:default").anchor_validity == 0.5


@pytest.mark.parametrize("swapped", ["root", "ancestor"])
def test_a_granted_root_swapped_for_a_symlink_after_the_grant_is_not_walked(backend, tmp_path, monkeypatch, swapped):
    """C12 (7.0 freeze): on the transport the grant's root, resolved at mint, is never re-resolved.

    A re-resolve would follow the swap and score anchors against the outside target: a
    content oracle outside the checkout. The walk must refuse it and leave the score unknown.
    """
    from trw_memory.models.memory import Anchor
    from trw_memory.tools import maintain

    base = tmp_path.resolve()
    granted = base / "parent" / "checkout"
    granted.mkdir(parents=True)
    outside = base / "outside" / "checkout"
    outside.mkdir(parents=True)
    (outside / "mod.py").write_text("def secret_symbol():\n    return 1\n")
    entry = _stale_entry("probe-1", "project:default").model_copy(
        update={
            "anchors": [Anchor(file="mod.py", symbol_name="secret_symbol", symbol_type="function")],
            "anchor_validity": 0.5,
        }
    )
    backend.store(entry)
    swap = granted if swapped == "root" else granted.parent
    swap.rename(swap.with_name(swap.name + "-moved"))
    swap.symlink_to(outside if swapped == "root" else outside.parent, target_is_directory=True)
    monkeypatch.setattr(maintain, "transport_root", lambda: (True, str(granted)))

    result = memory_maintain_impl("project:default", backend=backend, config=MemoryConfig(project_root=str(granted)))

    verification = result["passes"]["verification"]
    assert (verification["status"], verification["reason"]) == ("error", "project_root_unwalkable"), verification
    assert backend.get("probe-1", namespace="project:default").anchor_validity == 0.5


# -- rc9: a daemon maintain gives the serialized lane back between slices of its verification sweep ----


def _anchored_rows(tmp_path, backend, count, monkeypatch):
    """A checkout and *count* live rows anchored to a symbol in it (consolidation, which would merge
    them, stubbed out); returns the checkout."""
    from trw_memory.models.memory import Anchor
    from trw_memory.tools import maintain as maintain_mod

    monkeypatch.setattr(maintain_mod, "_run_consolidation", lambda *_args: {"status": "ok"})
    checkout = tmp_path.resolve() / "checkout"
    checkout.mkdir()
    (checkout / "mod.py").write_text("def present_symbol():\n    return 1\n")
    anchors = [Anchor(file="mod.py", symbol_name="present_symbol", symbol_type="function")]
    for index in range(count):
        backend.store(
            MemoryEntry(id=f"row-{index}", content=f"note {index}", namespace="project:default", anchors=anchors)
        )
    return checkout


def _checked_ids(monkeypatch, *, first_check=None):
    """The ids the sweep checks, in order; *first_check* runs inside the first check, on the lane."""
    from trw_memory.lifecycle import verification_pass

    seen: list[str] = []
    check = verification_pass.run_verification_pass

    def recording(entry_id, *args, **kwargs):
        seen.append(entry_id)
        if first_check is not None and len(seen) == 1:
            first_check()
        return check(entry_id, *args, **kwargs)

    monkeypatch.setattr(verification_pass, "run_verification_pass", recording)
    return seen


def test_an_unfinished_sweep_resumes_where_the_stamp_left_it(backend, tmp_path, monkeypatch):
    """Each call past its time checks one row (never none), stamps where it stopped, and the next call
    resumes there; last_maintained_at waits for the sweep that completes."""
    config = MemoryConfig(project_root=str(_anchored_rows(tmp_path, backend, 3, monkeypatch)))
    seen = _checked_ids(monkeypatch)

    replies = [
        memory_maintain_impl("project:default", backend=backend, config=config, verify_seconds=0.0) for _ in range(3)
    ]

    assert seen == ["row-0", "row-1", "row-2"]
    assert [(r["passes"]["verification"]["complete"], r["passes"]["verification"]["next"]) for r in replies] == [
        (False, ["project:default", "row-0"]),
        (False, ["project:default", "row-1"]),
        (True, None),
    ]
    assert [r["last_maintained_at"] for r in replies[:2]] == ["", ""]
    assert replies[2]["last_maintained_at"] == replies[2]["last_attempted_at"]
    record = json.loads((tmp_path / MAINTENANCE_STATE_FILE).read_text())["project:default"]
    assert "verify_sweeps" not in record


def test_one_maintain_call_sweeps_in_slices_up_to_its_row_budget(backend, tmp_path, monkeypatch):
    """The tool runs slices until the sweep completes or its budget is spent, adding their counts."""
    import asyncio

    monkeypatch.setenv("MEMORY_PROJECT_ROOT", str(_anchored_rows(tmp_path, backend, 3, monkeypatch)))
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.SLICE_SECONDS", 0.0)
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.BUDGET_ROWS", 2)
    seen = _checked_ids(monkeypatch)
    maintain, _ = _registered_maintain(backend, monkeypatch)

    first = asyncio.run(maintain(namespace="project:default"))
    second = asyncio.run(maintain(namespace="project:default"))

    assert seen == ["row-0", "row-1", "row-2"]
    verification = first["passes"]["verification"]
    assert (verification["complete"], verification["next"], verification["entries_processed"]) == (
        False,
        ["project:default", "row-1"],
        2,
    )
    assert first["last_maintained_at"] == ""
    assert (second["passes"]["verification"]["complete"], second["passes"]["verification"]["entries_processed"]) == (
        True,
        1,
    )
    assert second["last_maintained_at"] == second["last_attempted_at"]


def test_another_callers_serialized_write_runs_between_maintain_slices(backend, tmp_path, monkeypatch):
    """rc9: a write queued on the serialized lane while a maintain verifies runs before the sweep ends.
    Before, the whole sweep was one job, so one tenant's slow maintain held every tenant's writes."""
    import asyncio
    import threading

    from trw_memory.daemon._offload import run_serialized

    monkeypatch.setenv("MEMORY_PROJECT_ROOT", str(_anchored_rows(tmp_path, backend, 3, monkeypatch)))
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.SLICE_SECONDS", 0.0, raising=False)
    started, queued = threading.Event(), threading.Event()

    def hold_the_first_check():
        started.set()
        assert queued.wait(10)

    seen = _checked_ids(monkeypatch, first_check=hold_the_first_check)
    maintain, _ = _registered_maintain(backend, monkeypatch)

    async def scenario():
        sweep = asyncio.create_task(maintain(namespace="project:default"))
        assert await asyncio.to_thread(started.wait, 10)
        write = asyncio.create_task(run_serialized(lambda: len(seen)))
        await asyncio.sleep(0)  # the write is on the lane's queue, behind the running slice
        queued.set()
        return await write, await sweep

    checked_when_the_write_ran, reply = asyncio.run(scenario())

    assert checked_when_the_write_ran == 1
    assert reply["passes"]["verification"]["complete"] is True
    assert seen == ["row-0", "row-1", "row-2"]


def test_another_callers_serialized_write_runs_between_memory_verify_slices(backend, tmp_path, monkeypatch):
    """rc9: memory_verify runs the same sweep on the same lane, so it gives the lane back the same way,
    and still answers with the whole sweep's counts."""
    import asyncio
    import threading
    from contextlib import nullcontext

    from trw_memory.daemon._offload import run_serialized
    from trw_memory.tools import verify as verify_mod

    checkout = _anchored_rows(tmp_path, backend, 3, monkeypatch)
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.SLICE_SECONDS", 0.0, raising=False)
    monkeypatch.setattr(
        "trw_memory.integrations._backend.create_backend_from_config", lambda _cfg, namespace: nullcontext(backend)
    )
    started, queued = threading.Event(), threading.Event()

    def hold_the_first_check():
        started.set()
        assert queued.wait(10)

    seen = _checked_ids(monkeypatch, first_check=hold_the_first_check)
    tools: dict[str, object] = {}

    class _Server:
        def tool(self):
            return lambda fn: tools.setdefault(fn.__name__, fn)

    verify_mod.register_verify_tool(_Server())  # type: ignore[arg-type]

    async def scenario():
        sweep = asyncio.create_task(tools["memory_verify"](namespace="project:default", project_root=str(checkout)))
        assert await asyncio.to_thread(started.wait, 10)
        write = asyncio.create_task(run_serialized(lambda: len(seen)))
        await asyncio.sleep(0)
        queued.set()
        return await write, await sweep

    checked_when_the_write_ran, reply = asyncio.run(scenario())

    assert checked_when_the_write_ran == 1
    assert reply["status"] == "ok", reply
    assert reply["summary"]["entries_processed"] == 3
    assert "next" not in reply
    assert seen == ["row-0", "row-1", "row-2"]


def test_a_failure_in_an_earlier_call_of_a_sweep_keeps_the_sweep_from_counting_as_maintained(
    backend, tmp_path, monkeypatch
):
    """rc9 sol P1: the call that completes a sweep reports a failure an earlier call of it had."""
    from trw_memory.lifecycle import verification_pass

    config = MemoryConfig(project_root=str(_anchored_rows(tmp_path, backend, 3, monkeypatch)))
    check = verification_pass.run_verification_pass

    def failing_first_row(entry_id, *args, **kwargs):
        if entry_id == "row-0":
            raise RuntimeError("this row cannot be checked")
        return check(entry_id, *args, **kwargs)

    monkeypatch.setattr(verification_pass, "run_verification_pass", failing_first_row)

    replies = [
        memory_maintain_impl("project:default", backend=backend, config=config, verify_seconds=0.0) for _ in range(3)
    ]

    verifications = [r["passes"]["verification"] for r in replies]
    assert [(v["complete"], v["status"], v["reason"]) for v in verifications] == [
        (False, "error", "entry_failures"),
        (False, "error", "entry_failures"),
        (True, "error", "entry_failures"),
    ]
    assert [r["last_maintained_at"] for r in replies] == ["", "", ""]
    record = json.loads((tmp_path / MAINTENANCE_STATE_FILE).read_text())["project:default"]
    assert "verify_sweeps" not in record  # the next sweep starts over, from the first row


@pytest.mark.parametrize("foreign", ["namespace", "store"])
def test_a_stamped_position_from_another_namespace_or_store_starts_the_sweep_over(
    backend, tmp_path, monkeypatch, foreign
):
    """rc9 sol P1: a cursor is only resumed on the store it was stamped for, inside its namespace."""
    config = MemoryConfig(project_root=str(_anchored_rows(tmp_path, backend, 3, monkeypatch)))
    seen = _checked_ids(monkeypatch)
    memory_maintain_impl("project:default", backend=backend, config=config, verify_seconds=0.0)
    path = tmp_path / MAINTENANCE_STATE_FILE
    state = json.loads(path.read_text())
    (sweep,) = state["project:default"]["verify_sweeps"].values()
    if foreign == "namespace":
        sweep["next"] = ["project:other", "row-1"]
    else:
        sweep["store"] = [0, 0]
    path.write_text(json.dumps(state))

    memory_maintain_impl("project:default", backend=backend, config=config, verify_seconds=0.0)

    assert seen == ["row-0", "row-0"]


@pytest.mark.parametrize("other", ["root", "settings"])
def test_a_sweep_under_another_root_or_settings_keeps_its_own_position(backend, tmp_path, monkeypatch, other):
    """rc9 fix-delta sol round 2 P1: one stamped position per namespace let callers with other roots
    restart each other's sweep, and let a sweep finish under mixed thresholds. Each root and settings
    pair now keeps its own position."""
    import asyncio

    root = str(_anchored_rows(tmp_path, backend, 3, monkeypatch))
    (tmp_path / "other-checkout").mkdir()
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.SLICE_SECONDS", 0.0)
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.BUDGET_ROWS", 1)
    seen = _checked_ids(monkeypatch)
    verify = _registered_verify(backend, monkeypatch)
    elsewhere = (
        {"project_root": str(tmp_path.resolve() / "other-checkout")}
        if other == "root"
        else {"project_root": root, "settings": {"assertion_stale_threshold_days": 7}}
    )

    first = asyncio.run(verify(namespace="project:default", project_root=root))
    asyncio.run(verify(namespace="project:default", **elsewhere))
    resumed = asyncio.run(verify(namespace="project:default", project_root=root))

    assert (first["next"], resumed["next"]) == (["project:default", "row-0"], ["project:default", "row-1"])
    assert seen == ["row-0", "row-0", "row-1"]


def test_a_second_maintain_of_a_namespace_already_being_maintained_is_busy(backend, tmp_path, monkeypatch):
    """rc9 sol P1: two maintains of one namespace would interleave their slices and stamps."""
    import asyncio
    import threading

    monkeypatch.setenv("MEMORY_PROJECT_ROOT", str(_anchored_rows(tmp_path, backend, 3, monkeypatch)))
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.SLICE_SECONDS", 0.0, raising=False)
    started, queued = threading.Event(), threading.Event()

    def hold_the_first_check():
        started.set()
        assert queued.wait(10)

    _checked_ids(monkeypatch, first_check=hold_the_first_check)
    maintain, _ = _registered_maintain(backend, monkeypatch)

    async def scenario():
        first = asyncio.create_task(maintain(namespace="project:default"))
        assert await asyncio.to_thread(started.wait, 10)
        second = asyncio.create_task(maintain(namespace="project:default"))
        await asyncio.sleep(0)
        queued.set()
        return await first, await second

    first, second = asyncio.run(scenario())

    assert second["status"] == "busy", second
    assert first["passes"]["verification"]["complete"] is True
    third = asyncio.run(maintain(namespace="project:default"))
    assert third["status"] == "ok", third


def test_a_callers_batch_limit_cannot_widen_a_slices_read(backend, tmp_path, monkeypatch):
    """rc9 sol P1: the page is read before any clock check, so a timed sweep caps it."""
    from trw_memory.lifecycle import verification_pass

    root = _anchored_rows(tmp_path, backend, 3, monkeypatch)
    limits: list[int] = []
    read = backend.entries_with_assertions

    def recording(*args, **kwargs):
        limits.append(kwargs["limit"])
        return read(*args, **kwargs)

    monkeypatch.setattr(backend, "entries_with_assertions", recording)

    verification_pass.run_maintain_verify(
        backend, project_root=root, namespace="project:default", batch_limit=100_000, seconds=10.0
    )

    assert limits and max(limits) <= verification_pass.SLICED_PAGE_MAX


def test_another_callers_serialized_write_runs_between_maintain_passes(backend, tmp_path, monkeypatch):
    """rc9 sol P1: each pass is a lane job of its own, so a write waits for one pass, not for all of them."""
    import asyncio
    import threading

    from trw_memory.daemon._offload import run_serialized
    from trw_memory.tools import maintain as maintain_mod

    monkeypatch.setenv("MEMORY_PROJECT_ROOT", str(_anchored_rows(tmp_path, backend, 3, monkeypatch)))
    seen = _checked_ids(monkeypatch)
    maintain, _ = _registered_maintain(backend, monkeypatch)
    started, queued = threading.Event(), threading.Event()

    def held_consolidation(*_args):
        started.set()
        assert queued.wait(10)
        return {"status": "ok"}

    monkeypatch.setattr(maintain_mod, "_run_consolidation", held_consolidation)

    async def scenario():
        sweep = asyncio.create_task(maintain(namespace="project:default"))
        assert await asyncio.to_thread(started.wait, 10)
        write = asyncio.create_task(run_serialized(lambda: len(seen)))
        await asyncio.sleep(0)
        queued.set()
        return await write, await sweep

    checked_when_the_write_ran, reply = asyncio.run(scenario())

    assert checked_when_the_write_ran == 0
    assert reply["status"] == "ok", reply


@pytest.mark.parametrize("held", ["authorization", "verification"])
def test_a_maintain_cancelled_mid_job_leaves_the_namespace_free(backend, tmp_path, monkeypatch, held):
    """rc9 sol round 2 P1: a job outlives its cancelled caller, so a claim taken inside one could never be
    released and every later maintain would be busy."""
    import asyncio
    import threading

    from trw_memory.daemon._offload import run_serialized
    from trw_memory.tools import maintain as maintain_mod

    monkeypatch.setenv("MEMORY_PROJECT_ROOT", str(_anchored_rows(tmp_path, backend, 3, monkeypatch)))
    started, release = threading.Event(), threading.Event()

    def hold():
        started.set()
        assert release.wait(10)

    if held == "authorization":
        configure = maintain_mod.maintain_config

        def held_config(consolidation):
            hold()
            return configure(consolidation)

        monkeypatch.setattr(maintain_mod, "maintain_config", held_config)
        _checked_ids(monkeypatch)
    else:
        _checked_ids(monkeypatch, first_check=hold)
    maintain, _ = _registered_maintain(backend, monkeypatch)

    async def scenario():
        cancelled = asyncio.create_task(maintain(namespace="project:default"))
        assert await asyncio.to_thread(started.wait, 10)
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        release.set()
        await run_serialized(lambda: None)  # the orphaned job has finished
        return await maintain(namespace="project:default")

    reply = asyncio.run(scenario())

    assert reply["status"] == "ok", reply


def test_a_consolidation_pass_reads_at_most_the_row_bound_whatever_max_per_cycle_says(backend, monkeypatch):
    """rc9: consolidation runs on the serialized write lane, so its read is bounded by
    CONSOLIDATION_ROWS_MAX, never by a caller's policy or the daemon's config. Measured 2026-09-25 with
    the local embedder warm: about 0.17 s per pass at 50 near-duplicate rows (a first model load adds
    seconds, once per process)."""
    from trw_memory.lifecycle import consolidation
    from trw_memory.tools import maintain as maintain_mod
    from trw_memory.tools.consolidate import memory_consolidate_impl

    asked: list[int] = []

    def recording(_storage, _embedder=None, **kwargs):
        asked.append(kwargs["max_entries"])
        return []

    monkeypatch.setattr(consolidation, "find_clusters", recording)
    policy = {"enabled": True, "similarity_threshold": 0.75, "min_cluster": 3, "max_per_cycle": 1_000_000}

    maintain_mod._run_consolidation("project:default", backend, maintain_mod.maintain_config(policy))
    memory_consolidate_impl(
        "project:default", backend=backend, config=MemoryConfig(consolidation_max_per_cycle=1_000_000)
    )

    assert asked == [consolidation.CONSOLIDATION_ROWS_MAX] * 2


def _registered_verify(backend, monkeypatch):
    """The registered ``memory_verify`` over *backend*."""
    from contextlib import nullcontext

    from trw_memory.tools import verify as verify_mod

    monkeypatch.setattr(
        "trw_memory.integrations._backend.create_backend_from_config", lambda _cfg, namespace: nullcontext(backend)
    )
    tools: dict[str, object] = {}

    class _Server:
        def tool(self):
            return lambda fn: tools.setdefault(fn.__name__, fn)

    verify_mod.register_verify_tool(_Server())  # type: ignore[arg-type]
    return tools["memory_verify"]


def test_one_memory_verify_call_stops_at_its_row_budget_and_resumes_from_next(backend, tmp_path, monkeypatch):
    """rc9 sweep round 3 P1: memory_verify looped its slices to the end of the namespace, past the
    daemon's per-call budget. It now stops there, and the next call resumes from the stamped position
    (the caller holds no cursor, so it can neither skip rows nor restart another caller's sweep)."""
    import asyncio

    checkout = str(_anchored_rows(tmp_path, backend, 5, monkeypatch))
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.SLICE_SECONDS", 0.0)
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.BUDGET_ROWS", 2)
    seen = _checked_ids(monkeypatch)
    verify = _registered_verify(backend, monkeypatch)

    replies = [asyncio.run(verify(namespace="project:default", project_root=checkout))]
    while "next" in replies[-1] and len(replies) < 5:
        replies.append(asyncio.run(verify(namespace="project:default", project_root=checkout)))

    assert [(r["summary"]["entries_processed"], r.get("next")) for r in replies] == [
        (2, ["project:default", "row-1"]),
        (2, ["project:default", "row-3"]),
        (1, None),
    ]
    assert seen == [f"row-{index}" for index in range(5)]


@pytest.mark.parametrize("first", ["maintain", "verify"])
def test_a_verify_while_a_maintain_or_verify_of_the_namespace_runs_is_busy(backend, tmp_path, monkeypatch, first):
    """rc9 sweep round 3 P1: concurrent sweeps of one namespace each restarted it; one claim covers both."""
    import asyncio
    import threading

    checkout = str(_anchored_rows(tmp_path, backend, 3, monkeypatch))
    monkeypatch.setenv("MEMORY_PROJECT_ROOT", checkout)
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.SLICE_SECONDS", 0.0)
    started, queued = threading.Event(), threading.Event()

    def hold():
        started.set()
        assert queued.wait(10)

    _checked_ids(monkeypatch, first_check=hold)
    verify = _registered_verify(backend, monkeypatch)
    maintain, _ = _registered_maintain(backend, monkeypatch)
    running = (
        maintain(namespace="project:default")
        if first == "maintain"
        else verify(namespace="project:default", project_root=checkout)
    )

    async def scenario():
        holder = asyncio.create_task(running)
        assert await asyncio.to_thread(started.wait, 10)
        second = asyncio.create_task(verify(namespace="project:default", project_root=checkout))
        await asyncio.sleep(0)
        queued.set()
        return await holder, await second

    holder, second = asyncio.run(scenario())

    assert second["status"] == "busy", second
    assert holder["status"] == "ok", holder


@pytest.mark.parametrize("tool", ["maintain", "verify"])
def test_fast_rows_cannot_carry_one_slice_past_the_calls_row_budget(backend, tmp_path, monkeypatch, tool):
    """rc9 fix-delta sol P1: the row budget was checked only between slices, so a slice of fast rows
    ran past it. Each slice is now given the call's remaining allowance."""
    import asyncio

    checkout = str(_anchored_rows(tmp_path, backend, 5, monkeypatch))
    monkeypatch.setenv("MEMORY_PROJECT_ROOT", checkout)
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.SLICE_SECONDS", 60.0)
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.BUDGET_ROWS", 2)
    seen = _checked_ids(monkeypatch)
    if tool == "maintain":
        maintain, _ = _registered_maintain(backend, monkeypatch)
        reply = asyncio.run(maintain(namespace="project:default"))
        counts = reply["passes"]["verification"]
    else:
        reply = asyncio.run(
            _registered_verify(backend, monkeypatch)(namespace="project:default", project_root=checkout)
        )
        counts = reply["summary"]

    assert (counts["entries_processed"], seen) == (2, ["row-0", "row-1"])


def test_a_verify_whose_stamp_job_is_refused_reports_the_refusal(backend, tmp_path, monkeypatch):
    """rc9 fix-delta sol round 2 P1: a grant revoked before the final stamp job was answered ``ok``
    with a position that was never stored."""
    import asyncio

    root = str(_anchored_rows(tmp_path, backend, 3, monkeypatch))
    refusal = {"status": "refused", "error": "the grant was revoked"}
    monkeypatch.setattr("trw_memory.tools._maintain_sweep._save_sweep", lambda _run, _backend: refusal)

    reply = asyncio.run(_registered_verify(backend, monkeypatch)(namespace="project:default", project_root=root))

    assert reply == refusal


def test_a_namespace_keeps_only_its_newest_unfinished_sweeps(backend, tmp_path, monkeypatch):
    """rc9 fix-delta sol round 3 P1: sweeps are kept per root and settings, so abandoned ones would pile
    up in the stamp; only the newest VERIFY_SWEEPS_KEPT survive."""
    import asyncio

    from trw_memory.tools.maintain import VERIFY_SWEEPS_KEPT

    root = str(_anchored_rows(tmp_path, backend, 3, monkeypatch))
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.SLICE_SECONDS", 0.0)
    monkeypatch.setattr("trw_memory.tools._maintain_sweep.BUDGET_ROWS", 1)
    _checked_ids(monkeypatch)
    verify = _registered_verify(backend, monkeypatch)
    settings = [{"assertion_stale_threshold_days": days} for days in range(1, VERIFY_SWEEPS_KEPT + 3)]

    for each in settings:
        asyncio.run(verify(namespace="project:default", project_root=root, settings=each))

    sweeps = json.loads((tmp_path / MAINTENANCE_STATE_FILE).read_text())["project:default"]["verify_sweeps"]
    kept = sorted(json.loads(key)[1]["assertion_stale_threshold_days"] for key in sweeps)
    assert kept == [each["assertion_stale_threshold_days"] for each in settings][-VERIFY_SWEEPS_KEPT:]


def test_a_verify_whose_sweep_failed_answers_error_with_its_counts(backend, tmp_path, monkeypatch):
    """B71-109: a sweep with an entry that could not be checked was answered ``status: ok``; it now says
    ``error`` with the reason, and keeps its counts."""
    import asyncio

    from trw_memory.lifecycle import verification_pass

    root = str(_anchored_rows(tmp_path, backend, 3, monkeypatch))
    check = verification_pass.run_verification_pass

    def failing_first_row(entry_id, *args, **kwargs):
        if entry_id == "row-0":
            raise RuntimeError("this row cannot be checked")
        return check(entry_id, *args, **kwargs)

    monkeypatch.setattr(verification_pass, "run_verification_pass", failing_first_row)

    reply = asyncio.run(_registered_verify(backend, monkeypatch)(namespace="project:default", project_root=root))

    assert (reply["status"], reply["error"]) == ("error", "entry_failures"), reply
    assert (reply["summary"]["entry_failures"], reply["summary"]["entries_processed"]) == (1, 2)


def test_a_verify_without_a_project_root_answers_skipped(backend, monkeypatch):
    """B71-109 fix-delta: with no root nothing is checked, so the reply says ``skipped`` with its reason,
    not ``ok``."""
    import asyncio

    reply = asyncio.run(_registered_verify(backend, monkeypatch)(namespace="project:default"))

    assert (reply["status"], reply["reason"]) == ("skipped", "no project_root"), reply
    assert "summary" in reply
