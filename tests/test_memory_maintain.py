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
    assert set(passes) == {"decay", "consolidation", "wal_checkpoint"}
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
    from trw_memory.models.config import MemoryConfig

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
