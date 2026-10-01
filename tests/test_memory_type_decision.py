"""PRD-CORE-334: a DECISION memory type, a ``types`` recall filter, and a lenient sync ingress.

FR01 adds ``MemoryType.DECISION``. NFR02: a row with no type is PATTERN and
never matches a non-pattern filter. The ``types`` filter runs inside the
candidate query (``memory_recall`` and ``memory_list_page``), so a match past the
limit is still found. FR05: a synced row whose type this build does not know
(an older client reading a newer one's row) is stored as PATTERN with the raw
string in ``metadata["type_raw"]`` instead of being refused.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from tests.conftest import make_entry
from trw_memory.client import MemoryClient
from trw_memory.models._type_coercion import coerce_memory_type_lenient
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryType
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.listing import memory_list_page_impl, register_list_page_tool
from trw_memory.tools.recall import memory_recall_impl, register_recall_tool
from trw_memory.tools.status import memory_status_impl
from trw_memory.tools.sync import memory_sync_apply_impl

_NS = "project:ledger-1a2b3c4d"
_T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
_LEGACY = ("incident", "pattern", "convention", "hypothesis", "workaround")


def _row(entry_id: str, kind: str, *, minutes: int = 0, content: str = "") -> MemoryEntry:
    stamp = _T0 + timedelta(minutes=minutes)
    base = make_entry(entry_id=entry_id, namespace=_NS, content=content or f"ledger {kind} {entry_id}")
    return base.model_copy(update={"type": kind, "created_at": stamp, "updated_at": stamp})


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    yield store
    store.close()


@pytest.fixture
def config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_path=str(tmp_path), hybrid_search_candidate_pool_size=10)


def _recall_ids(backend: SQLiteBackend, config: MemoryConfig, query: str, **kwargs: object) -> list[str]:
    answer = memory_recall_impl(
        query, _NS, backend=backend, config=config, include_org_memories=False, record_access=False, **kwargs
    )
    return sorted(str(row["id"]) for row in answer["memories"])  # type: ignore[union-attr]


# -- FR01 -------------------------------------------------------------------------------------------


def test_decision_round_trips(backend: SQLiteBackend) -> None:
    backend.store(_row("D-1", "decision"))

    loaded = backend.get("D-1", namespace=_NS)

    assert loaded is not None and loaded.type == "decision"
    assert MemoryEntry.model_validate_json(loaded.model_dump_json()).type == "decision"


@pytest.mark.parametrize("kind", _LEGACY)
def test_every_pre_existing_type_loads_unchanged(backend: SQLiteBackend, kind: str) -> None:
    backend.store(_row("L-1", kind))

    loaded = backend.get("L-1", namespace=_NS)

    assert loaded is not None and loaded.type == kind and "type_raw" not in loaded.metadata


def test_a_strict_model_still_rejects_an_unknown_type_naming_all_six() -> None:
    with pytest.raises(ValidationError, match="decision") as caught:
        MemoryEntry(id="X-1", content="c", type="retrospective")  # type: ignore[arg-type]
    assert all(kind in str(caught.value) for kind in _LEGACY)


# -- NFR02 ------------------------------------------------------------------------------------------


def test_a_row_with_no_type_is_pattern_and_never_a_decision(backend: SQLiteBackend, config: MemoryConfig) -> None:
    legacy = make_entry(entry_id="OLD-1", namespace=_NS, content="ledger legacy").model_dump(mode="json")
    legacy.pop("type")
    backend.store(MemoryEntry.model_validate(legacy))
    backend.store(_row("D-1", "decision", minutes=1))

    assert backend.get("OLD-1", namespace=_NS).type == "pattern"  # type: ignore[union-attr]
    assert _recall_ids(backend, config, "", types=["decision"]) == ["D-1"]
    assert _recall_ids(backend, config, "", types=["pattern"]) == ["OLD-1"]


# -- the types filter -------------------------------------------------------------------------------


def _mixed(backend: SQLiteBackend) -> None:
    for index, kind in enumerate(("incident", "decision", "pattern", "convention", "hypothesis", "workaround")):
        backend.store(_row(f"{kind[0].upper()}-1", kind, minutes=index))


@pytest.mark.parametrize("query", ["", "ledger"])
@pytest.mark.parametrize(("types", "expected"), [(["decision"], ["D-1"]), (["incident"], ["I-1"])])
def test_recall_returns_only_the_requested_types(
    backend: SQLiteBackend, config: MemoryConfig, query: str, types: list[str], expected: list[str]
) -> None:
    _mixed(backend)

    assert _recall_ids(backend, config, query, types=types) == expected


def test_omitting_types_is_unfiltered(backend: SQLiteBackend, config: MemoryConfig) -> None:
    _mixed(backend)

    assert _recall_ids(backend, config, "") == ["C-1", "D-1", "H-1", "I-1", "P-1", "W-1"]


def test_a_match_older_than_the_candidate_pool_is_still_recalled(
    backend: SQLiteBackend, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # limit 1 reads a pool of 25 rows; the one decision is older than 40 patterns. The tier index would
    # supply it anyway (it holds every row here), so it is off: this measures the store query alone.
    monkeypatch.setattr("trw_memory.tools.recall.supports_tier_runtime", lambda _backend: False)
    backend.store(_row("D-OLD", "decision", minutes=0))
    for index in range(40):
        backend.store(_row(f"P-{index:02d}", "pattern", minutes=10 + index))

    assert _recall_ids(backend, config, "", types=["decision"], limit=1) == ["D-OLD"]


@pytest.mark.parametrize("types", [["decisions"], ["decision", "bogus"]])
def test_an_unknown_type_filter_is_refused(backend: SQLiteBackend, config: MemoryConfig, types: list[str]) -> None:
    answer = memory_recall_impl("", _NS, backend=backend, config=config, types=types)
    assert answer["status"] == "invalid"
    assert "memories" not in answer
    listed = memory_list_page_impl(_NS, 10, None, backend=backend, config=config, types=types)
    assert listed["status"] == "invalid"


def test_a_list_page_filters_by_type_before_its_limit(backend: SQLiteBackend, config: MemoryConfig) -> None:
    backend.store(_row("D-OLD", "decision", minutes=0))
    for index in range(5):
        backend.store(_row(f"P-{index}", "pattern", minutes=10 + index))

    page = memory_list_page_impl(_NS, 1, None, backend=backend, config=config, types=["decision"])

    assert [row["id"] for row in page["entries"]] == ["D-OLD"]  # type: ignore[union-attr]
    unfiltered = memory_list_page_impl(_NS, 1, None, backend=backend, config=config)
    assert [row["id"] for row in unfiltered["entries"]] == ["P-4"]  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("register", "tool", "arguments"),
    [
        # A 4.x daemon's memory_list_page has no ``types``: the call is refused, not answered unfiltered.
        (
            "old_list_page",
            "memory_list_page",
            {"namespace": _NS, "limit": 10, "types": ["decision"]},
        ),
        # A misspelt filter reaching a current daemon is refused the same way.
        ("current", "memory_recall", {"query": "", "namespace": _NS, "record_types": ["decision"]}),
        ("current", "memory_list_page", {"namespace": _NS, "limit": 10, "type": "decision"}),
    ],
)
async def test_a_daemon_that_does_not_know_a_filter_refuses_the_call(
    register: str, tool: str, arguments: dict[str, object]
) -> None:
    mcp = FastMCP("daemon-under-test")
    if register == "old_list_page":

        async def memory_list_page(
            namespace: str, limit: int = 500, after: dict[str, str] | None = None, status: str | None = None
        ) -> dict[str, object]:
            return {"status": "ok", "entries": [{"id": "UNFILTERED"}], "next": None}

        mcp.tool()(memory_list_page)
    else:
        register_recall_tool(mcp)
        register_list_page_tool(mcp)

    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="nexpected keyword argument"):
            await client.call_tool(tool, arguments)


def test_status_counts_each_type_exactly_and_a_flagged_row_still_counts(backend: SQLiteBackend) -> None:
    for index in range(30):
        backend.store(_row(f"D-{index:02d}", "decision", minutes=index))
    backend.store(_row("I-1", "incident"))
    # A row that merely carries the flag is an ordinary row (UF-MEM-15); the store's own canary is excluded by identity.
    flagged = _row("C-1", "decision")
    backend.store(flagged.model_copy(update={"metadata": {"system_canary": "true"}}))
    backend.store(_row("X-1", "decision").model_copy(update={"namespace": "project:other-00000000"}))

    health = memory_status_impl(_NS, backend=backend)["health"]

    assert health["types"] == {"decision": 31, "incident": 1}  # type: ignore[index]


async def test_the_sdk_client_filters_by_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    client = MemoryClient(namespace="default", mode="local")
    backend = client._get_backend()
    for index, kind in enumerate(("incident", "decision", "pattern")):
        backend.store(_row(f"{kind[0].upper()}-1", kind, minutes=index).model_copy(update={"namespace": "default"}))

    rows = await client.recall("ledger", include_org_memories=False, types=["decision"])

    assert [row["memory_id"] for row in rows] == ["D-1"]
    assert len(await client.recall("ledger", include_org_memories=False)) == 3


def test_a_shared_result_carries_its_type() -> None:
    from trw_memory._client_org_shared import shared_result_to_result

    assert shared_result_to_result({"id": "S-1", "content": "c", "type": "decision"})["type"] == "decision"  # type: ignore[typeddict-item]
    assert "type" not in shared_result_to_result({"id": "S-2", "content": "c"})
    # An SSE publish names its event in "type": that is not a memory type.
    assert "type" not in shared_result_to_result({"id": "S-3", "content": "c", "type": "learning_published"})


# -- FR05 -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected", "type_raw"),
    [
        ("decision", "decision", None),
        ("incident", "incident", None),
        ("", "pattern", None),
        (None, "pattern", None),
        (MemoryType.WORKAROUND, "workaround", None),
        ("retrospective", "pattern", "retrospective"),
    ],
)
def test_the_lenient_rule_degrades_only_an_unknown_type(raw: object, expected: str, type_raw: str | None) -> None:
    assert coerce_memory_type_lenient(raw) == (MemoryType(expected), type_raw)


def test_a_synced_row_of_an_unknown_type_is_stored_as_pattern(backend: SQLiteBackend, config: MemoryConfig) -> None:
    # What an older build does with a newer build's type: the same path any unknown value takes.
    wire = make_entry(entry_id="S-1", namespace=_NS, content="pulled", metadata={"k": "v"}).model_dump(mode="json")
    wire["type"] = "retrospective"

    answer = memory_sync_apply_impl(_NS, wire, backend=backend, config=config, if_revision=None)

    assert answer == {"status": "stored", "reason": ""}
    stored = backend.get("S-1", namespace=_NS)
    assert stored is not None and stored.type == "pattern"
    assert (stored.metadata["k"], stored.metadata["type_raw"]) == ("v", "retrospective")


@pytest.mark.parametrize("kind", [*_LEGACY, "decision"])
def test_a_synced_row_of_a_known_type_is_unchanged(backend: SQLiteBackend, config: MemoryConfig, kind: str) -> None:
    wire = make_entry(entry_id="S-2", namespace=_NS, content="pulled").model_dump(mode="json")
    wire["type"] = kind

    assert memory_sync_apply_impl(_NS, wire, backend=backend, config=config, if_revision=None)["status"] == "stored"
    stored = backend.get("S-2", namespace=_NS)
    assert stored is not None and stored.type == kind and "type_raw" not in stored.metadata


@pytest.mark.parametrize(("types", "sent"), [(None, False), (["decision"], True)])
def test_list_page_sends_types_only_when_set(types: list[str] | None, sent: bool) -> None:
    """A daemon started before the upgrade refuses an unknown argument, so an unfiltered page must not carry one."""
    import asyncio

    from trw_memory.daemon.client import DaemonClient

    captured: dict[str, object] = {}

    async def fake_call_tool(_self: object, tool: str, payload: dict[str, object]) -> dict[str, object]:
        captured.update(tool=tool, payload=payload)
        return {}

    client = DaemonClient.__new__(DaemonClient)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DaemonClient, "call_tool", fake_call_tool)
        asyncio.run(client.list_page("project:x", 10, None, types=types))

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert ("types" in payload) is sent
    assert payload.get("status", "absent") is None, "other optional args keep their declared default"
