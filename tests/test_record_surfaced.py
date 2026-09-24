"""memory_record_surfaced counts exactly what a caller showed (W1-d; PRD-FIX-104 FR01/FR02).

trw-mcp's recall pages over-fetch and then admit, cap and budget; counting the
page would inflate the scores of rows nobody saw. So trw-mcp recalls with
``record_access=False`` and reports the rows it surfaced here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.tools.recall_support import memory_record_surfaced_impl


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    for entry_id in ("L-a", "L-b"):
        store.store(MemoryEntry(id=entry_id, content=f"pooling note {entry_id}"))
    yield store
    store.close()


def _counts(backend: SQLiteBackend, entry_id: str) -> tuple[int, int, int]:
    entry = backend.get(entry_id, namespace="default")
    assert entry is not None
    return entry.access_count, entry.recall_count, entry.session_count


def test_each_surfaced_id_this_store_holds_is_counted_once(backend: SQLiteBackend) -> None:
    answer = memory_record_surfaced_impl("default", ["L-a", "L-a", "L-missing"], False, backend=backend)

    assert answer == {"status": "ok", "counted": ["L-a"]}
    assert _counts(backend, "L-a") == (1, 1, 0)
    assert _counts(backend, "L-b") == (0, 0, 0)


def test_a_session_start_surface_also_counts_the_session(backend: SQLiteBackend) -> None:
    memory_record_surfaced_impl("default", ["L-a"], True, backend=backend)

    assert _counts(backend, "L-a") == (1, 1, 1)


def test_a_failed_session_count_leaves_no_access_count_behind(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> int:
        raise StorageError("session count failed")

    monkeypatch.setattr(backend, "increment_session_counts", boom)

    with pytest.raises(StorageError):
        memory_record_surfaced_impl("default", ["L-a"], True, backend=backend)

    assert _counts(backend, "L-a") == (0, 0, 0)


def test_a_recall_that_does_not_record_access_leaves_its_page_uncounted(backend: SQLiteBackend) -> None:
    config = MemoryConfig(storage_path=str(Path(backend.db_path).parent))
    shown = memory_recall_impl("pooling", "default", backend=backend, config=config, record_access=False)
    memory_recall_impl("pooling", "default", backend=backend, config=config)

    assert len(shown["memories"]) == 2  # type: ignore[arg-type]
    assert _counts(backend, "L-a")[0] == 1  # only the recording recall counted


@pytest.mark.parametrize("ids", [[], [f"L-{i}" for i in range(1001)]])
def test_the_tool_refuses_an_empty_or_oversized_id_list(ids: list[str]) -> None:
    from trw_memory.tools.recall_support import register_recall_support_tools

    tools: dict[str, object] = {}

    class _Captured:
        def tool(self) -> object:
            return lambda fn: tools.setdefault(fn.__name__, fn)

    register_recall_support_tools(_Captured())  # type: ignore[arg-type]
    answer = asyncio.run(tools["memory_record_surfaced"](namespace="default", ids=ids))  # type: ignore[operator]

    assert answer["status"] == "invalid", answer


async def test_over_the_wire_a_grant_counts_its_own_namespace_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write is refused for an ungranted namespace, and the same id there is never counted."""
    from fastmcp.exceptions import ToolError

    from trw_memory.daemon import DaemonClient, DaemonPaths
    from trw_memory.daemon._grants import mint_grant

    from .test_daemon_server import _await_discovery, _spawn_daemon

    user_dir = tmp_path / "userhome"
    monkeypatch.setenv("TRW_USER_DIR", str(user_dir))
    paths = DaemonPaths.resolve()
    paths.store.parent.mkdir(parents=True, exist_ok=True)
    seeded = SQLiteBackend(paths.store)
    for namespace in ("project:mine-aaaaaaaa", "project:theirs-bbbbbbbb"):
        seeded.store(MemoryEntry(id="L-same", content="same id", namespace=namespace))
    seeded.close()
    client = DaemonClient(mint_grant(paths, ["project:mine-aaaaaaaa"]), paths=paths)
    proc = _spawn_daemon(user_dir)
    try:
        _await_discovery(paths, proc)
        mine = await client.record_surfaced("project:mine-aaaaaaaa", ["L-same"], session_start=True)
        with pytest.raises(ToolError, match="theirs"):
            await client.record_surfaced("project:theirs-bbbbbbbb", ["L-same"], session_start=True)
    finally:
        proc.kill()
        proc.wait(timeout=30)

    assert mine == {"status": "ok", "counted": ["L-same"]}
    after = SQLiteBackend(paths.store)
    try:
        counted = after.get("L-same", namespace="project:mine-aaaaaaaa")
        untouched = after.get("L-same", namespace="project:theirs-bbbbbbbb")
    finally:
        after.close()
    assert counted is not None and untouched is not None
    assert (counted.access_count, counted.recall_count, counted.session_count) == (1, 1, 1)
    assert (untouched.access_count, untouched.recall_count, untouched.session_count) == (0, 0, 0)
