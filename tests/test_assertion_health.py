"""The session-start assertion summary counts the cached verdicts where the rows live (PRD-CORE-086 FR07).

The stale window is the caller's ``stale_days``, the same knob the maintain-verify
sweep reads, so session start and maintenance classify one store identically
(PRD-CORE-263-FR08).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from trw_memory.lifecycle.verification_pass import assertion_health
from trw_memory.models.memory import Assertion, AssertionType, MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _assertion(result: bool | None, days_ago: float | None) -> Assertion:
    moment = None if days_ago is None else datetime.now(timezone.utc) - timedelta(days=days_ago)
    return Assertion(type=AssertionType.GLOB_EXISTS, target="source.py", last_result=result, last_verified_at=moment)


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    yield store
    store.close()


def _put(backend: SQLiteBackend, entry_id: str, *assertions: Assertion, namespace: str = "default") -> None:
    backend.store(MemoryEntry(id=entry_id, content=entry_id, namespace=namespace, assertions=list(assertions)))


def test_each_assertion_is_counted_under_its_cached_state(backend: SQLiteBackend) -> None:
    _put(backend, "L-a", _assertion(True, 0.1), _assertion(None, None))  # passing; never verified
    _put(backend, "L-b", _assertion(False, 0.1), _assertion(True, 10))  # failing; verified too long ago
    _put(backend, "L-c", _assertion(None, 0.1))  # checked recently, no verdict
    _put(backend, "L-plain")  # no assertions: not a row the summary counts

    health = assertion_health(backend, namespace="default", stale_days=7)

    assert health == {"passing": 1, "failing": 1, "stale": 2, "unverifiable": 1, "total": 3}


def test_the_stale_window_is_the_callers(backend: SQLiteBackend) -> None:
    _put(backend, "L-ten", _assertion(True, 10))

    assert assertion_health(backend, namespace="default", stale_days=30) == {
        "passing": 1,
        "failing": 0,
        "stale": 0,
        "unverifiable": 0,
        "total": 1,
    }
    assert assertion_health(backend, namespace="default", stale_days=7)["stale"] == 1  # type: ignore[index]


def test_a_namespace_without_assertion_rows_has_no_summary(backend: SQLiteBackend) -> None:
    _put(backend, "L-other", _assertion(True, 0.1), namespace="project:other-11111111")
    _put(backend, "L-plain")

    assert assertion_health(backend, namespace="default", stale_days=7) is None


@pytest.mark.parametrize("stale_days", [0, True, "7", 36_501, 10**9])
def test_the_tool_refuses_a_stale_window_that_is_not_a_positive_int(stale_days: object) -> None:
    from trw_memory.tools.verify import register_verify_tool

    tools: dict[str, object] = {}

    class _Captured:
        def tool(self) -> object:
            return lambda fn: tools.setdefault(fn.__name__, fn)

    register_verify_tool(_Captured())  # type: ignore[arg-type]
    answer = asyncio.run(tools["memory_assertion_health"](namespace="default", stale_days=stale_days))  # type: ignore[operator]

    assert answer["status"] == "invalid", answer


async def test_over_the_wire_a_grant_reads_its_own_namespace_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.daemon import DaemonClient, DaemonPaths
    from trw_memory.daemon._grants import mint_grant

    from .test_daemon_server import _await_discovery, _spawn_daemon

    user_dir = tmp_path / "userhome"
    monkeypatch.setenv("TRW_USER_DIR", str(user_dir))
    paths = DaemonPaths.resolve()
    paths.store.parent.mkdir(parents=True, exist_ok=True)
    seeded = SQLiteBackend(paths.store)
    _put(seeded, "L-mine", _assertion(True, 0.1), namespace="project:mine-aaaaaaaa")
    _put(seeded, "L-theirs", _assertion(False, 0.1), _assertion(None, None), namespace="project:theirs-bbbbbbbb")
    seeded.close()
    client = DaemonClient(mint_grant(paths, ["project:mine-aaaaaaaa"]), paths=paths)
    proc = _spawn_daemon(user_dir)
    try:
        _await_discovery(paths, proc)
        mine = await client.assertion_health("project:mine-aaaaaaaa", 7)
        with pytest.raises(ToolError, match="theirs"):
            await client.assertion_health("project:theirs-bbbbbbbb", 7)
    finally:
        proc.kill()
        proc.wait(timeout=30)

    assert mine == {"status": "ok", "health": {"passing": 1, "failing": 0, "stale": 0, "unverifiable": 0, "total": 1}}
