"""The forced graph sweep builds the edges a pre-wiring corpus never got, one resumable page at a time (F5-B).

trw-mcp keeps the resume point and calls ``memory_graph_backfill`` over the daemon;
the page itself -- enrichment, the canary skip, the deadline and fail-open per row
-- is pinned here against a real store.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trw_memory.graph import backfill_graph_page
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import EntryCursor
from trw_memory.storage.sqlite_backend import SQLiteBackend


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    """A consolidation-linked corpus written straight to the store, so nothing graphed it.

    Entries 2 and 3 name entry 1 as a consolidation source: each swept entry has one
    materialised edge to build, deterministically (no embedder, no similarity threshold).
    """
    store = SQLiteBackend(tmp_path / "memory.db")
    base = datetime(2026, 9, 3, tzinfo=timezone.utc)
    store.store(
        MemoryEntry(id="L-bf-1", content="Postgres connection pooling tuning", created_at=base, updated_at=base)
    )
    for index in (2, 3):
        moment = base + timedelta(minutes=index)
        store.store(
            MemoryEntry(
                id=f"L-bf-{index}",
                content=f"Postgres note {index}",
                created_at=moment,
                updated_at=moment,
                consolidated_from=["L-bf-1"],
            )
        )
    yield store
    store.close()


def _edges(backend: SQLiteBackend) -> int:
    return int(
        backend._conn.execute("SELECT COUNT(*) FROM memory_graph_edges WHERE edge_type = 'consolidation'").fetchone()[0]
    )


def test_a_page_builds_the_edges_the_corpus_never_got(backend: SQLiteBackend) -> None:
    page = backfill_graph_page(backend, "default", after=None, limit=10)

    assert (page["processed"], page["edges_built"], page["complete"]) == (3, 2, True)
    assert _edges(backend) == 2


def test_bounded_pages_resume_after_the_last_row_and_visit_each_once(backend: SQLiteBackend) -> None:
    after, seen = None, []
    for _ in range(3):
        page = backfill_graph_page(backend, "default", after=after, limit=1)
        assert (page["processed"], page["complete"]) == (1, False)
        after = EntryCursor(**page["next"])
        seen.append(after.entry_id)

    assert sorted(seen) == ["L-bf-1", "L-bf-2", "L-bf-3"]
    assert backfill_graph_page(backend, "default", after=after, limit=1)["complete"] is True
    assert _edges(backend) == 2


def test_a_spent_deadline_reads_nothing_and_claims_no_progress(backend: SQLiteBackend) -> None:
    page = backfill_graph_page(backend, "default", after=None, limit=10, deadline_seconds=0.0)

    assert (page["processed"], page["next"], page["complete"]) == (0, None, False)


def test_a_failing_row_is_counted_and_passed(backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> dict[str, int]:
        raise RuntimeError("graph enrichment exploded")

    monkeypatch.setattr("trw_memory.graph.update_entry_graph", boom)

    page = backfill_graph_page(backend, "default", after=None, limit=10)

    assert (page["failed"], page["edges_built"], page["complete"]) == (3, 0, True)


def test_no_edge_touches_a_canary_whichever_row_proposes_it(backend: SQLiteBackend) -> None:
    backend.store(MemoryEntry(id="L-canary", content="canary", metadata={"system_canary": "true"}))
    backend.store(MemoryEntry(id="L-lure", content="consolidates the canary", consolidated_from=["L-canary"]))

    page = backfill_graph_page(backend, "default", after=None, limit=10)

    # The lure's refused edge is not reported as built: the count is the table's truth.
    assert (page["processed"], page["skipped"], page["edges_built"]) == (4, 1, 2)
    touching = backend._conn.execute(
        "SELECT COUNT(*) FROM memory_graph_edges WHERE 'L-canary' IN (source_id, target_id)"
    ).fetchone()[0]
    assert touching == 0
    assert _edges(backend) == 2  # the corpus's own consolidation edges still land


def test_reading_the_page_counts_against_the_deadline(backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    listed = backend.list_entries

    def slow_listing(**kwargs: object) -> object:
        time.sleep(0.05)
        return listed(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(backend, "list_entries", slow_listing)

    page = backfill_graph_page(backend, "default", after=None, limit=10, deadline_seconds=0.01)

    assert (page["processed"], page["complete"]) == (0, False)


@pytest.mark.parametrize(
    "arguments", [{"limit": 0}, {"limit": 10_001}, {"deadline_seconds": -1.0}, {"after": {"entry_id": "x"}}]
)
def test_the_tool_refuses_an_unbounded_page_or_a_malformed_cursor(arguments: dict[str, object]) -> None:
    from trw_memory.tools.maintain import register_maintain_tool

    tools: dict[str, object] = {}

    class _Captured:
        def tool(self) -> object:
            return lambda fn: tools.setdefault(fn.__name__, fn)

    register_maintain_tool(_Captured())  # type: ignore[arg-type]
    answer = asyncio.run(tools["memory_graph_backfill"](namespace="default", **arguments))  # type: ignore[operator]

    assert answer["status"] == "invalid", answer
