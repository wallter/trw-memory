"""PRD-CORE-279 FR07: the recall tool says its search population is bounded.

Measured 2026-09-17 on a namespace of 6500 rows with the default bound of 1000:
the oldest row was not in the candidate pool and was not returned, while the
same query found it at every smaller size. Raising the bound to 10000 found it
at 1045.8 ms warm instead of 139.6 ms. The bound stays; what changes is that a
caller can read about it, because an empty result from a truncated search is
otherwise indistinguishable from "not stored".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.recall import memory_recall_impl


async def test_recall_description_states_the_candidate_bound():
    """FR07: the MCP description names the cap, its ordering and the knob."""
    from trw_memory.server import mcp

    tool = await mcp.get_tool("memory_recall")
    description = tool.description or ""
    lowered = description.lower()

    assert "bounded" in lowered
    assert "most recently updated" in lowered
    assert "hybrid_search_candidate_pool_size" in description
    assert "does not mean the fact is absent" in lowered


def test_the_bound_is_what_the_description_says_it_is(tmp_path):
    """FR07: the disclosed formula is the one the code applies."""
    backend = SQLiteBackend(tmp_path / "m.db")
    try:
        base = datetime.now(timezone.utc) - timedelta(days=30)
        for index in range(12):
            stamp = base + timedelta(seconds=index)
            backend.store(
                MemoryEntry(
                    id=f"e{index}",
                    content=f"alpha note {index}",
                    namespace="project:default",
                    status=MemoryStatus.ACTIVE,
                    created_at=stamp,
                    updated_at=stamp,
                )
            )

        config = MemoryConfig()
        observed: list[int | None] = []
        original = backend.list_entries

        def _spy(**kwargs):
            observed.append(kwargs.get("limit"))
            return original(**kwargs)

        backend.list_entries = _spy  # type: ignore[method-assign]
        memory_recall_impl(
            "alpha",
            "project:default",
            backend=backend,
            limit=3,
            include_org_memories=False,
            config=config,
        )
        backend.list_entries = original  # type: ignore[method-assign]

        assert observed, "recall never loaded candidates"
        assert observed[0] == max(3 * 5, config.hybrid_search_candidate_pool_size)
    finally:
        backend.close()


def _seed_namespace(backend: SQLiteBackend, fillers: int) -> None:
    """One old needle, then ``fillers`` newer rows, all in project:default."""
    base = datetime.now(timezone.utc) - timedelta(days=30)
    backend.store(
        MemoryEntry(
            id="needle",
            content="the zarbaxos calibration constant is 7.41",
            namespace="project:default",
            status=MemoryStatus.ACTIVE,
            created_at=base,
            updated_at=base,
        )
    )
    for index in range(1, fillers + 1):
        stamp = base + timedelta(seconds=index)
        backend.store(
            MemoryEntry(
                id=f"f{index}",
                content=f"routine deployment note {index}",
                namespace="project:default",
                status=MemoryStatus.ACTIVE,
                created_at=stamp,
                updated_at=stamp,
            )
        )


def _recall_needle(backend: SQLiteBackend, config: MemoryConfig) -> tuple[list[bool], list[dict[str, object]]]:
    """Run the recall and report, per store scan, whether the needle was loaded."""
    pool: list[bool] = []
    original = backend.list_entries

    def _spy(**kwargs):
        entries = original(**kwargs)
        pool.append(any(entry.id == "needle" for entry in entries))
        return entries

    backend.list_entries = _spy  # type: ignore[method-assign]
    try:
        result = memory_recall_impl(
            "zarbaxos calibration constant",
            "project:default",
            backend=backend,
            limit=2,
            include_org_memories=False,
            config=config,
        )
    finally:
        backend.list_entries = original  # type: ignore[method-assign]
    return pool, list(result["memories"])


def test_entries_past_the_bound_are_not_searched(tmp_path, monkeypatch):
    """FR07: the truthful shape of the limit -- an old row outside both the
    store-scan bound and the tier index is not searched.

    Uses a small configured bound so the property is observable without
    building a 6500-row store; the mechanism is identical. The tier runtime
    is switched off here because its index is the documented carve-out (next
    test); MEMORY_STORAGE_PATH is pinned so the tier cache, keyed on it, can
    never resolve to the repo's own store and make the outcome depend on cwd.
    """
    monkeypatch.setenv("MEMORY_HYBRID_SEARCH_CANDIDATE_POOL_SIZE", "10")
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "store"))
    monkeypatch.setattr("trw_memory.tools.recall.supports_tier_runtime", lambda backend: False)
    backend = SQLiteBackend(tmp_path / "m.db")
    try:
        _seed_namespace(backend, fillers=39)
        config = MemoryConfig()
        assert config.hybrid_search_candidate_pool_size == 10

        pool, memories = _recall_needle(backend, config)

        assert pool and pool[0] is False, "the oldest row was inside the pool; the fixture is not exercising the bound"
        assert all(entry.get("id") != "needle" for entry in memories)
    finally:
        backend.close()


def test_the_tier_index_is_the_documented_carve_out(tmp_path, monkeypatch):
    """FR07: the description says the tier index is searched past the scan
    bound, and it is -- the same old row the bounded scan skips comes back
    through the tier runtime's first-warmup seed."""
    monkeypatch.setenv("MEMORY_HYBRID_SEARCH_CANDIDATE_POOL_SIZE", "10")
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "store"))
    backend = SQLiteBackend(tmp_path / "m.db")
    try:
        _seed_namespace(backend, fillers=39)
        config = MemoryConfig()
        assert 40 <= max(config.hot_max_entries * 8, 200), "fixture must sit inside the warmup seed"

        pool, memories = _recall_needle(backend, config)

        assert pool and pool[0] is False, "the store scan must still skip the needle"
        assert any(entry.get("id") == "needle" for entry in memories)
    finally:
        backend.close()


def test_a_large_limit_lifts_the_bound(tmp_path):
    """FR07: the disclosed max(limit * 5, pool) is what recall really asks for.

    Observed through the backend, not recomputed in the test: a limit large
    enough to beat the configured pool must widen the loaded population.
    """
    backend = SQLiteBackend(tmp_path / "m.db")
    try:
        base = datetime.now(timezone.utc) - timedelta(days=1)
        backend.store(
            MemoryEntry(
                id="only",
                content="alpha",
                namespace="project:default",
                status=MemoryStatus.ACTIVE,
                created_at=base,
                updated_at=base,
            )
        )
        config = MemoryConfig()
        observed: list[int | None] = []
        original = backend.list_entries

        def _spy(**kwargs):
            observed.append(kwargs.get("limit"))
            return original(**kwargs)

        backend.list_entries = _spy  # type: ignore[method-assign]
        big = config.hybrid_search_candidate_pool_size  # limit * 5 beats the pool
        memory_recall_impl(
            "alpha",
            "project:default",
            backend=backend,
            limit=big,
            include_org_memories=False,
            config=config,
        )
        backend.list_entries = original  # type: ignore[method-assign]
        assert observed[0] == big * 5, f"recall asked for {observed[0]}, not limit*5"
    finally:
        backend.close()
