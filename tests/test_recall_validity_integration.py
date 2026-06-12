"""PRD-CORE-194 FR03 — validity-aware recall through MemoryClient.recall.

End-to-end: a superseded record is excluded from default recall and re-included
(ranked below open) with include_superseded=True; as_of re-scopes the window.
"""

from __future__ import annotations

from datetime import datetime, timezone

from trw_memory.client import MemoryClient
from trw_memory.models.memory import MemoryEntry

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 2, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 3, tzinfo=timezone.utc)


def _store_ab(client: MemoryClient) -> None:
    """A (superseded by B, window [T0,T2)) and B (open, opens at T2)."""
    backend = client._get_backend()
    backend.store(
        MemoryEntry(
            id="A",
            content="rollback git stash safe",
            created_at=T0,
            valid_from=T0,
            invalid_from=T2,
            invalidated_by="B",
        )
    )
    backend.store(
        MemoryEntry(id="B", content="rollback git stash safe", created_at=T2, valid_from=T2)
    )


async def test_recall_excludes_superseded_by_default(memory_client: MemoryClient) -> None:
    client = memory_client
    _store_ab(client)
    results = await client.recall("rollback git stash", limit=10)
    ids = {r["memory_id"] for r in results}
    assert "B" in ids
    assert "A" not in ids


async def test_recall_include_superseded_ranks_below_open(memory_client: MemoryClient) -> None:
    client = memory_client
    _store_ab(client)
    results = await client.recall("rollback git stash", limit=10, include_superseded=True)
    ids = [r["memory_id"] for r in results]
    assert "A" in ids and "B" in ids
    # Open B is ranked strictly above superseded A (positional penalty).
    assert ids.index("B") < ids.index("A")


async def test_recall_as_of_reincludes_in_window(memory_client: MemoryClient) -> None:
    client = memory_client
    _store_ab(client)
    # as_of=T1 is inside A's [T0,T2) window; B (opens at T2) is out of window.
    results = await client.recall("rollback git stash", limit=10, as_of=T1)
    ids = {r["memory_id"] for r in results}
    assert "A" in ids
    assert "B" not in ids
