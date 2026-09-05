"""Vector-clock monotonicity on local update + downstream conflict resolution.

PRD-CORE-047 FR04: a node's vector-clock counter must advance on every local
edit so causality is expressible. Before the fix, ``_client_store`` reset the
clock to ``init_clock(node)`` ({node: 1}) on update, so an entry edited N times
still reported ``{node: 1}`` -- causality went nowhere and ``resolve_conflict``
mis-classified a locally-newer entry as concurrent (lossy merge) instead of a
clean local win.
"""

from __future__ import annotations

from trw_memory.client import MemoryClient
from trw_memory.sync.conflict import compare_clocks, resolve_conflict


async def test_local_update_advances_vector_clock(memory_client: MemoryClient) -> None:
    """Re-storing the same entry id must advance the local node's counter."""
    node = memory_client._local_node_id

    stored = await memory_client.store("v1 content", importance=0.8, entry_id="M-clock-1")
    backend = memory_client._get_backend()
    after_create = backend.get(stored["memory_id"], namespace="default")
    assert after_create is not None
    assert after_create.vector_clock == {node: 1}

    await memory_client.store("v2 content", importance=0.8, entry_id="M-clock-1")
    after_update_1 = backend.get("M-clock-1", namespace="default")
    assert after_update_1 is not None

    await memory_client.store("v3 content", importance=0.8, entry_id="M-clock-1")
    after_update_2 = backend.get("M-clock-1", namespace="default")
    assert after_update_2 is not None

    # The counter must strictly increase on each edit, not reset to 1.
    assert after_update_1.vector_clock[node] == 2
    assert after_update_2.vector_clock[node] == 3
    await memory_client.close()


async def test_locally_updated_entry_wins_conflict_over_stale_remote(
    memory_client: MemoryClient,
) -> None:
    """A locally re-edited entry causally dominates the version that was pushed.

    Mirrors the live trw-mcp pull path: a local entry is published (its clock
    is the snapshot the backend stored), then edited locally a few more times,
    then pulled back. ``resolve_conflict(local, remote_snapshot)`` must return
    the local entry (a_wins) -- never a lossy concurrent merge.
    """
    node = memory_client._local_node_id
    backend = memory_client._get_backend()

    await memory_client.store("original", importance=0.8, entry_id="M-clock-2")
    # Snapshot the clock as it would have been pushed to the backend at v1.
    remote_snapshot = backend.get("M-clock-2", namespace="default")
    assert remote_snapshot is not None

    # Local keeps editing after the push snapshot.
    await memory_client.store("edited locally", importance=0.8, entry_id="M-clock-2")
    local_latest = backend.get("M-clock-2", namespace="default")
    assert local_latest is not None

    assert compare_clocks(local_latest.vector_clock, remote_snapshot.vector_clock) == "a_wins"
    resolved = resolve_conflict(local_latest, remote_snapshot)
    assert resolved.id == local_latest.id
    assert resolved.content == "edited locally"
    # A clean causal win does not stamp a conflict-merge record.
    assert not any("conflict_merged" in event for event in resolved.outcome_history)
    assert resolved.vector_clock[node] == 2
    await memory_client.close()
