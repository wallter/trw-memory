"""Tests for Graph-Augmented Recall (GAR) — frontier-001.

NOTE: the knowledge-graph edge-insertion primitive is ``_upsert_edge``
(``trw_memory.graph._upsert_edge``), not a hypothetical
``update_entry_graph(conn, src, dst, ...)`` helper. ``update_entry_graph``
is a *whole-backend enrichment* pass with a different signature
(``update_entry_graph(entry, backend, *, embedding, config)``).  These
tests insert edges directly with ``_upsert_edge`` using a VALID edge
type from ``VALID_EDGE_TYPES`` (e.g. ``related_to`` — ``related`` is
not a valid type and ``_upsert_edge`` rejects it).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trw_memory.graph import _upsert_edge
from trw_memory.retrieval.recall_selection import LocalCandidate


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TestMemoryClientRecallGraphExpansion:
    """Integration tests for MemoryClient.recall(include_graph_expansion=...)."""

    @pytest.mark.parametrize(
        ("include_graph_expansion", "neighbour_expected"),
        [(True, True), (False, False)],
        ids=["expansion-on", "expansion-off"],
    )
    async def test_graph_expansion_gates_neighbours_only(
        self,
        memory_client,
        monkeypatch: pytest.MonkeyPatch,
        include_graph_expansion: bool,
        neighbour_expected: bool,
    ) -> None:
        """A graph neighbour of a hit surfaces only with expansion on; a non-neighbour never does.

        On a three-row store every row is a primary-search candidate, which would
        hide the graph step. The primary acquisition is therefore pinned to the
        anchor alone, so the neighbour can only arrive through its graph edge;
        everything after acquisition (expansion, merge, finish) runs for real.
        """
        await memory_client.store("zzqqxx unique anchor token", tags=["anchor"])
        await memory_client.store("totally different vocabulary here", tags=["other"])
        await memory_client.store("unrelated orchard pruning notes", tags=["island"])

        backend = memory_client._backend
        namespace = memory_client._namespace
        entries = {e.content: e for e in backend.list_entries(namespace=namespace)}
        anchor = entries["zzqqxx unique anchor token"]
        neighbour_id = entries["totally different vocabulary here"].id
        non_neighbour_id = entries["unrelated orchard pruning notes"].id
        _upsert_edge(backend._conn, anchor.id, neighbour_id, "related_to", 1.0, _now(), namespace=namespace)

        async def _anchor_only(*_args: object, **_kwargs: object) -> list[LocalCandidate]:
            return [LocalCandidate(anchor, 1.0)]

        monkeypatch.setattr(memory_client, "_try_hybrid_recall", _anchor_only)

        result = await memory_client.recall("zzqqxx", include_graph_expansion=include_graph_expansion)
        ids = {r["memory_id"] for r in result}

        assert anchor.id in ids, "the primary hit must always be returned"
        assert (neighbour_id in ids) is neighbour_expected
        assert non_neighbour_id not in ids, "an entry with no edge to a hit must never be expanded in"
