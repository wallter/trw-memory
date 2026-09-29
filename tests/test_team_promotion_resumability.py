"""PRD-CORE-331 B71-135(c): a team promotion interrupted mid-run (the daemon lane's statement
deadline, or any exception between two stores) and rerun must never re-store a row it already
promoted -- that is a possible lost update (an operator edit to the landed row) or, at minimum,
wasted duplicate work on every rerun until the namespace finally completes in one uninterrupted
pass."""

from __future__ import annotations

import pytest

from trw_memory.tools import consolidate as consolidate_module

from ._test_team_memory_support import _InMemoryBackend, _make_entry


def test_a_promotion_interrupted_and_rerun_never_overwrites_an_already_promoted_row(monkeypatch):
    backend = _InMemoryBackend()
    for i in range(4):
        backend.store(_make_entry(f"e{i}", importance=0.9))

    calls = {"n": 0}
    real_store = backend.store

    def flaky_store(entry):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated lane statement-deadline interruption")
        real_store(entry)

    monkeypatch.setattr(backend, "store", flaky_store)
    with pytest.raises(RuntimeError):
        consolidate_module._promote_team_memories("team:sprint-37", backend)

    promoted_before = backend.list_entries(namespace="project:default", limit=10)
    assert len(promoted_before) == 1

    # Simulate an operator edit to the row that DID land, between the interrupted pass and the rerun.
    edited = promoted_before[0].model_copy(update={"content": "operator-edited"})
    real_store(edited)

    result = consolidate_module._promote_team_memories("team:sprint-37", backend)

    still_there = backend.get(edited.id, namespace="project:default")
    assert still_there is not None and still_there.content == "operator-edited", (
        "the rerun re-stored an already-promoted row and clobbered an edit made to it -- a lost update"
    )
    assert result["promoted_count"] == 3, "the rerun must only report the rows IT newly promoted"
    assert len(backend.list_entries(namespace="project:default", limit=10)) == 4
