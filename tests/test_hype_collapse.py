"""PRD-CORE-195 FR04 — _hype_collapse pure-function unit tests."""

from __future__ import annotations

from trw_memory._client_hype import hype_sibling_id, is_hype_id, parent_of_hype_id
from trw_memory.retrieval._hype_collapse import collapse_hype_ranking, hype_sibling_ids_in


def test_id_helpers_round_trip() -> None:
    sid = hype_sibling_id("parent-1", 2)
    assert sid == "parent-1#hype2"
    assert is_hype_id(sid)
    assert not is_hype_id("parent-1")
    assert parent_of_hype_id(sid) == "parent-1"
    assert parent_of_hype_id("parent-1") == "parent-1"  # non-hype passes through


def test_sibling_ids_in_filters_synthetic() -> None:
    stored = {"p1": [0.0], "p1#hype0": [0.0], "p2": [0.0], "p2#hype1": [0.0]}
    assert set(hype_sibling_ids_in(stored)) == {"p1#hype0", "p2#hype1"}
    assert hype_sibling_ids_in(None) == []


def test_collapse_maps_sibling_to_parent() -> None:
    ranking = [("p1#hype0", 0.9), ("p2", 0.5)]
    collapsed, hits = collapse_hype_ranking(ranking, {"p1", "p2"})
    assert collapsed == [("p1", 0.9), ("p2", 0.5)]
    assert hits == 1


def test_collapse_dedups_keeping_best_rank() -> None:
    # p1 reached via two siblings AND its own primary vector → counted once, at
    # its FIRST (best) appearance.
    ranking = [("p1#hype0", 0.95), ("p1", 0.90), ("p1#hype1", 0.80), ("p2", 0.50)]
    collapsed, hits = collapse_hype_ranking(ranking, {"p1", "p2"})
    assert collapsed == [("p1", 0.95), ("p2", 0.50)]
    assert hits == 2  # two of the four rows were #hype hits


def test_collapse_drops_orphan_parent() -> None:
    # A sibling whose parent is gone (forgotten/superseded) is dropped.
    ranking = [("gone#hype0", 0.99), ("p1", 0.4)]
    collapsed, hits = collapse_hype_ranking(ranking, {"p1"})
    assert collapsed == [("p1", 0.4)]
    assert hits == 1


def test_collapse_emits_no_synthetic_ids() -> None:
    ranking = [("p1#hype0", 0.9), ("p1#hype1", 0.8)]
    collapsed, _ = collapse_hype_ranking(ranking, {"p1"})
    assert all(not is_hype_id(eid) for eid, _ in collapsed)
