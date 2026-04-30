"""Tests for trw_memory.sync conflict helpers — PRD-CORE-047."""

from __future__ import annotations

from trw_memory.sync.conflict import (
    MAX_MERGED_DETAIL_LENGTH,
    compare_clocks,
    increment_clock,
    init_clock,
    merge_clocks,
    resolve_conflict,
)

from ._test_sync_support import make_sync_entry as _make_entry


class TestCompareClocks:
    """FR04: compare_clocks returns correct causal ordering."""

    def test_a_wins_when_strictly_dominates(self) -> None:
        """a_wins when a >= b on all keys and > on at least one."""
        result = compare_clocks({"A": 3, "B": 1}, {"A": 2, "B": 1})
        assert result == "a_wins"

    def test_b_wins_when_strictly_dominates(self) -> None:
        """b_wins when b >= a on all keys and > on at least one."""
        result = compare_clocks({"A": 1, "B": 1}, {"A": 2, "B": 1})
        assert result == "b_wins"

    def test_concurrent_when_neither_dominates(self) -> None:
        """concurrent when a > b on some key and b > a on another."""
        result = compare_clocks({"A": 2, "B": 1}, {"A": 1, "B": 2})
        assert result == "concurrent"

    def test_concurrent_when_clocks_are_equal(self) -> None:
        """Equal clocks are concurrent (not a win for either)."""
        result = compare_clocks({"A": 3}, {"A": 3})
        assert result == "concurrent"

    def test_concurrent_when_both_empty(self) -> None:
        """Empty clocks are concurrent."""
        result = compare_clocks({}, {})
        assert result == "concurrent"

    def test_handles_missing_keys_default_zero(self) -> None:
        """Missing keys default to 0 in comparison."""
        result = compare_clocks({"A": 1, "B": 1}, {"A": 1, "C": 1})
        assert result == "concurrent"

    def test_a_wins_with_superset_keys(self) -> None:
        """a_wins when a has all keys of b plus extra with > values."""
        result = compare_clocks({"A": 2, "B": 1}, {"A": 1})
        assert result == "a_wins"

    def test_b_wins_with_superset_keys(self) -> None:
        """b_wins when b has all keys of a plus extra with > values."""
        result = compare_clocks({"A": 1}, {"A": 2, "B": 1})
        assert result == "b_wins"

    def test_single_node_a_wins(self) -> None:
        """Single node: a_wins when a[node] > b[node]."""
        result = compare_clocks({"X": 5}, {"X": 3})
        assert result == "a_wins"


class TestInitClock:
    """FR04: init_clock creates initial vector clock."""

    def test_creates_clock_with_counter_one(self) -> None:
        """New clock has the node_id with counter 1."""
        clock = init_clock("node-abc")
        assert clock == {"node-abc": 1}

    def test_does_not_modify_original(self) -> None:
        """Init clock returns a new dict each time."""
        c1 = init_clock("node-1")
        c2 = init_clock("node-1")
        assert c1 == c2
        assert c1 is not c2


class TestIncrementClock:
    """FR04: increment_clock increments the node's counter."""

    def test_increments_existing_counter(self) -> None:
        """Incrementing an existing node increases its counter by 1."""
        clock = increment_clock({"node-1": 3, "node-2": 1}, "node-1")
        assert clock == {"node-1": 4, "node-2": 1}

    def test_adds_new_node_to_existing_clock(self) -> None:
        """Incrementing a new node adds it with counter 1."""
        clock = increment_clock({"node-1": 3}, "node-2")
        assert clock == {"node-1": 3, "node-2": 1}

    def test_does_not_mutate_original(self) -> None:
        """Returns a new dict without modifying the original."""
        original = {"node-1": 3}
        result = increment_clock(original, "node-1")
        assert original == {"node-1": 3}
        assert result == {"node-1": 4}


class TestMergeClocks:
    """FR04: merge_clocks takes max of each counter."""

    def test_merges_by_taking_max(self) -> None:
        """Max of each node's counter across both clocks."""
        result = merge_clocks({"A": 3, "B": 1}, {"A": 1, "B": 2})
        assert result == {"A": 3, "B": 2}

    def test_merges_disjoint_keys(self) -> None:
        """Disjoint keys included with their values."""
        result = merge_clocks({"A": 1}, {"B": 2})
        assert result == {"A": 1, "B": 2}

    def test_merges_empty_clocks(self) -> None:
        """Merging empty clocks yields empty clock."""
        result = merge_clocks({}, {})
        assert result == {}

    def test_merge_with_one_empty(self) -> None:
        """Merging with an empty clock returns the non-empty clock's values."""
        result = merge_clocks({"A": 5}, {})
        assert result == {"A": 5}


class TestResolveConflict:
    """FR05: resolve_conflict handles causal order and concurrent merge."""

    def test_a_wins_returns_local(self) -> None:
        """When local clock dominates, local entry is returned unchanged."""
        local = _make_entry(entry_id="L-1", content="local content", vector_clock={"A": 3, "B": 1})
        remote = _make_entry(entry_id="R-1", content="remote content", vector_clock={"A": 2, "B": 1})
        result = resolve_conflict(local, remote)
        assert result.id == "L-1"
        assert result.content == "local content"

    def test_b_wins_returns_remote(self) -> None:
        """When remote clock dominates, remote entry is returned."""
        local = _make_entry(entry_id="L-1", content="local content", vector_clock={"A": 1, "B": 1})
        remote = _make_entry(entry_id="R-1", content="remote content", vector_clock={"A": 2, "B": 1})
        result = resolve_conflict(local, remote)
        assert result.id == "R-1"
        assert result.content == "remote content"

    def test_concurrent_merges_detail_with_separator(self) -> None:
        """Concurrent clocks: details concatenated with separator."""
        local = _make_entry(entry_id="L-1", detail="local detail", vector_clock={"A": 2, "B": 1})
        remote = _make_entry(entry_id="R-1", detail="remote detail", vector_clock={"A": 1, "B": 2})
        result = resolve_conflict(local, remote)
        assert "local detail" in result.detail
        assert "remote detail" in result.detail
        assert "\n\n---\n\n" in result.detail

    def test_concurrent_takes_max_importance(self) -> None:
        """Concurrent: importance = max(local, remote)."""
        local = _make_entry(importance=0.6, vector_clock={"A": 2, "B": 1})
        remote = _make_entry(importance=0.9, vector_clock={"A": 1, "B": 2})
        result = resolve_conflict(local, remote)
        assert result.importance == 0.9

    def test_concurrent_unions_tags_sorted(self) -> None:
        """Concurrent: tags = sorted union of both tag sets."""
        local = _make_entry(tags=["python", "testing"], vector_clock={"A": 2, "B": 1})
        remote = _make_entry(tags=["testing", "deployment"], vector_clock={"A": 1, "B": 2})
        result = resolve_conflict(local, remote)
        assert result.tags == ["deployment", "python", "testing"]

    def test_concurrent_merges_clocks(self) -> None:
        """Concurrent: vector_clock = max of each counter."""
        local = _make_entry(vector_clock={"A": 2, "B": 1})
        remote = _make_entry(vector_clock={"A": 1, "B": 2})
        result = resolve_conflict(local, remote)
        assert result.vector_clock == {"A": 2, "B": 2}

    def test_merged_detail_truncated_to_max_length(self) -> None:
        """Concurrent: merged detail truncated to MAX_MERGED_DETAIL_LENGTH."""
        local = _make_entry(detail="x" * 1500, vector_clock={"A": 2, "B": 1})
        remote = _make_entry(detail="y" * 1500, vector_clock={"A": 1, "B": 2})
        result = resolve_conflict(local, remote)
        assert len(result.detail) <= MAX_MERGED_DETAIL_LENGTH

    def test_adds_conflict_merged_to_outcome_history(self) -> None:
        """Concurrent merge adds a conflict_merged record to outcome_history."""
        local = _make_entry(
            entry_id="L-1",
            vector_clock={"A": 2, "B": 1},
            outcome_history=["existing-event"],
        )
        remote = _make_entry(entry_id="R-1", vector_clock={"A": 1, "B": 2})
        result = resolve_conflict(local, remote)
        assert len(result.outcome_history) >= 2
        conflict_entry = result.outcome_history[-1]
        assert "conflict_merged" in conflict_entry
        assert "L-1" in conflict_entry
        assert "R-1" in conflict_entry

    def test_concurrent_preserves_local_content(self) -> None:
        """Concurrent merge uses local content (preferred)."""
        local = _make_entry(content="local preferred", vector_clock={"A": 2, "B": 1})
        remote = _make_entry(content="remote content", vector_clock={"A": 1, "B": 2})
        result = resolve_conflict(local, remote)
        assert result.content == "local preferred"

    def test_concurrent_unions_merged_from(self) -> None:
        """Concurrent merge records the participating vector-clock nodes."""
        local = _make_entry(merged_from=["src-1"], vector_clock={"A": 2, "B": 1})
        remote = _make_entry(merged_from=["src-2"], vector_clock={"A": 1, "B": 2})
        result = resolve_conflict(local, remote)
        assert result.merged_from == ["A", "B"]

    def test_concurrent_same_detail_no_double(self) -> None:
        """When local and remote detail are equal, don't duplicate."""
        local = _make_entry(detail="same detail", vector_clock={"A": 2, "B": 1})
        remote = _make_entry(detail="same detail", vector_clock={"A": 1, "B": 2})
        result = resolve_conflict(local, remote)
        assert result.detail == "same detail"
