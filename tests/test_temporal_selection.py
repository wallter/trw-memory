"""Canonical temporal selection before candidate caps; no storage mocks."""

from datetime import datetime, timedelta, timezone

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.temporal_selection import TemporalSelection
from trw_memory.retrieval.validity_prior import apply_validity_prior

T = datetime(2022, 1, 1, tzinfo=timezone.utc)


def entry(name: str, *, closed: bool = False) -> MemoryEntry:
    return MemoryEntry(
        id=name,
        content=name,
        valid_from=T - timedelta(days=1),
        invalid_from=T if closed else None,
        invalidated_by="replacement" if closed else None,
    )


@pytest.mark.parametrize("include", [False, True])
def test_select_past_ineligible_prefix(include: bool) -> None:
    old = [entry(f"old-{i}", closed=True) for i in range(20)]
    current = entry("current")
    selected = TemporalSelection(include_superseded=include).select(iter([*old, current]), limit=3)
    assert selected == ([current, *old[:2]] if include else [current])


def test_stops_consuming_at_eligible_limit() -> None:
    def source():
        yield entry("one")
        yield entry("two")
        raise AssertionError("Selector consumed beyond its eligible limit")

    assert len(TemporalSelection().select(source(), limit=2)) == 2


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        TemporalSelection().select([], limit=limit)


def test_frozen_expiry_reference_and_default_open_semantics() -> None:
    expiring = entry("expiry").model_copy(update={"expires": "2022-01-01"})
    assert TemporalSelection(reference_time=T).eligible(expiring)
    assert not TemporalSelection(reference_time=T + timedelta(days=1)).eligible(expiring)
    future = entry("future").model_copy(update={"valid_from": T + timedelta(days=1)})
    assert TemporalSelection(reference_time=T).eligible(future)  # Existing no-as-of policy.
    assert not TemporalSelection(as_of=T).eligible(future)


@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("include", [False, True])
def test_microsecond_window_matches_prior(delta: int, include: bool) -> None:
    at = T + timedelta(microseconds=delta)
    entries = [entry("closed", closed=True), entry("open")]
    selection = TemporalSelection(as_of=at, include_superseded=include)
    assert selection.select(entries, limit=3) == apply_validity_prior(entries, as_of=at, include_superseded=include)


def test_canary_exclusion_precedes_limit_even_when_superseded_included() -> None:
    canary = entry("canary").model_copy(update={"metadata": {"system_canary": "true"}})
    useful = entry("useful")
    assert TemporalSelection().select([canary, useful], limit=1) == [canary]
    selection = TemporalSelection(include_superseded=True, exclude_system_canaries=True)
    assert selection.select([canary, useful], limit=1) == [useful]


def test_ranked_selection_retains_bounded_live_models() -> None:
    import weakref

    live = 0
    peak = 0

    def released():
        nonlocal live
        live -= 1

    def stream():
        nonlocal live, peak
        for i in range(1000):
            candidate = entry(f"candidate{i:04}")
            candidate.importance = i / 1000
            live += 1
            peak = max(peak, live)
            weakref.finalize(candidate, released)
            yield candidate

    result = TemporalSelection().select_ranked(stream(), limit=5, rank_key=lambda row: (row.importance, "", ""))
    assert [row.id for row in result] == [f"candidate{i:04}" for i in range(999, 994, -1)]
    assert peak <= 8, f"Unbounded candidate retention: {peak} live models"


def test_ranked_selection_prioritizes_eligible_over_high_rank_deferred() -> None:
    old = entry("old", closed=True).model_copy(update={"importance": 1.0})
    current = entry("current").model_copy(update={"importance": 0.1})
    selection = TemporalSelection(include_superseded=True)
    assert selection.select_ranked([old, current], limit=1, rank_key=lambda row: (row.importance, "", "")) == [current]
