"""trw_memory._sweep: bounded, resumable work over a keyset-paged source (rc9 sweep B2)."""

from __future__ import annotations

import time
from collections.abc import Sequence

import pytest

from trw_memory._sweep import MAX_TOKEN_CHARS, decode_token, encode_token, sweep

pytestmark = pytest.mark.unit

ROWS = list(range(1, 8))


def _fetch(after: int | None, limit: int) -> list[int]:
    return [row for row in ROWS if after is None or row > after][:limit]


def _visit_until(seen: list[int]):  # type: ignore[no-untyped-def]
    def visit(batch: Sequence[int], deadline: float) -> int:
        for consumed, row in enumerate(batch, start=1):
            seen.append(row)
            if time.monotonic() >= deadline:
                return consumed
        return len(batch)

    return visit


def test_a_sweep_within_budget_visits_every_row_and_reports_the_source_exhausted() -> None:
    seen: list[int] = []

    assert sweep(_fetch, int, _visit_until(seen), after=None, page=2, rows=100, seconds=60) is None
    assert seen == ROWS


def test_the_row_budget_stops_the_call_and_the_cursor_resumes_it() -> None:
    seen: list[int] = []

    cursor = sweep(_fetch, int, _visit_until(seen), after=None, page=2, rows=3, seconds=60)
    assert (cursor, seen) == (3, [1, 2, 3])

    assert sweep(_fetch, int, _visit_until(seen), after=cursor, page=2, rows=100, seconds=60) is None
    assert seen == ROWS


def test_a_spent_clock_still_advances_one_row_per_call() -> None:
    """A row slower than the whole budget must not wedge the sweep."""
    seen: list[int] = []
    cursor = None
    for expected in [*ROWS[:-1], None]:  # the last row comes back as a short page: done
        cursor = sweep(_fetch, int, _visit_until(seen), after=cursor, page=4, rows=100, seconds=-1.0)
        assert cursor == expected

    assert seen == ROWS


def test_a_visit_that_consumes_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="at least the first"):
        sweep(_fetch, int, lambda batch, deadline: 0, after=None, page=2, rows=10, seconds=60)


def test_a_token_round_trips_and_anything_else_is_refused() -> None:
    assert decode_token(encode_token(["2026-09-25T00:00:00+00:00", "M-1"]), 2) == ["2026-09-25T00:00:00+00:00", "M-1"]
    for bad in ("[1, 2]", '["only-one"]', '{"a": "b"}', "not json", '["x", "' + "y" * MAX_TOKEN_CHARS + '"]'):
        with pytest.raises(ValueError):
            decode_token(bad, 2)


def test_a_short_last_page_ends_the_sweep_even_on_a_spent_clock() -> None:
    """No extra no-op call, and no cursor pointing past the end (worker-2's maintain stamp)."""
    seen: list[int] = []
    rows = [1, 2, 3]

    def fetch(after: int | None, limit: int) -> list[int]:
        return [row for row in rows if after is None or row > after][:limit]

    first = sweep(fetch, int, _visit_until(seen), after=None, page=5, rows=100, seconds=-1.0)
    second = sweep(fetch, int, _visit_until(seen), after=first, page=5, rows=100, seconds=-1.0)
    third = sweep(fetch, int, _visit_until(seen), after=second, page=5, rows=100, seconds=-1.0)

    assert (first, second, third, seen) == (1, 2, None, [1, 2, 3])


@pytest.mark.parametrize(("page", "rows"), [(0, 10), (2, 0), (-1, 10)])
def test_a_non_positive_budget_is_refused_not_reported_as_exhausted(page: int, rows: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        sweep(_fetch, int, _visit_until([]), after=None, page=page, rows=rows, seconds=60)


def test_a_fetch_that_returns_more_than_asked_is_refused_before_any_visit() -> None:
    seen: list[int] = []

    with pytest.raises(ValueError, match="for a limit of 2"):
        sweep(lambda after, limit: ROWS, int, _visit_until(seen), after=None, page=2, rows=10, seconds=60)
    assert seen == []


def test_encode_refuses_what_decode_would_refuse() -> None:
    with pytest.raises(ValueError):
        encode_token(["x" * MAX_TOKEN_CHARS])
    with pytest.raises(ValueError):
        encode_token([1, "a"])  # type: ignore[list-item]
