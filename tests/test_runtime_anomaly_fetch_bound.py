"""Bounded reference-fetch for the anomaly hot path (trw-memory-isolation-9).

``score_anomaly`` previously hydrated up to 1,000 full MemoryEntry objects on
every write, then discarded all but the most-recent 100. The fetch is now
bounded to a small multiple of the rolling window, which cuts per-write
deserialization while keeping the rolling-window contents identical (the
filtered-out set in the main store is tiny: system canaries are capped at 5 and
quarantined rows live in a separate store).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.security._runtime_anomaly import (
    _REFERENCE_FETCH_LIMIT,
    _ROLLING_WINDOW,
    score_anomaly,
)


def _entry(idx: int, *, content: str = "normal content", minutes_old: int = 0) -> MemoryEntry:
    return MemoryEntry(
        id=f"M-{idx:04d}",
        content=content,
        namespace="project:alpha",
        updated_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_old),
    )


class TestScoreAnomalyFetchBound:
    def test_fetch_limit_is_bounded_not_one_thousand(self) -> None:
        assert _REFERENCE_FETCH_LIMIT == _ROLLING_WINDOW * 2
        assert _REFERENCE_FETCH_LIMIT < 1_000

    def test_list_entries_called_with_bounded_limit(self) -> None:
        backend = MagicMock()
        backend.list_entries.return_value = [_entry(i, minutes_old=i) for i in range(50)]
        cfg = MemoryConfig()

        score_anomaly(_entry(999, content="A" * 5000), backend, config=cfg)

        backend.list_entries.assert_called_once()
        _, kwargs = backend.list_entries.call_args
        assert kwargs["limit"] == _REFERENCE_FETCH_LIMIT
        assert kwargs["namespace"] == "project:alpha"
        assert kwargs["status"] is MemoryStatus.ACTIVE

    def test_rolling_window_capped_and_stats_use_window(self) -> None:
        # More clean entries than the window exist; stats must reflect exactly
        # the most-recent _ROLLING_WINDOW of them.
        backend = MagicMock()
        ref = [_entry(i, minutes_old=i) for i in range(_REFERENCE_FETCH_LIMIT)]
        backend.list_entries.return_value = ref
        cfg = MemoryConfig()

        _, stats = score_anomaly(_entry(999, content="x"), backend, config=cfg)

        assert stats.sample_count == _ROLLING_WINDOW

    def test_canary_entries_skipped_window_still_full(self) -> None:
        # Mix system-canary rows into the recent window; the over-fetch buffer
        # lets them be filtered out while the clean window remains full.
        backend = MagicMock()
        clean = [_entry(i, minutes_old=i) for i in range(_ROLLING_WINDOW)]
        canaries = [_entry(900 + i, minutes_old=i) for i in range(5)]
        for c in canaries:
            c.metadata["system_canary"] = "true"
        # Interleave so canaries sit inside the most-recent region.
        backend.list_entries.return_value = canaries + clean
        cfg = MemoryConfig()

        _, stats = score_anomaly(_entry(999, content="x"), backend, config=cfg)

        # All 5 canaries are filtered, leaving exactly the window of clean rows.
        assert stats.sample_count == _ROLLING_WINDOW
