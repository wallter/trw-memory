"""14-day observe-mode calibration clock tests (PRD-SEC-001 Item 5).

Sprint-96 carry-forward-b.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trw_memory.security import trust_scorer as trust_scorer_mod
from trw_memory.security.observe_clock import (
    read_observe_clock,
    start_observe_clock,
)
from trw_memory.security.trust_scorer import score_intake


@pytest.fixture(autouse=True)
def _reset_clock_flag() -> None:
    """Reset the module-level clock flag between tests."""
    trust_scorer_mod._clock_started = False


def test_start_writes_sidecar(tmp_path: Path) -> None:
    state = start_observe_clock(tmp_path)
    sidecar = tmp_path / "memory" / "security" / "observe_start.yaml"
    assert sidecar.exists()
    assert state.phase == "observe"
    assert state.prd == "PRD-SEC-001"
    # started_at + 14 days == promotion_review_at
    started = datetime.fromisoformat(state.started_at)
    review = datetime.fromisoformat(state.promotion_review_at)
    assert abs((review - started) - timedelta(days=14)) < timedelta(seconds=1)


def test_start_is_idempotent(tmp_path: Path) -> None:
    first = start_observe_clock(tmp_path)
    # Second call must return the SAME started_at, not overwrite
    second = start_observe_clock(tmp_path)
    assert first.started_at == second.started_at
    assert first.promotion_review_at == second.promotion_review_at


def test_read_returns_none_when_unset(tmp_path: Path) -> None:
    assert read_observe_clock(tmp_path) is None


def test_read_after_start_returns_state(tmp_path: Path) -> None:
    started = start_observe_clock(tmp_path, prd="PRD-TEST-001", window_days=7)
    read = read_observe_clock(tmp_path)
    assert read is not None
    assert read.prd == "PRD-TEST-001"
    assert read.started_at == started.started_at
    # window_days honored
    started_dt = datetime.fromisoformat(read.started_at)
    review_dt = datetime.fromisoformat(read.promotion_review_at)
    assert abs((review_dt - started_dt) - timedelta(days=7)) < timedelta(seconds=1)


def test_score_intake_starts_clock_on_first_call(tmp_path: Path) -> None:
    sidecar = tmp_path / "memory" / "security" / "observe_start.yaml"
    assert not sidecar.exists()
    score_intake("hello", {}, trw_dir=tmp_path)
    assert sidecar.exists()


def test_score_intake_does_not_restart_clock(tmp_path: Path) -> None:
    score_intake("first", {}, trw_dir=tmp_path)
    first_state = read_observe_clock(tmp_path)
    assert first_state is not None
    # Simulate process restart -- reset the flag but keep the sidecar
    trust_scorer_mod._clock_started = False
    score_intake("second", {}, trw_dir=tmp_path)
    second_state = read_observe_clock(tmp_path)
    assert second_state is not None
    assert second_state.started_at == first_state.started_at


def test_score_intake_without_trw_dir_does_not_start_clock(tmp_path: Path) -> None:
    score_intake("hello", {})  # no trw_dir
    # Nothing written anywhere under tmp_path
    assert not any(tmp_path.rglob("observe_start.yaml"))


def test_clock_started_at_is_utc(tmp_path: Path) -> None:
    state = start_observe_clock(tmp_path)
    dt = datetime.fromisoformat(state.started_at)
    # Must carry tzinfo and be close to now
    assert dt.tzinfo is not None
    assert abs(dt - datetime.now(timezone.utc)) < timedelta(seconds=10)
