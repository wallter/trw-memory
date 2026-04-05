"""Tests for the Thompson Sampling bandit selector (PRD-CORE-105-FR01).

These are unit tests -- no filesystem I/O, no tmp_path needed.
"""

from __future__ import annotations

import json
import random

import pytest

from trw_memory.bandit.thompson import ArmState, BanditDecision, BanditSelector


# ---------------------------------------------------------------------------
# test_select_single_arm
# ---------------------------------------------------------------------------


def test_select_single_arm() -> None:
    """Single arm is always selected with probability 1.0 conceptually."""
    selector = BanditSelector()
    decision = selector.select(["arm-a"])
    assert decision.selected_id == "arm-a"
    assert decision.runner_up_id is None
    assert decision.runner_up_probability is None


# ---------------------------------------------------------------------------
# test_select_cold_start_round_robin
# ---------------------------------------------------------------------------


def test_select_cold_start_round_robin() -> None:
    """New arms get guaranteed cold_start_min exposures via round-robin."""
    selector = BanditSelector(cold_start_min=3)
    arms = ["a", "b", "c"]

    # Track selections during cold start phase (3 arms * 3 min = 9 rounds)
    counts: dict[str, int] = {a: 0 for a in arms}
    for _ in range(9):
        decision = selector.select(arms)
        assert decision.exploration is True, "Cold-start selections should be exploration"
        counts[decision.selected_id] += 1
        selector.update(decision.selected_id, 0.5)

    # Each arm must have been selected at least cold_start_min times
    for arm in arms:
        assert counts[arm] >= 3, f"Arm {arm} only got {counts[arm]} exposures, expected >= 3"


# ---------------------------------------------------------------------------
# test_update_clamps_reward
# ---------------------------------------------------------------------------


def test_update_clamps_reward() -> None:
    """Rewards outside [0.0, 1.0] are clamped."""
    selector = BanditSelector(tau=10)

    selector.update("arm-a", 2.5)
    arm = selector._arms["arm-a"]
    # reward clamped to 1.0 -> window = [1.0]
    assert arm.window == [1.0]

    selector.update("arm-b", -0.5)
    arm_b = selector._arms["arm-b"]
    # reward clamped to 0.0 -> window = [0.0]
    assert arm_b.window == [0.0]


# ---------------------------------------------------------------------------
# test_sliding_window_evicts_oldest
# ---------------------------------------------------------------------------


def test_sliding_window_evicts_oldest() -> None:
    """Window respects tau limit by evicting the oldest observation."""
    tau = 5
    selector = BanditSelector(tau=tau)

    # Fill window beyond tau
    for i in range(tau + 3):
        selector.update("arm-a", float(i) / 10.0)

    arm = selector._arms["arm-a"]
    assert len(arm.window) == tau
    # Oldest values (0.0, 0.1, 0.2) should be evicted; window has [0.3, 0.4, 0.5, 0.6, 0.7]
    assert arm.window[0] == pytest.approx(0.3)
    assert arm.window[-1] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# test_convergence_to_best_arm
# ---------------------------------------------------------------------------


def test_convergence_to_best_arm() -> None:
    """Over 500 cycles with 5 arms (best=0.8, rest=0.4), best arm selected >60% after cycle 100."""
    rng = random.Random(42)
    selector = BanditSelector(tau=25, cold_start_min=3, floor_exploration=0.12)
    arms = ["best", "avg1", "avg2", "avg3", "avg4"]

    # Reward functions
    reward_rates = {"best": 0.8, "avg1": 0.4, "avg2": 0.4, "avg3": 0.4, "avg4": 0.4}

    # Warm up (first 100 cycles)
    for _ in range(100):
        decision = selector.select(arms)
        reward = 1.0 if rng.random() < reward_rates[decision.selected_id] else 0.0
        selector.update(decision.selected_id, reward)

    # Count selections in cycles 100-500
    best_count = 0
    total = 400
    for _ in range(total):
        decision = selector.select(arms)
        if decision.selected_id == "best":
            best_count += 1
        reward = 1.0 if rng.random() < reward_rates[decision.selected_id] else 0.0
        selector.update(decision.selected_id, reward)

    best_fraction = best_count / total
    assert best_fraction > 0.60, (
        f"Best arm selected {best_fraction:.1%} of the time after warmup, expected >60%"
    )


# ---------------------------------------------------------------------------
# test_json_round_trip
# ---------------------------------------------------------------------------


def test_json_round_trip() -> None:
    """to_json/from_json preserves selector state."""
    selector = BanditSelector(tau=10, cold_start_min=2, floor_exploration=0.15)
    selector.update("arm-a", 0.9)
    selector.update("arm-a", 0.8)
    selector.update("arm-b", 0.3)

    json_str = selector.to_json()
    restored = BanditSelector.from_json(json_str)

    # Verify hyperparameters restored
    assert restored._tau == selector._tau
    assert restored._cold_start_min == selector._cold_start_min
    assert restored._floor_exploration == pytest.approx(selector._floor_exploration)

    # Verify arm state
    assert set(restored._arms.keys()) == {"arm-a", "arm-b"}
    assert restored._arms["arm-a"].window == pytest.approx(selector._arms["arm-a"].window)
    assert restored._arms["arm-a"].alpha == pytest.approx(selector._arms["arm-a"].alpha)
    assert restored._arms["arm-a"].beta == pytest.approx(selector._arms["arm-a"].beta)
    assert restored._arms["arm-a"].exposure_count == selector._arms["arm-a"].exposure_count


# ---------------------------------------------------------------------------
# test_json_corrupt_returns_fresh
# ---------------------------------------------------------------------------


def test_json_corrupt_returns_fresh() -> None:
    """Corrupt JSON string returns a fresh BanditSelector without crashing."""
    result = BanditSelector.from_json("this is not valid json {{{")
    assert isinstance(result, BanditSelector)
    assert len(result._arms) == 0


def test_json_corrupt_missing_keys_returns_fresh() -> None:
    """JSON with unexpected structure returns a fresh instance."""
    result = BanditSelector.from_json('{"unexpected": true}')
    assert isinstance(result, BanditSelector)
    assert len(result._arms) == 0


# ---------------------------------------------------------------------------
# test_empty_eligible_raises
# ---------------------------------------------------------------------------


def test_empty_eligible_raises() -> None:
    """Empty eligible_ids list raises ValueError."""
    selector = BanditSelector()
    with pytest.raises(ValueError, match="eligible_ids"):
        selector.select([])


# ---------------------------------------------------------------------------
# test_floor_exploration_rate
# ---------------------------------------------------------------------------


def test_floor_exploration_rate() -> None:
    """Over 1000 selections with a dominant arm, exploration rate >= 10%."""
    selector = BanditSelector(tau=25, cold_start_min=3, floor_exploration=0.12)
    arms = ["dominant", "other1", "other2"]

    # Prime the dominant arm with strong signal
    for _ in range(50):
        selector.update("dominant", 1.0)
    for _ in range(50):
        selector.update("other1", 0.0)
    for _ in range(50):
        selector.update("other2", 0.0)

    # Now measure exploration rate
    exploration_count = 0
    total = 1000
    for _ in range(total):
        decision = selector.select(arms)
        if decision.exploration:
            exploration_count += 1

    exploration_rate = exploration_count / total
    assert exploration_rate >= 0.10, (
        f"Exploration rate {exploration_rate:.1%} is below 10% floor"
    )


# ---------------------------------------------------------------------------
# test_new_arm_added_midstream
# ---------------------------------------------------------------------------


def test_new_arm_added_midstream() -> None:
    """Add arm at cycle 200, verify it gets cold_start_min exposures."""
    cold_start_min = 3
    selector = BanditSelector(tau=25, cold_start_min=cold_start_min, floor_exploration=0.12)
    original_arms = ["a", "b"]

    # Run 200 cycles with original arms
    for _ in range(200):
        decision = selector.select(original_arms)
        selector.update(decision.selected_id, 0.5)

    # Add a new arm
    all_arms = ["a", "b", "new_arm"]
    new_arm_selections = 0
    for _ in range(50):
        decision = selector.select(all_arms)
        if decision.selected_id == "new_arm":
            new_arm_selections += 1
        selector.update(decision.selected_id, 0.5)

    assert new_arm_selections >= cold_start_min, (
        f"New arm only got {new_arm_selections} selections, expected >= {cold_start_min}"
    )


# ---------------------------------------------------------------------------
# test_decision_has_runner_up
# ---------------------------------------------------------------------------


def test_decision_has_runner_up() -> None:
    """With 2+ arms, runner_up_id is populated."""
    selector = BanditSelector()
    arms = ["arm-a", "arm-b", "arm-c"]
    # Warm up all arms past cold start
    for arm in arms:
        for _ in range(5):
            selector.update(arm, 0.5)

    decision = selector.select(arms)
    assert decision.runner_up_id is not None
    assert decision.runner_up_id != decision.selected_id
    assert decision.runner_up_probability is not None
    assert 0.0 <= decision.runner_up_probability <= 1.0


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


def test_arm_state_defaults() -> None:
    """ArmState dataclass has correct optimistic priors."""
    arm = ArmState()
    assert arm.alpha == 2.0
    assert arm.beta == 1.0
    assert arm.window == []
    assert arm.exposure_count == 0


def test_decision_dataclass_fields() -> None:
    """BanditDecision has all required fields."""
    decision = BanditDecision(
        selected_id="test",
        selection_probability=0.75,
        runner_up_id="other",
        runner_up_probability=0.25,
        exploration=False,
    )
    assert decision.selected_id == "test"
    assert decision.selection_probability == 0.75
    assert decision.runner_up_id == "other"
    assert decision.runner_up_probability == 0.25
    assert decision.exploration is False


def test_update_recomputes_alpha_beta() -> None:
    """Alpha and beta are recomputed from the full window on each update."""
    selector = BanditSelector(tau=100)

    selector.update("arm-a", 1.0)
    arm = selector._arms["arm-a"]
    # alpha = 2.0 + sum([1.0]) = 3.0
    # beta = 1.0 + sum([1 - 1.0]) = 1.0
    assert arm.alpha == pytest.approx(3.0)
    assert arm.beta == pytest.approx(1.0)

    selector.update("arm-a", 0.0)
    # window = [1.0, 0.0]
    # alpha = 2.0 + 1.0 = 3.0
    # beta = 1.0 + (0.0 + 1.0) = 2.0
    assert arm.alpha == pytest.approx(3.0)
    assert arm.beta == pytest.approx(2.0)


def test_selection_probability_in_valid_range() -> None:
    """selection_probability is always between 0 and 1."""
    selector = BanditSelector()
    for _ in range(50):
        decision = selector.select(["a", "b"])
        assert 0.0 <= decision.selection_probability <= 1.0
        selector.update(decision.selected_id, 0.5)


def test_exposure_count_increments() -> None:
    """exposure_count increments with each update call."""
    selector = BanditSelector()
    for i in range(10):
        selector.update("arm-a", 0.5)
        assert selector._arms["arm-a"].exposure_count == i + 1


# ---------------------------------------------------------------------------
# Pool cap tests
# ---------------------------------------------------------------------------


def test_select_pool_cap() -> None:
    """Pools exceeding 500 arms are capped without raising."""
    bandit = BanditSelector()
    ids = [f"arm-{i}" for i in range(600)]
    decision = bandit.select(ids)
    assert decision.selected_id.startswith("arm-")
    # Only 500 arms should be tracked (cap prevents the rest)
    assert len(bandit._arms) <= 500


def test_select_pool_cap_preserves_order() -> None:
    """Pool cap keeps the first 500 ids (preserves caller ordering)."""
    bandit = BanditSelector()
    ids = [f"arm-{i}" for i in range(600)]
    decision = bandit.select(ids)
    # Selected arm must be from the first 500
    idx = int(decision.selected_id.split("-")[1])
    assert idx < 500


# ---------------------------------------------------------------------------
# Constructor validation tests
# ---------------------------------------------------------------------------


def test_constructor_invalid_tau() -> None:
    """tau < 1 raises ValueError."""
    with pytest.raises(ValueError, match="tau must be >= 1"):
        BanditSelector(tau=0)


def test_constructor_invalid_tau_negative() -> None:
    """Negative tau raises ValueError."""
    with pytest.raises(ValueError, match="tau must be >= 1"):
        BanditSelector(tau=-5)


def test_constructor_invalid_cold_start_min() -> None:
    """Negative cold_start_min raises ValueError."""
    with pytest.raises(ValueError, match="cold_start_min must be >= 0"):
        BanditSelector(cold_start_min=-1)


def test_constructor_cold_start_min_zero_ok() -> None:
    """cold_start_min=0 is valid (no cold start)."""
    selector = BanditSelector(cold_start_min=0)
    assert selector._cold_start_min == 0


def test_constructor_invalid_floor_too_high() -> None:
    """floor_exploration > 1.0 raises ValueError."""
    with pytest.raises(ValueError, match="floor_exploration must be in"):
        BanditSelector(floor_exploration=2.0)


def test_constructor_invalid_floor_negative() -> None:
    """Negative floor_exploration raises ValueError."""
    with pytest.raises(ValueError, match="floor_exploration must be in"):
        BanditSelector(floor_exploration=-0.1)


def test_constructor_valid_edge_values() -> None:
    """Boundary values for floor_exploration are accepted."""
    s1 = BanditSelector(floor_exploration=0.0)
    assert s1._floor_exploration == 0.0
    s2 = BanditSelector(floor_exploration=1.0)
    assert s2._floor_exploration == 1.0
