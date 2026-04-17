"""Tests for the contextual bandit (LinUCB) selector (PRD-CORE-105-FR02).

Unit tests -- no filesystem I/O, no tmp_path needed.
"""

from __future__ import annotations

import random

import pytest

from trw_memory.bandit.contextual import ContextualBanditSelector

# ---------------------------------------------------------------------------
# test_select_no_context_random
# ---------------------------------------------------------------------------


def test_select_no_context_uses_thompson_not_random() -> None:
    """With context_vector=None, selection uses Thompson Sampling not uniform random.

    After training the Thompson fallback (via update with no context), the
    arm with higher reward should be selected more often than chance.
    This verifies FR02: 'fall back to non-contextual Thompson Sampling'.
    """
    selector = ContextualBanditSelector(feature_dim=3)
    arms = ["arm_good", "arm_bad"]

    for _ in range(30):
        selector.update("arm_good", reward=0.9, context_vector=None)
        selector.update("arm_bad", reward=0.1, context_vector=None)

    counts: dict[str, int] = {"arm_good": 0, "arm_bad": 0}
    n_trials = 200
    for _ in range(n_trials):
        selected_id, _ = selector.select(arms, context_vector=None)
        counts[selected_id] = counts.get(selected_id, 0) + 1

    good_rate = counts["arm_good"] / n_trials
    assert good_rate > 0.60, f"Expected arm_good selected > 60% after training, got {good_rate:.2f}"


def test_thompson_state_serialized_in_to_dict() -> None:
    """Thompson fallback state is included in to_dict() output."""
    selector = ContextualBanditSelector(feature_dim=2)
    selector.update("arm_a", reward=0.9, context_vector=None)

    data = selector.to_dict()
    assert "thompson_state" in data
    assert "arms" in data["thompson_state"]


def test_thompson_state_restored_in_from_dict() -> None:
    """Thompson fallback state is restored from from_dict()."""
    selector = ContextualBanditSelector(feature_dim=2)
    selector.update("arm_a", reward=0.9, context_vector=None)
    selector.update("arm_b", reward=0.1, context_vector=None)

    data = selector.to_dict()
    restored = ContextualBanditSelector.from_dict(data)

    assert len(restored._thompson._arms) == 2
    assert "arm_a" in restored._thompson._arms


def test_select_no_context_random() -> None:
    """With context_vector=None, selection returns a valid arm (no crash)."""
    selector = ContextualBanditSelector(feature_dim=3)
    arms = ["a", "b", "c"]

    selected_id, score = selector.select(arms, context_vector=None)
    assert selected_id in arms
    assert isinstance(score, float)


def test_select_empty_context_uses_thompson_fallback() -> None:
    """Empty context vectors use the Thompson fallback too."""
    selector = ContextualBanditSelector(feature_dim=2)
    arms = ["arm_good", "arm_bad"]

    for _ in range(30):
        selector.update("arm_good", reward=0.9, context_vector=None)
        selector.update("arm_bad", reward=0.1, context_vector=None)

    counts = {"arm_good": 0, "arm_bad": 0}
    for _ in range(200):
        selected_id, _ = selector.select(arms, context_vector=[])
        counts[selected_id] += 1

    assert counts["arm_good"] > counts["arm_bad"]


# ---------------------------------------------------------------------------
# test_select_with_context
# ---------------------------------------------------------------------------


def test_select_with_context() -> None:
    """With a valid context vector, a specific arm is selected."""
    selector = ContextualBanditSelector(feature_dim=2, alpha=1.0)
    arms = ["a", "b"]

    selected_id, score = selector.select(arms, context_vector=[1.0, 0.0])
    assert selected_id in arms
    assert isinstance(score, float)
    assert score > 0.0  # UCB should be positive with identity A_inv


def test_select_decision_returns_runner_up_metadata() -> None:
    """Contextual live integrations can request a Thompson-compatible decision."""
    selector = ContextualBanditSelector(feature_dim=2, alpha=0.5)
    arms = ["arm_a", "arm_b"]

    for _ in range(40):
        selector.update("arm_a", reward=1.0, context_vector=[1.0, 0.0])
        selector.update("arm_b", reward=0.0, context_vector=[1.0, 0.0])

    decision = selector.select_decision(arms, context_vector=[1.0, 0.0])

    assert decision.selected_id == "arm_a"
    assert decision.runner_up_id == "arm_b"
    assert 0.0 < decision.selection_probability <= 1.0
    assert 0.0 <= (decision.runner_up_probability or 0.0) < 1.0


# ---------------------------------------------------------------------------
# test_dimension_mismatch_raises
# ---------------------------------------------------------------------------


def test_dimension_mismatch_raises() -> None:
    """Wrong dimension context vector raises ValueError."""
    selector = ContextualBanditSelector(feature_dim=3)
    with pytest.raises(ValueError, match="context_vector dimension"):
        selector.select(["a", "b"], context_vector=[1.0, 0.0])


# ---------------------------------------------------------------------------
# test_linucb_different_context_different_selection
# ---------------------------------------------------------------------------


def test_linucb_different_context_different_selection() -> None:
    """Two different context vectors produce different arm selections after training."""
    selector = ContextualBanditSelector(feature_dim=2, alpha=1.0)
    arms = ["arm_a", "arm_b"]

    # Train: context [1, 0] -> arm_a is good (high reward)
    for _ in range(50):
        selector.update("arm_a", reward=1.0, context_vector=[1.0, 0.0])
        selector.update("arm_b", reward=0.0, context_vector=[1.0, 0.0])

    # Train: context [0, 1] -> arm_b is good (high reward)
    for _ in range(50):
        selector.update("arm_b", reward=1.0, context_vector=[0.0, 1.0])
        selector.update("arm_a", reward=0.0, context_vector=[0.0, 1.0])

    # After training, context [1, 0] should select arm_a
    selected_a, _ = selector.select(arms, context_vector=[1.0, 0.0])
    # After training, context [0, 1] should select arm_b
    selected_b, _ = selector.select(arms, context_vector=[0.0, 1.0])

    assert selected_a != selected_b, f"Expected different selections for different contexts, got {selected_a} for both"


# ---------------------------------------------------------------------------
# test_degenerate_scores_fallback
# ---------------------------------------------------------------------------


def test_degenerate_scores_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """When all UCB scores within 1%, falls back to Thompson."""
    selector = ContextualBanditSelector(feature_dim=2, alpha=1.0)
    arms = ["a", "b", "c"]

    # With no training data and identity matrices, all arms are identical.
    # All scores should be degenerate (within 1% of each other).
    import logging

    import structlog

    # Configure structlog to emit to stdlib for caplog capture
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    )

    selected_id, score = selector.select(arms, context_vector=[1.0, 0.0])
    assert selected_id in arms

    # Re-configure structlog back to default to not pollute other tests
    structlog.reset_defaults()


def test_degenerate_scores_fallback_uses_trained_thompson() -> None:
    """Degenerate contextual scores defer to the Thompson fallback."""
    selector = ContextualBanditSelector(feature_dim=2, alpha=1.0)
    arms = ["arm_good", "arm_bad"]

    for _ in range(40):
        selector.update("arm_good", reward=0.95, context_vector=None)
        selector.update("arm_bad", reward=0.05, context_vector=None)

    counts = {"arm_good": 0, "arm_bad": 0}
    for _ in range(200):
        selected_id, _ = selector.select(arms, context_vector=[1.0, 0.0])
        counts[selected_id] += 1

    assert counts["arm_good"] > counts["arm_bad"]


# ---------------------------------------------------------------------------
# test_update_no_context_noop
# ---------------------------------------------------------------------------


def test_update_no_context_noop() -> None:
    """update() with context_vector=None is a no-op."""
    selector = ContextualBanditSelector(feature_dim=2)

    selector.update("arm_a", reward=1.0, context_vector=None)

    # The arm should not be created or updated
    assert len(selector._arms) == 0


# ---------------------------------------------------------------------------
# test_to_dict_from_dict_round_trip
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_round_trip() -> None:
    """Serialize/deserialize preserves state."""
    selector = ContextualBanditSelector(feature_dim=2, alpha=1.5)

    # Train some arms
    selector.update("arm_a", reward=0.9, context_vector=[1.0, 0.0])
    selector.update("arm_a", reward=0.8, context_vector=[0.5, 0.5])
    selector.update("arm_b", reward=0.3, context_vector=[0.0, 1.0])

    data = selector.to_dict()
    restored = ContextualBanditSelector.from_dict(data)

    # Verify hyperparameters
    assert restored._feature_dim == selector._feature_dim
    assert restored._alpha == pytest.approx(selector._alpha)

    # Verify arm states
    assert set(restored._arms.keys()) == set(selector._arms.keys())

    for arm_id in selector._arms:
        orig = selector._arms[arm_id]
        rest = restored._arms[arm_id]
        assert rest.n_obs == orig.n_obs
        assert rest.b == pytest.approx(orig.b)
        # Verify A_inv matrix
        for i in range(len(orig.A_inv)):
            assert rest.A_inv[i] == pytest.approx(orig.A_inv[i])

    # Verify that selections produce the same results
    context = [1.0, 0.0]
    arms = ["arm_a", "arm_b"]
    orig_sel, orig_score = selector.select(arms, context_vector=context)
    rest_sel, rest_score = restored.select(arms, context_vector=context)
    assert orig_sel == rest_sel
    assert orig_score == pytest.approx(rest_score)


def test_compact_dict_round_trip_restores_observed_arms() -> None:
    """Compact envelope persistence restores a behaviorally useful selector."""
    selector = ContextualBanditSelector(feature_dim=2, alpha=0.5)
    for _ in range(30):
        selector.update("arm_a", reward=1.0, context_vector=[1.0, 0.0])
        selector.update("arm_b", reward=0.0, context_vector=[1.0, 0.0])

    compact = selector.to_compact_dict(max_arms=4)
    restored = ContextualBanditSelector.from_compact_dict(compact)

    assert restored._arms["arm_a"].n_obs == selector._arms["arm_a"].n_obs
    selected_id, _ = restored.select(["arm_a", "arm_b"], context_vector=[1.0, 0.0])
    assert selected_id == "arm_a"


# ---------------------------------------------------------------------------
# test_from_dict_corrupt_returns_fresh
# ---------------------------------------------------------------------------


def test_from_dict_corrupt_returns_fresh() -> None:
    """Malformed data returns a fresh instance."""
    # Completely wrong structure
    result = ContextualBanditSelector.from_dict({"garbage": True})
    assert isinstance(result, ContextualBanditSelector)
    assert len(result._arms) == 0

    # Wrong types
    result2 = ContextualBanditSelector.from_dict({"feature_dim": "not_a_number", "alpha": [], "arms": "wrong"})
    assert isinstance(result2, ContextualBanditSelector)


def test_from_dict_bad_arms_type_returns_fresh() -> None:
    """arms field that is not a dict returns a fresh instance with correct params."""
    result = ContextualBanditSelector.from_dict({"feature_dim": 3, "alpha": 2.0, "arms": "not_a_dict"})
    assert isinstance(result, ContextualBanditSelector)
    assert result._feature_dim == 3
    assert result._alpha == 2.0
    assert len(result._arms) == 0


# ---------------------------------------------------------------------------
# test_select_empty_raises
# ---------------------------------------------------------------------------


def test_select_empty_raises() -> None:
    """Empty eligible_ids raises ValueError."""
    selector = ContextualBanditSelector(feature_dim=2)
    with pytest.raises(ValueError, match="eligible_ids"):
        selector.select([])


# ---------------------------------------------------------------------------
# test_convergence_contextual
# ---------------------------------------------------------------------------


def test_convergence_contextual() -> None:
    """Train with context [1,0] -> arm A good, [0,1] -> arm B good.

    After training, select with [1,0] returns A, [0,1] returns B.
    """
    rng = random.Random(42)
    selector = ContextualBanditSelector(feature_dim=2, alpha=0.5)
    arms = ["A", "B"]

    # Train 200 rounds
    for _ in range(200):
        # Context [1, 0]: arm A has high reward, arm B has low
        ctx_1 = [1.0, 0.0]
        reward_a = 0.8 + rng.gauss(0, 0.05)
        reward_b = 0.2 + rng.gauss(0, 0.05)
        selector.update("A", reward=max(0, min(1, reward_a)), context_vector=ctx_1)
        selector.update("B", reward=max(0, min(1, reward_b)), context_vector=ctx_1)

        # Context [0, 1]: arm B has high reward, arm A has low
        ctx_2 = [0.0, 1.0]
        reward_a2 = 0.2 + rng.gauss(0, 0.05)
        reward_b2 = 0.8 + rng.gauss(0, 0.05)
        selector.update("A", reward=max(0, min(1, reward_a2)), context_vector=ctx_2)
        selector.update("B", reward=max(0, min(1, reward_b2)), context_vector=ctx_2)

    # After training:
    sel_ctx1, _ = selector.select(arms, context_vector=[1.0, 0.0])
    sel_ctx2, _ = selector.select(arms, context_vector=[0.0, 1.0])

    assert sel_ctx1 == "A", f"Expected A for context [1,0], got {sel_ctx1}"
    assert sel_ctx2 == "B", f"Expected B for context [0,1], got {sel_ctx2}"


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_single_arm_select() -> None:
    """Single eligible arm is always returned."""
    selector = ContextualBanditSelector(feature_dim=2)
    selected_id, score = selector.select(["only_arm"], context_vector=[1.0, 0.0])
    assert selected_id == "only_arm"
    assert isinstance(score, float)


def test_update_creates_arm_state() -> None:
    """update() with a new arm_id creates the arm state."""
    selector = ContextualBanditSelector(feature_dim=2)
    selector.update("new_arm", reward=0.5, context_vector=[1.0, 0.0])
    assert "new_arm" in selector._arms
    assert selector._arms["new_arm"].n_obs == 1


def test_matrix_operations_correctness() -> None:
    """Verify the helper matrix operations produce correct results."""
    from trw_memory.bandit.contextual import (
        _identity,
        _mat_vec_mul,
        _outer_product,
        _vec_dot,
    )

    # Identity matrix
    I = _identity(3)
    assert I == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    # Matrix-vector multiply with identity
    v = [1.0, 2.0, 3.0]
    result = _mat_vec_mul(I, v)
    assert result == pytest.approx(v)

    # Dot product
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    assert _vec_dot(a, b) == pytest.approx(32.0)  # 1*4 + 2*5 + 3*6

    # Outer product
    c = [1.0, 2.0]
    d = [3.0, 4.0]
    outer = _outer_product(c, d)
    assert outer[0] == pytest.approx([3.0, 4.0])
    assert outer[1] == pytest.approx([6.0, 8.0])


def test_sherman_morrison_preserves_symmetry() -> None:
    """After multiple updates, A_inv should remain approximately symmetric."""
    selector = ContextualBanditSelector(feature_dim=3, alpha=1.0)

    # Do several updates with different contexts
    contexts = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0],
        [0.3, 0.3, 0.4],
    ]
    for ctx in contexts:
        selector.update("arm_a", reward=0.7, context_vector=ctx)

    arm = selector._arms["arm_a"]
    d = len(arm.A_inv)
    for i in range(d):
        for j in range(d):
            assert arm.A_inv[i][j] == pytest.approx(arm.A_inv[j][i], abs=1e-10), f"A_inv not symmetric at ({i},{j})"
