"""Contextual bandit via LinUCB with Thompson Sampling no-context fallback.

General-purpose primitive -- no TRW/engineering-specific concepts.
The caller provides opaque context feature vectors.

Uses the LinUCB algorithm (Li et al., 2010) with Sherman-Morrison
incremental inverse updates to avoid matrix inversion.  No numpy
dependency -- all matrix operations are loop-based for small dimensions.

When context_vector is None, falls back to the internal Thompson Sampling
selector (FR01) rather than uniform random, satisfying PRD-CORE-105-FR02.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import structlog

from trw_memory.bandit.thompson import BanditDecision, BanditSelector

_logger = structlog.get_logger(__name__)
_COMPACT_FLOAT_DIGITS = 6


# ---------------------------------------------------------------------------
# Matrix helpers (no numpy -- loop-based, suitable for small d)
# ---------------------------------------------------------------------------


def _mat_vec_mul(mat: list[list[float]], vec: list[float]) -> list[float]:
    """Multiply matrix by vector."""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


def _vec_dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _outer_product(a: list[float], b: list[float]) -> list[list[float]]:
    """Outer product of two vectors."""
    return [[x * y for y in b] for x in a]


def _identity(d: int) -> list[list[float]]:
    """Create d x d identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Convert arbitrary arm scores into a stable probability-like distribution."""
    if len(scores) == 1:
        only_arm = next(iter(scores))
        return {only_arm: 1.0}

    min_score = min(scores.values())
    shift = (-min_score) + 1e-9 if min_score <= 0.0 else 0.0
    adjusted = {arm_id: max(0.0, score + shift) for arm_id, score in scores.items()}
    total = sum(adjusted.values())
    if total <= 0.0:
        uniform = 1.0 / len(scores)
        return {arm_id: uniform for arm_id in scores}
    return {arm_id: adjusted[arm_id] / total for arm_id in scores}


def _round_float(value: float) -> float:
    """Round persisted floats to keep JSON compact and deterministic."""
    return round(float(value), _COMPACT_FLOAT_DIGITS)


def _trim_trailing_defaults(values: list[float], default: float) -> list[float]:
    """Drop trailing default values from compact vectors."""
    trimmed = [_round_float(value) for value in values]
    while trimmed and math.isclose(trimmed[-1], default, abs_tol=1e-12):
        trimmed.pop()
    return trimmed


def _restore_vector(values: object, length: int, default: float) -> list[float]:
    """Restore a compact vector to fixed length."""
    restored = [default] * length
    if not isinstance(values, list):
        return restored
    for index, value in enumerate(values[:length]):
        restored[index] = float(value)
    return restored


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ContextualArmState:
    """Internal state for a single contextual bandit arm.

    Stores A_inv (the inverse of the context covariance matrix) directly
    so that we can update it incrementally via Sherman-Morrison without
    ever performing a full matrix inversion.
    """

    A_inv: list[list[float]] = field(default_factory=list)
    b: list[float] = field(default_factory=list)
    n_obs: int = 0


# ---------------------------------------------------------------------------
# ContextualBanditSelector
# ---------------------------------------------------------------------------


class ContextualBanditSelector:
    """LinUCB contextual bandit selector.

    For each arm, maintains an inverse covariance matrix (A_inv) and a
    reward-weighted context vector (b).  On ``select()``, computes a
    UCB score for each arm given the current context and returns the arm
    with the highest score.

    When ``context_vector`` is ``None``, selection falls back to an
    internal Thompson Sampling selector (FR01) rather than uniform random.
    This satisfies PRD-CORE-105-FR02: "When context_vector is None or
    empty, the system SHALL fall back to non-contextual Thompson Sampling."

    Args:
        feature_dim: Dimension of the context feature vector.
        alpha: Exploration parameter controlling the UCB width.
    """

    def __init__(self, feature_dim: int, alpha: float = 1.0) -> None:
        self._feature_dim = feature_dim
        self._alpha = alpha
        self._arms: dict[str, ContextualArmState] = {}
        # Internal Thompson Sampling selector for no-context fallback (FR02)
        self._thompson: BanditSelector = BanditSelector()

    # -- internal helpers --------------------------------------------------

    def _ensure_arm(self, arm_id: str) -> ContextualArmState:
        """Get or create arm state with proper initialization."""
        if arm_id not in self._arms:
            self._arms[arm_id] = ContextualArmState(
                A_inv=_identity(self._feature_dim),
                b=[0.0] * self._feature_dim,
            )
        return self._arms[arm_id]

    # -- public API --------------------------------------------------------

    def select(
        self,
        eligible_ids: list[str],
        context_vector: list[float] | None = None,
    ) -> tuple[str, float]:
        """Select an arm from *eligible_ids* using LinUCB.

        If *context_vector* is ``None``, falls back to uniform random
        selection.

        Args:
            eligible_ids: Non-empty list of candidate arm identifiers.
            context_vector: Feature vector of length ``feature_dim``, or
                ``None`` for context-free random selection.

        Returns:
            Tuple of (selected_arm_id, ucb_score).

        Raises:
            ValueError: If *eligible_ids* is empty or *context_vector*
                has wrong dimension.
        """
        if not eligible_ids:
            msg = "eligible_ids must not be empty"
            raise ValueError(msg)

        decision, selected_score = self._select_decision_with_score(
            eligible_ids,
            context_vector=context_vector,
        )
        return decision.selected_id, selected_score

    def select_decision(
        self,
        eligible_ids: list[str],
        context_vector: list[float] | None = None,
    ) -> BanditDecision:
        """Return a Thompson-compatible decision envelope for live integrations."""
        decision, _ = self._select_decision_with_score(
            eligible_ids,
            context_vector=context_vector,
        )
        return decision

    def update(
        self,
        arm_id: str,
        reward: float,
        context_vector: list[float] | None = None,
    ) -> None:
        """Record a reward observation for *arm_id* given a context.

        If *context_vector* is ``None``, the LinUCB A_inv/b update is
        skipped (no context to learn from), but the internal Thompson
        Sampling selector is updated so that no-context selections can
        still learn from rewards.

        Uses the Sherman-Morrison formula for rank-1 update of A_inv:
            A_inv_new = A_inv - (A_inv @ x @ x^T @ A_inv) / (1 + x^T @ A_inv @ x)

        Args:
            arm_id: Identifier of the arm that was pulled.
            reward: Observed reward value.
            context_vector: Feature vector used when the arm was selected,
                or ``None`` to skip the LinUCB update (but not Thompson).
        """
        # Always update Thompson fallback so no-context selections learn
        self._thompson.update(arm_id, reward)

        if not context_vector:
            return

        arm = self._ensure_arm(arm_id)
        x = context_vector

        # Sherman-Morrison rank-1 update of A_inv
        # numerator: (A_inv @ x) outer (x^T @ A_inv)
        a_inv_x = _mat_vec_mul(arm.A_inv, x)
        denom = 1.0 + _vec_dot(x, a_inv_x)

        # Compute the outer product of a_inv_x with itself (since A_inv
        # is symmetric, x^T @ A_inv = (A_inv @ x)^T = a_inv_x^T)
        outer = _outer_product(a_inv_x, a_inv_x)

        d = self._feature_dim
        for i in range(d):
            for j in range(d):
                arm.A_inv[i][j] -= outer[i][j] / denom

        # Update b: b += reward * x
        for i in range(d):
            arm.b[i] += reward * x[i]

        arm.n_obs += 1

    def seed_thompson_fallback(self, selector: BanditSelector) -> None:
        """Seed the internal Thompson fallback from an external selector state."""
        self._thompson = BanditSelector.from_json(selector.to_json())

    def to_dict(self) -> dict[str, Any]:
        """Serialize selector state for JSON persistence."""
        import json

        return {
            "feature_dim": self._feature_dim,
            "alpha": self._alpha,
            "arms": {
                arm_id: {
                    "A_inv": arm.A_inv,
                    "b": arm.b,
                    "n_obs": arm.n_obs,
                }
                for arm_id, arm in self._arms.items()
            },
            # Thompson fallback state for no-context selections (FR02)
            "thompson_state": json.loads(self._thompson.to_json()),
        }

    def to_compact_dict(self, *, max_arms: int = 16) -> dict[str, Any]:
        """Serialize a bounded compact state for envelope persistence.

        The shared state envelope uses a diagonal approximation of ``A_inv`` and
        persists only the most-observed arms. This keeps storage bounded while
        still restoring behaviorally real contextual state for frequently shown
        learnings.
        """
        active_arms = sorted(
            ((arm_id, arm) for arm_id, arm in self._arms.items() if arm.n_obs > 0),
            key=lambda item: (-item[1].n_obs, item[0]),
        )[:max_arms]
        return {
            "feature_dim": self._feature_dim,
            "alpha": self._alpha,
            "max_arms": max_arms,
            "arms": {
                arm_id: {
                    "d": _trim_trailing_defaults(
                        [arm.A_inv[index][index] for index in range(self._feature_dim)],
                        1.0,
                    ),
                    "b": _trim_trailing_defaults(arm.b, 0.0),
                    "n": arm.n_obs,
                }
                for arm_id, arm in active_arms
            },
        }

    @classmethod
    def from_compact_dict(cls, data: dict[str, Any]) -> ContextualBanditSelector:
        """Restore a compact envelope-persisted selector state."""
        try:
            selector = cls(
                feature_dim=int(data["feature_dim"]),
                alpha=float(data["alpha"]),
            )
            arms_data = data.get("arms", {})
            if not isinstance(arms_data, dict):
                return selector
            for arm_id, arm_dict in arms_data.items():
                if not isinstance(arm_dict, dict):
                    continue
                diagonal = _restore_vector(
                    arm_dict.get("d"),
                    selector._feature_dim,
                    1.0,
                )
                b = _restore_vector(arm_dict.get("b"), selector._feature_dim, 0.0)
                a_inv = _identity(selector._feature_dim)
                for index, value in enumerate(diagonal):
                    a_inv[index][index] = value
                selector._arms[str(arm_id)] = ContextualArmState(
                    A_inv=a_inv,
                    b=b,
                    n_obs=int(arm_dict.get("n", 0)),
                )
            return selector
        except (TypeError, ValueError, KeyError):
            _logger.warning("contextual_from_compact_failed", exc_info=True)
            return cls(feature_dim=2, alpha=1.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextualBanditSelector:
        """Deserialize from dict.

        Returns a fresh instance with default parameters on malformed
        input.
        """
        import json

        try:
            feature_dim = int(data["feature_dim"])
            alpha = float(data["alpha"])
            arms_data = data.get("arms", {})

            if not isinstance(arms_data, dict):
                _logger.warning("contextual_from_dict_bad_arms_type")
                return cls(feature_dim=feature_dim, alpha=alpha)

            selector = cls(feature_dim=feature_dim, alpha=alpha)

            for arm_id, arm_dict in arms_data.items():
                a_inv = [[float(v) for v in row] for row in arm_dict["A_inv"]]
                b = [float(v) for v in arm_dict["b"]]
                n_obs = int(arm_dict["n_obs"])

                selector._arms[arm_id] = ContextualArmState(
                    A_inv=a_inv,
                    b=b,
                    n_obs=n_obs,
                )

            # Restore Thompson fallback state if present (backward-compatible)
            thompson_state = data.get("thompson_state")
            if thompson_state is not None:
                try:
                    selector._thompson = BanditSelector.from_json(json.dumps(thompson_state))
                except Exception:  # justified: Thompson restore is best-effort
                    _logger.debug("contextual_thompson_restore_failed", exc_info=True)
                    selector._thompson = BanditSelector()

            return selector
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            _logger.warning("contextual_from_dict_failed", exc_info=True)
            return cls(feature_dim=2, alpha=1.0)

    def _select_decision_with_score(
        self,
        eligible_ids: list[str],
        *,
        context_vector: list[float] | None,
    ) -> tuple[BanditDecision, float]:
        """Shared implementation for tuple- and decision-based selection."""
        if not eligible_ids:
            msg = "eligible_ids must not be empty"
            raise ValueError(msg)

        if not context_vector:
            decision = self._thompson.select(eligible_ids)
            _logger.info(
                "linucb_no_context_thompson_fallback",
                selected=decision.selected_id,
                n_eligible=len(eligible_ids),
                selection_probability=round(decision.selection_probability, 4),
            )
            return decision, decision.selection_probability

        if len(context_vector) != self._feature_dim:
            raise ValueError(f"context_vector dimension {len(context_vector)} != expected {self._feature_dim}")

        x = context_vector
        scores: dict[str, float] = {}
        for arm_id in eligible_ids:
            arm = self._ensure_arm(arm_id)
            theta = _mat_vec_mul(arm.A_inv, arm.b)
            pred = _vec_dot(theta, x)
            a_inv_x = _mat_vec_mul(arm.A_inv, x)
            exploration = self._alpha * math.sqrt(max(0.0, _vec_dot(x, a_inv_x)))
            scores[arm_id] = pred + exploration

        score_vals = list(scores.values())
        max_score = max(score_vals)
        min_score = min(score_vals)
        ref = max(abs(max_score), abs(min_score), 1e-12)
        relative_spread = (max_score - min_score) / ref

        if relative_spread < 0.01:
            decision = self._thompson.select(eligible_ids)
            _logger.debug(
                "contextual_degenerate_scores",
                spread=round(relative_spread, 6),
                selected=decision.selected_id,
            )
            return decision, decision.selection_probability

        ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)
        selected = ranked_ids[0]
        runner_up_id = ranked_ids[1] if len(ranked_ids) > 1 else None
        probabilities = _normalize_scores(scores)
        _logger.debug(
            "contextual_linucb_select",
            selected=selected,
            score=round(scores[selected], 4),
        )
        return (
            BanditDecision(
                selected_id=selected,
                selection_probability=probabilities[selected],
                runner_up_id=runner_up_id,
                runner_up_probability=(probabilities.get(runner_up_id) if runner_up_id is not None else None),
                exploration=False,
            ),
            scores[selected],
        )
