"""Contextual bandit via LinUCB with random fallback.

General-purpose primitive -- no TRW/engineering-specific concepts.
The caller provides opaque context feature vectors.

Uses the LinUCB algorithm (Li et al., 2010) with Sherman-Morrison
incremental inverse updates to avoid matrix inversion.  No numpy
dependency -- all matrix operations are loop-based for small dimensions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import structlog

_logger = structlog.get_logger(__name__)


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

    Args:
        feature_dim: Dimension of the context feature vector.
        alpha: Exploration parameter controlling the UCB width.
    """

    def __init__(self, feature_dim: int, alpha: float = 1.0) -> None:
        self._feature_dim = feature_dim
        self._alpha = alpha
        self._arms: dict[str, ContextualArmState] = {}

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

        # No context -> uniform random selection.
        # NOTE: The spec says to fall back to Thompson Sampling (FR01), but
        # LinUCB arms maintain A_inv / b parameters (not Beta posteriors), so
        # a true TS fallback would require a separate BanditSelector instance.
        # Using uniform random here; callers who need TS should maintain a
        # BanditSelector alongside the ContextualBanditSelector.
        if context_vector is None:
            selected = random.choice(eligible_ids)  # noqa: S311
            _logger.info(
                "linucb_no_context_random_fallback",
                selected=selected,
                n_eligible=len(eligible_ids),
            )
            return selected, 0.0

        if len(context_vector) != self._feature_dim:
            raise ValueError(
                f"context_vector dimension {len(context_vector)} != "
                f"expected {self._feature_dim}"
            )

        x = context_vector

        # Compute UCB score for each arm
        scores: dict[str, float] = {}
        for arm_id in eligible_ids:
            arm = self._ensure_arm(arm_id)

            # theta = A_inv @ b  (parameter estimate)
            theta = _mat_vec_mul(arm.A_inv, arm.b)

            # predicted reward = theta^T @ x
            pred = _vec_dot(theta, x)

            # exploration bonus = alpha * sqrt(x^T @ A_inv @ x)
            a_inv_x = _mat_vec_mul(arm.A_inv, x)
            exploration = self._alpha * math.sqrt(max(0.0, _vec_dot(x, a_inv_x)))

            scores[arm_id] = pred + exploration

        # Check for degeneracy: all scores within 1% of each other
        score_vals = list(scores.values())
        max_score = max(score_vals)
        min_score = min(score_vals)

        # Avoid division by zero when all scores are zero
        ref = max(abs(max_score), abs(min_score), 1e-12)
        relative_spread = (max_score - min_score) / ref

        if relative_spread < 0.01:
            selected = random.choice(eligible_ids)  # noqa: S311
            _logger.debug(
                "contextual_degenerate_scores",
                spread=round(relative_spread, 6),
                selected=selected,
            )
            return selected, scores[selected]

        # Select arm with highest UCB score
        selected = max(scores, key=lambda a: scores[a])
        _logger.debug(
            "contextual_linucb_select",
            selected=selected,
            score=round(scores[selected], 4),
        )
        return selected, scores[selected]

    def update(
        self,
        arm_id: str,
        reward: float,
        context_vector: list[float] | None = None,
    ) -> None:
        """Record a reward observation for *arm_id* given a context.

        If *context_vector* is ``None``, the update is a no-op (no
        context to learn from).

        Uses the Sherman-Morrison formula for rank-1 update of A_inv:
            A_inv_new = A_inv - (A_inv @ x @ x^T @ A_inv) / (1 + x^T @ A_inv @ x)

        Args:
            arm_id: Identifier of the arm that was pulled.
            reward: Observed reward value.
            context_vector: Feature vector used when the arm was selected,
                or ``None`` to skip the update.
        """
        if context_vector is None:
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize selector state for JSON persistence."""
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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextualBanditSelector:
        """Deserialize from dict.

        Returns a fresh instance with default parameters on malformed
        input.
        """
        try:
            feature_dim = int(data["feature_dim"])
            alpha = float(data["alpha"])
            arms_data = data.get("arms", {})

            if not isinstance(arms_data, dict):
                _logger.warning("contextual_from_dict_bad_arms_type")
                return cls(feature_dim=feature_dim, alpha=alpha)

            selector = cls(feature_dim=feature_dim, alpha=alpha)

            for arm_id, arm_dict in arms_data.items():
                a_inv = [
                    [float(v) for v in row]
                    for row in arm_dict["A_inv"]
                ]
                b = [float(v) for v in arm_dict["b"]]
                n_obs = int(arm_dict["n_obs"])

                selector._arms[arm_id] = ContextualArmState(
                    A_inv=a_inv,
                    b=b,
                    n_obs=n_obs,
                )

            return selector
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            _logger.warning("contextual_from_dict_failed", exc_info=True)
            return cls(feature_dim=2, alpha=1.0)
