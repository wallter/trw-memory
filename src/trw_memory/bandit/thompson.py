"""Thompson Sampling bandit selector with sliding window (PRD-CORE-105-FR01).

A general-purpose multi-armed bandit using Thompson Sampling with Beta
conjugate priors, a sliding observation window, cold-start round-robin,
and floor-rate exploration.

No numpy dependency -- uses ``random.betavariate`` from the standard library.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ArmState:
    """Internal state for a single bandit arm.

    Uses optimistic Beta(2, 1) priors so that new arms are explored
    before exploitation kicks in.
    """

    alpha: float = 2.0
    beta: float = 1.0
    window: list[float] = field(default_factory=list)
    exposure_count: int = 0


@dataclass
class BanditDecision:
    """Result of a bandit ``select()`` call."""

    selected_id: str
    selection_probability: float
    runner_up_id: str | None
    runner_up_probability: float | None
    exploration: bool  # True if cold-start round-robin or floor-rate exploration


# ---------------------------------------------------------------------------
# BanditSelector
# ---------------------------------------------------------------------------


class BanditSelector:
    """Thompson Sampling selector with sliding window and exploration floor.

    Args:
        tau: Sliding window size -- only the most recent ``tau`` observations
            are used to compute each arm's Beta posterior.
        cold_start_min: Guaranteed minimum exposures for every new arm
            before normal Thompson Sampling takes over.
        floor_exploration: Minimum exploration rate (e.g. 0.12 = 12%).
            Even when one arm dominates, with this probability a random
            non-top arm is selected instead.
    """

    _MAX_POOL_SIZE = 500

    def __init__(
        self,
        tau: int = 25,
        cold_start_min: int = 3,
        floor_exploration: float = 0.12,
    ) -> None:
        if tau < 1:
            raise ValueError(f"tau must be >= 1, got {tau}")
        if cold_start_min < 0:
            raise ValueError(f"cold_start_min must be >= 0, got {cold_start_min}")
        if not 0.0 <= floor_exploration <= 1.0:
            raise ValueError(
                f"floor_exploration must be in [0.0, 1.0], got {floor_exploration}"
            )
        self._tau = tau
        self._cold_start_min = cold_start_min
        self._floor_exploration = floor_exploration
        self._arms: dict[str, ArmState] = {}

    # -- public API ---------------------------------------------------------

    def select(self, eligible_ids: list[str]) -> BanditDecision:
        """Select an arm from *eligible_ids* using Thompson Sampling.

        Raises:
            ValueError: If *eligible_ids* is empty.
        """
        if not eligible_ids:
            msg = "eligible_ids must not be empty"
            raise ValueError(msg)

        # Cap pool size to prevent combinatorial explosion
        if len(eligible_ids) > self._MAX_POOL_SIZE:
            logger.warning(
                "bandit_pool_capped",
                original=len(eligible_ids),
                capped=self._MAX_POOL_SIZE,
            )
            eligible_ids = eligible_ids[: self._MAX_POOL_SIZE]

        # Ensure all eligible arms exist
        for arm_id in eligible_ids:
            if arm_id not in self._arms:
                self._arms[arm_id] = ArmState()

        # --- Cold-start: round-robin for under-exposed arms ----------------
        cold_arms = [
            arm_id
            for arm_id in eligible_ids
            if self._arms[arm_id].exposure_count < self._cold_start_min
        ]
        if cold_arms:
            # Pick the least-exposed cold-start arm (round-robin)
            selected = min(cold_arms, key=lambda a: self._arms[a].exposure_count)
            sample_val = random.betavariate(
                self._arms[selected].alpha,
                self._arms[selected].beta,
            )
            runner_up_id, runner_up_prob = self._compute_runner_up(
                eligible_ids, selected,
            )
            logger.debug(
                "cold_start_selection",
                arm=selected,
                exposure_count=self._arms[selected].exposure_count,
            )
            return BanditDecision(
                selected_id=selected,
                selection_probability=sample_val,
                runner_up_id=runner_up_id,
                runner_up_probability=runner_up_prob,
                exploration=True,
            )

        # --- Normal Thompson Sampling --------------------------------------
        samples: dict[str, float] = {}
        for arm_id in eligible_ids:
            arm = self._arms[arm_id]
            samples[arm_id] = random.betavariate(arm.alpha, arm.beta)

        sorted_arms = sorted(samples, key=lambda a: samples[a], reverse=True)
        top_arm = sorted_arms[0]

        # --- Floor exploration: override top arm with probability ----------
        exploration = False
        selected = top_arm
        if len(eligible_ids) > 1 and random.random() < self._floor_exploration:
            non_top = [a for a in eligible_ids if a != top_arm]
            selected = random.choice(non_top)
            exploration = True
            logger.debug(
                "floor_exploration_triggered",
                original=top_arm,
                selected=selected,
            )

        runner_up_id, runner_up_prob = self._compute_runner_up(
            eligible_ids, selected, samples=samples,
        )

        return BanditDecision(
            selected_id=selected,
            selection_probability=samples[selected],
            runner_up_id=runner_up_id,
            runner_up_probability=runner_up_prob,
            exploration=exploration,
        )

    def update(self, arm_id: str, reward: float) -> None:
        """Record an observation for *arm_id*.

        The reward is clamped to [0.0, 1.0].  If the observation window
        exceeds ``tau``, the oldest observation is evicted before the new
        one is appended.  Alpha and beta are then recomputed from the
        full window.
        """
        reward = max(0.0, min(1.0, reward))

        if arm_id not in self._arms:
            self._arms[arm_id] = ArmState()
        arm = self._arms[arm_id]

        # Sliding window eviction
        if len(arm.window) >= self._tau:
            arm.window.pop(0)

        arm.window.append(reward)

        # Recompute posterior from window
        arm.alpha = 2.0 + sum(arm.window)
        arm.beta = 1.0 + sum(1.0 - r for r in arm.window)

        arm.exposure_count += 1

    def to_json(self) -> str:
        """Serialize the selector state (hyperparameters + arms) to JSON."""
        data: dict[str, Any] = {
            "tau": self._tau,
            "cold_start_min": self._cold_start_min,
            "floor_exploration": self._floor_exploration,
            "arms": {
                arm_id: {
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                    "window": arm.window,
                    "exposure_count": arm.exposure_count,
                }
                for arm_id, arm in self._arms.items()
            },
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> BanditSelector:
        """Deserialize a selector from JSON.

        Returns a fresh instance if the JSON is corrupt or has an
        unexpected structure.
        """
        try:
            parsed = json.loads(data)
            if not isinstance(parsed, dict) or "arms" not in parsed:
                logger.warning("bandit_deserialize_missing_keys")
                return cls()

            selector = cls(
                tau=parsed.get("tau", 25),
                cold_start_min=parsed.get("cold_start_min", 3),
                floor_exploration=parsed.get("floor_exploration", 0.12),
            )
            arms_data = parsed["arms"]
            if not isinstance(arms_data, dict):
                logger.warning("bandit_deserialize_bad_arms_type")
                return cls()

            for arm_id, arm_dict in arms_data.items():
                selector._arms[arm_id] = ArmState(
                    alpha=float(arm_dict.get("alpha", 2.0)),
                    beta=float(arm_dict.get("beta", 1.0)),
                    window=[float(v) for v in arm_dict.get("window", [])],
                    exposure_count=int(arm_dict.get("exposure_count", 0)),
                )
            return selector
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            logger.warning("bandit_deserialize_corrupt", exc_info=True)
            return cls()

    # -- private helpers ----------------------------------------------------

    def _compute_runner_up(
        self,
        eligible_ids: list[str],
        selected_id: str,
        *,
        samples: dict[str, float] | None = None,
    ) -> tuple[str | None, float | None]:
        """Return the runner-up arm and its sampled probability.

        If there is only one eligible arm, returns ``(None, None)``.
        """
        if len(eligible_ids) < 2:
            return None, None

        if samples is None:
            # Generate samples for runner-up computation (cold-start path)
            samples = {}
            for arm_id in eligible_ids:
                if arm_id != selected_id:
                    arm = self._arms[arm_id]
                    samples[arm_id] = random.betavariate(arm.alpha, arm.beta)
            if not samples:
                return None, None
            runner_up = max(samples, key=lambda a: samples[a])
            return runner_up, samples[runner_up]

        # Normal path: use existing samples
        others = {a: v for a, v in samples.items() if a != selected_id}
        if not others:
            return None, None
        runner_up = max(others, key=lambda a: others[a])
        return runner_up, others[runner_up]
