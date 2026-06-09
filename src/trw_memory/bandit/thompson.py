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
    """Result of a bandit ``select()`` call.

    ``selection_probability`` and ``runner_up_probability`` are estimated
    policy propensities, not single posterior samples.
    """

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
    _PROPENSITY_MONTE_CARLO_SAMPLES = 96
    _PRIOR_ALPHA = 2.0
    _PRIOR_BETA = 1.0

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
            raise ValueError(f"floor_exploration must be in [0.0, 1.0], got {floor_exploration}")
        self._tau = tau
        self._cold_start_min = cold_start_min
        self._floor_exploration = floor_exploration
        self._arms: dict[str, ArmState] = {}
        self._state_version = 0
        self._propensity_cache: dict[tuple[str, ...], tuple[int, dict[str, float]]] = {}

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
                self._state_version += 1

        # --- Cold-start: round-robin for under-exposed arms ----------------
        cold_arms = [arm_id for arm_id in eligible_ids if self._arms[arm_id].exposure_count < self._cold_start_min]
        if cold_arms:
            # Pick the least-exposed cold-start arm (round-robin)
            selected = min(cold_arms, key=lambda a: self._arms[a].exposure_count)
            runner_up_id = self._compute_cold_start_runner_up(eligible_ids, selected)
            logger.debug(
                "cold_start_selection",
                arm=selected,
                exposure_count=self._arms[selected].exposure_count,
            )
            return BanditDecision(
                selected_id=selected,
                selection_probability=1.0,
                runner_up_id=runner_up_id,
                runner_up_probability=0.0 if runner_up_id is not None else None,
                exploration=True,
            )

        # --- Normal Thompson Sampling --------------------------------------
        top_arm: str | None = None
        top_index = 0
        top_sample = -1.0
        runner_up_id = None
        runner_up_sample: float | None = None
        for index, arm_id in enumerate(eligible_ids):
            arm = self._arms[arm_id]
            sample = random.betavariate(arm.alpha, arm.beta)
            if sample > top_sample:
                runner_up_id = top_arm
                runner_up_sample = top_sample if top_arm is not None else None
                top_arm = arm_id
                top_index = index
                top_sample = sample
            elif runner_up_sample is None or sample > runner_up_sample:
                runner_up_id = arm_id
                runner_up_sample = sample

        assert top_arm is not None  # noqa: S101 — guaranteed by eligible_ids non-empty check

        # --- Floor exploration: override top arm with probability ----------
        exploration = False
        selected = top_arm
        if len(eligible_ids) > 1 and random.random() < self._floor_exploration:  # noqa: S311
            selected_index = random.randrange(len(eligible_ids) - 1)  # noqa: S311
            if selected_index >= top_index:
                selected_index += 1
            selected = eligible_ids[selected_index]
            exploration = True
            logger.debug(
                "floor_exploration_triggered",
                original=top_arm,
                selected=selected,
            )

        if selected != top_arm:
            runner_up_id = top_arm
        propensities = self._estimate_propensities(eligible_ids)

        return BanditDecision(
            selected_id=selected,
            selection_probability=propensities[selected],
            runner_up_id=runner_up_id,
            runner_up_probability=(propensities.get(runner_up_id) if runner_up_id is not None else None),
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
        arm.alpha = self._PRIOR_ALPHA + sum(arm.window)
        arm.beta = self._PRIOR_BETA + sum(1.0 - r for r in arm.window)

        arm.exposure_count += 1
        self._invalidate_propensity_cache()

    def soft_reset_arm(self, arm_id: str) -> None:
        """Soft-reset an arm posterior to the optimistic prior.

        Keeps ``exposure_count`` intact so the caller can reset belief without
        forcing the arm back through cold-start round-robin.
        """
        arm = self._arms.get(arm_id)
        if arm is None:
            self._arms[arm_id] = ArmState()
            return
        arm.alpha = self._PRIOR_ALPHA
        arm.beta = self._PRIOR_BETA
        arm.window = []
        self._invalidate_propensity_cache()

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
        unexpected top-level structure. A single malformed arm entry is
        skipped (per-row fail-open) rather than discarding every valid arm,
        matching :meth:`ContextualBanditSelector.from_compact_dict` and the
        warm/retry sidecar readers; the deserialize never raises.
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

            skipped = 0
            for arm_id, arm_dict in arms_data.items():
                if not isinstance(arm_dict, dict):
                    skipped += 1
                    continue
                try:
                    selector._arms[str(arm_id)] = ArmState(
                        alpha=float(arm_dict.get("alpha", cls._PRIOR_ALPHA)),
                        beta=float(arm_dict.get("beta", cls._PRIOR_BETA)),
                        window=[float(v) for v in arm_dict.get("window", [])],
                        exposure_count=int(arm_dict.get("exposure_count", 0)),
                    )
                except (TypeError, ValueError):
                    # Content-free per-arm fail-open: one arm with non-numeric
                    # fields (or a non-iterable window) must not abort the
                    # restore of the remaining valid arms.
                    skipped += 1
            if skipped:
                logger.warning("bandit_deserialize_skipped_arms", skipped=skipped)
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

    def _compute_cold_start_runner_up(
        self,
        eligible_ids: list[str],
        selected_id: str,
    ) -> str | None:
        """Return the deterministic cold-start fallback arm."""
        candidates = [arm_id for arm_id in eligible_ids if arm_id != selected_id]
        if not candidates:
            return None
        return min(candidates, key=lambda arm_id: self._arms[arm_id].exposure_count)

    def _estimate_propensities(self, eligible_ids: list[str]) -> dict[str, float]:
        """Estimate per-arm selection propensities via Monte Carlo simulation."""
        if len(eligible_ids) == 1:
            return {eligible_ids[0]: 1.0}

        cache_key = tuple(eligible_ids)
        cached = self._propensity_cache.get(cache_key)
        if cached is not None and cached[0] == self._state_version:
            return cached[1]

        betavariate = random.betavariate
        random_float = random.random
        randrange = random.randrange
        floor_exploration = self._floor_exploration
        sample_count = self._PROPENSITY_MONTE_CARLO_SAMPLES
        arm_count = len(eligible_ids)

        alphas = [0.0] * arm_count
        betas = [0.0] * arm_count
        for index, arm_id in enumerate(eligible_ids):
            arm = self._arms[arm_id]
            alphas[index] = arm.alpha
            betas[index] = arm.beta

        counts = [0] * arm_count

        if floor_exploration <= 0.0:
            for _ in range(sample_count):
                top_index = 0
                top_sample = betavariate(alphas[0], betas[0])
                for index in range(1, arm_count):
                    sample = betavariate(alphas[index], betas[index])
                    if sample > top_sample:
                        top_index = index
                        top_sample = sample
                counts[top_index] += 1
        else:
            for _ in range(sample_count):
                top_index = 0
                top_sample = betavariate(alphas[0], betas[0])
                for index in range(1, arm_count):
                    sample = betavariate(alphas[index], betas[index])
                    if sample > top_sample:
                        top_index = index
                        top_sample = sample

                selected_index = top_index
                if random_float() < floor_exploration:
                    selected_index = randrange(arm_count - 1)
                    if selected_index >= top_index:
                        selected_index += 1
                counts[selected_index] += 1

        sample_total = float(sample_count)
        result = {arm_id: counts[index] / sample_total for index, arm_id in enumerate(eligible_ids)}
        self._propensity_cache[cache_key] = (self._state_version, result)
        return result

    def _invalidate_propensity_cache(self) -> None:
        self._state_version += 1
        self._propensity_cache.clear()
