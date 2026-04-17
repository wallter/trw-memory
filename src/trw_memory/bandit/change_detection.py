"""Page-Hinkley-style change detection for non-stationary reward streams.

General-purpose primitive -- no TRW/engineering-specific concepts.
Used by consuming applications to detect when an arm's reward distribution shifts.
"""

from __future__ import annotations

import structlog

_logger = structlog.get_logger(__name__)

# JSON cannot encode float("inf"), so we use None as a sentinel for
# uninitialized minimum values in serialized dicts.


class PageHinkleyDetector:
    """Detects distributional changes in sequential reward observations.

    Uses a two-sided exponentially-decayed Page-Hinkley statistic: tracks
    cumulative deviation from the running mean in both directions, applies a
    short burn-in period, and decays stale evidence to keep false alarms low
    on bounded reward streams.

    Args:
        delta: Tolerance for false alarms (higher = less sensitive). Default: 0.01.
        alarm_threshold: Detection threshold (higher = larger shift required). Default: 20.0.
            Calibrated for bounded rewards in [0, 1] with internal decayed
            accumulation so the PRD acceptance sequence still fires while the
            stable false alarm rate stays below the required bound.
    """

    def __init__(self, delta: float = 0.01, alarm_threshold: float = 20.0) -> None:
        self._delta = delta
        self._alarm_threshold = alarm_threshold
        self._forgetting_factor = 0.95
        self._min_observations = 10
        self._reward_scale = 10.0
        self._n: int = 0
        self._sum: float = 0.0
        # Upward cumulative sum and its running minimum.
        self._h: float = 0.0
        self._m: float = float("inf")
        # Downward cumulative sum and its running minimum.
        self._h_down: float = 0.0
        self._m_down: float = float("inf")

    def update(self, reward: float) -> bool:
        """Process a new reward observation and check for change.

        Returns True if a distributional change is detected (alarm fires).
        On detection, internal state is reset for subsequent monitoring.
        """
        reward = max(0.0, min(1.0, reward))
        self._n += 1
        self._sum += reward
        mean = self._sum / self._n

        deviation_up = self._reward_scale * (reward - mean - self._delta)
        deviation_down = self._reward_scale * (mean - reward - self._delta)

        # Two-sided decayed accumulation keeps the detector sensitive to
        # sustained shifts without turning stable noise into certain alarms.
        self._h = max(0.0, (self._forgetting_factor * self._h) + deviation_up)
        self._h_down = max(
            0.0,
            (self._forgetting_factor * self._h_down) + deviation_down,
        )
        self._m = 0.0 if self._m == float("inf") else min(self._m, self._h)
        self._m_down = 0.0 if self._m_down == float("inf") else min(self._m_down, self._h_down)

        up_dev = self._h - self._m
        down_dev = self._h_down - self._m_down
        max_dev = max(up_dev, down_dev)

        if self._n < self._min_observations:
            return False

        if max_dev > self._alarm_threshold:
            _logger.info(
                "page_hinkley_change_detected",
                observations=self._n,
                deviation=round(max_dev, 4),
            )
            self.reset()
            return True
        return False

    def reset(self) -> None:
        """Reset internal state for fresh monitoring."""
        self._n = 0
        self._sum = 0.0
        self._h = 0.0
        self._m = float("inf")
        self._h_down = 0.0
        self._m_down = float("inf")

    def to_dict(self) -> dict[str, int | float | None]:
        """Serialize state for JSON persistence."""
        return {
            "delta": self._delta,
            "alarm_threshold": self._alarm_threshold,
            "forgetting_factor": self._forgetting_factor,
            "min_observations": self._min_observations,
            "reward_scale": self._reward_scale,
            "n": self._n,
            "sum": self._sum,
            "h": self._h,
            "m": self._m if self._m != float("inf") else None,
            "h_down": self._h_down,
            "m_down": self._m_down if self._m_down != float("inf") else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int | float | None]) -> PageHinkleyDetector:
        """Deserialize from dict. Returns fresh detector on malformed input."""
        try:
            delta_raw = data.get("delta")
            alarm_raw = data.get("alarm_threshold")
            det = cls(
                delta=float(delta_raw) if delta_raw is not None else 0.01,
                alarm_threshold=float(alarm_raw) if alarm_raw is not None else 20.0,
            )
            ff_raw = data.get("forgetting_factor")
            if ff_raw is not None:
                det._forgetting_factor = float(ff_raw)
            min_obs_raw = data.get("min_observations")
            if min_obs_raw is not None:
                det._min_observations = int(min_obs_raw)
            reward_scale_raw = data.get("reward_scale")
            if reward_scale_raw is not None:
                det._reward_scale = float(reward_scale_raw)
            n_raw = data.get("n")
            det._n = int(n_raw) if n_raw is not None else 0
            sum_raw = data.get("sum")
            det._sum = float(sum_raw) if sum_raw is not None else 0.0
            h_raw = data.get("h")
            det._h = float(h_raw) if h_raw is not None else 0.0
            m_raw = data.get("m")
            det._m = float("inf") if m_raw is None else float(m_raw)
            h_down_raw = data.get("h_down")
            det._h_down = float(h_down_raw) if h_down_raw is not None else 0.0
            m_down_raw = data.get("m_down")
            det._m_down = float("inf") if m_down_raw is None else float(m_down_raw)
            return det
        except (TypeError, ValueError, KeyError):
            _logger.warning("page_hinkley_from_dict_failed", exc_info=True)
            return cls()
