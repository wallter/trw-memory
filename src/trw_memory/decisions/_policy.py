"""Advisory routing policy and the reliability (calibration) helper.

Belongs to the ``toolkit.py`` facade; re-exported there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from trw_memory.decisions._results import DEAD_BAND, MIN_MARGIN, ClassifyResult, Route


@dataclass(frozen=True)
class Policy:
    """Threshold-and-escalate, advisory. ``act_at`` must be fitted on held-out labels from THIS
    task: absolute scores do not transfer (AUC 1.000 in-sample fell to 0.766 held out)."""

    act_at: float
    escalate_below: float | None = None
    dead_band: float = DEAD_BAND
    min_margin: float = MIN_MARGIN

    def __post_init__(self) -> None:
        for name in ("act_at", "dead_band", "min_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite value in [0, 1], got {value!r}")

    def route(self, probability: float | None, *, margin: float | None = None) -> Route:
        if probability is None:
            return "abstain"
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be a finite value in [0, 1], got {probability!r}")
        if margin is not None and (not math.isfinite(margin) or margin < self.min_margin):
            return "escalate"
        if abs(probability - self.act_at) <= self.dead_band:
            return "escalate"
        if probability >= self.act_at:
            return "act"
        if self.escalate_below is not None and probability >= self.escalate_below:
            return "escalate"
        return "abstain"

    def decide(self, result: ClassifyResult) -> Route:
        """Route a classification with its own margin -- the one call site that cannot forget it."""
        if not result.answered:
            return "abstain"
        return self.route(result.probabilities.get(result.label or "", 0.0), margin=result.margin)


def reliability(predictions: Sequence[tuple[float, bool]], *, bins: int = 10) -> dict[str, Any]:
    """Reliability curve, ECE and Brier from (probability, was_correct) pairs.

    Run on held-out labels from YOUR task before trusting any threshold: measured calibration
    ranged from usable (98.2% in the top bucket) to badly overconfident (ECE 0.27–0.31).
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    for p, ok in predictions:
        if not isinstance(ok, bool):
            raise TypeError(f"labels must be bool, got {ok!r}")
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError(f"probabilities must be finite in [0, 1], got {p!r}")
    if not predictions:
        return {"n": 0, "bins": [], "ece": None, "brier": None}
    rows: list[dict[str, Any]] = []
    total = len(predictions)
    ece = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        bucket = [p for p in predictions if (lo <= p[0] < hi) or (index == bins - 1 and p[0] == 1.0)]
        if not bucket:
            continue
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += (len(bucket) / total) * abs(mean_p - accuracy)
        rows.append({"lo": lo, "hi": hi, "n": len(bucket), "mean_p": mean_p, "accuracy": accuracy})
    brier = sum((p - (1.0 if ok else 0.0)) ** 2 for p, ok in predictions) / total
    return {"n": total, "bins": rows, "ece": ece, "brier": brier, "rmse": math.sqrt(brier)}
