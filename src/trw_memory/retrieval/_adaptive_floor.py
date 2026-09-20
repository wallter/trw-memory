"""Confidence floor for cross-encoder re-ranked recall, scaled by the requested limit.

PRD-CORE-284. After the cross-encoder scores the fused pool, recall drops rows
whose logit is below :data:`RERANK_MIN_SCORE`, except the top ``min_keep`` rows,
which are always kept so a low-confidence query still answers. ``min_keep`` used
to be a fixed 5 whatever the caller asked for, which starved a wide
``limit=50`` evidence-gathering call down to the same handful of guaranteed
rows as a ``limit=10`` answer-reading call.

``min_keep`` now scales with the limit: ``max(5, ceil(limit * 0.5))``, with no
cap at ``limit``. For every ``limit`` in 1..10 that is exactly 5, the legacy
value, so direct small-limit callers see no change (a keep larger than the scored
pool is a no-op, and the pipeline over-fetches before trimming to ``limit``); at
``limit=50`` it is 25.

The constants are deliberately not configuration: the three ``recall_rerank*``
settings this replaces were removed rather than joined by a fourth knob.
"""

from __future__ import annotations

import math
from typing import NamedTuple

__all__ = ["RERANK_MIN_SCORE", "RerankFloor", "adaptive_rerank_floor"]

#: ms-marco MiniLM logit below which a re-ranked row is dropped (about -11 is
#: unrelated text, +10 an exact answer). Calibrated on LOCOMO top-50 pools.
RERANK_MIN_SCORE = -8.0
#: Rows always kept at small limits -- the legacy fixed ``min_keep``.
_BASE_MIN_KEEP = 5
#: Fraction of a larger limit exempt from the score floor.
_KEEP_FRACTION = 0.5


class RerankFloor(NamedTuple):
    """Score floor and the number of top-ranked rows exempt from it."""

    min_score: float
    min_keep: int


def adaptive_rerank_floor(limit: int) -> RerankFloor:
    """Return the confidence floor for a recall that asked for *limit* rows."""
    if limit <= 0:
        return RerankFloor(RERANK_MIN_SCORE, 0)
    return RerankFloor(RERANK_MIN_SCORE, max(_BASE_MIN_KEEP, math.ceil(limit * _KEEP_FRACTION)))
