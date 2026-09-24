"""Multi-armed bandit primitives for adaptive selection.

Public API:

- ``PageHinkleyDetector`` -- change-point detection for non-stationary
  reward streams.
"""

from __future__ import annotations

from trw_memory.bandit.change_detection import PageHinkleyDetector

__all__ = [
    "PageHinkleyDetector",
]
