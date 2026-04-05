"""Multi-armed bandit primitives for adaptive selection.

Public API:

- ``BanditSelector`` / ``BanditDecision`` -- Thompson Sampling with sliding
  window, cold-start round-robin, and floor-rate exploration.
- ``ContextualBanditSelector`` -- LinUCB contextual bandit with
  Sherman-Morrison incremental updates.
- ``PageHinkleyDetector`` -- change-point detection for non-stationary
  reward streams.
"""

from __future__ import annotations

from trw_memory.bandit.change_detection import PageHinkleyDetector
from trw_memory.bandit.contextual import ContextualBanditSelector
from trw_memory.bandit.thompson import BanditDecision, BanditSelector

__all__ = [
    "BanditDecision",
    "BanditSelector",
    "ContextualBanditSelector",
    "PageHinkleyDetector",
]
