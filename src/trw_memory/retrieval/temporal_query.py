"""Temporal query classifier for automatic recency boost.

Detects temporal language in queries so callers can auto-populate
``recency_weight`` and ``as_of`` without user configuration.

The classifier is intentionally conservative (high precision, not recall):
false positives incorrectly degrade exact-match precision for non-temporal
queries, which is worse than missing a temporal query.

Patterns captured:
- Relative recency adverbs: recently, lately, just, now
- Relative time phrases: last week/month/year, past N days/weeks
- Superlative recency: newest, most recent, latest, current
- Present-state anchors: today, this week/month/year
- Temporal verbs: was updated, has changed, is now

Returns a dataclass with the detected ``is_temporal`` flag plus a suggested
``recency_weight`` scaled by detection confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TemporalClassification:
    is_temporal: bool
    confidence: float
    recency_weight: float
    matched_patterns: list[str] = field(default_factory=list)


# Pattern groups ordered from most specific (high confidence) to broadest.
# Each tuple is (regex, confidence_contribution, label).
_PATTERNS: list[tuple[re.Pattern[str], float, str]] = [
    # Superlative recency — very strong signal
    (re.compile(r"\b(latest|newest|most recent|current|up[- ]to[- ]date)\b", re.IGNORECASE), 0.9, "superlative"),
    # Explicit relative window — strong
    (re.compile(r"\blast\s+\d+\s+(day|week|month|year)s?\b", re.IGNORECASE), 0.85, "relative_window_n"),
    # Named recent windows — strong
    (re.compile(r"\b(last|past|previous)\s+(week|month|year|quarter|sprint|cycle|session)\b", re.IGNORECASE), 0.8, "relative_window"),
    # Today / this-period anchors — moderate
    (re.compile(r"\b(today|this (week|month|year|quarter|sprint|session))\b", re.IGNORECASE), 0.7, "present_anchor"),
    # Recency adverbs — moderate
    (re.compile(r"\b(recently|lately|just now|just\b|right now)\b", re.IGNORECASE), 0.65, "recency_adverb"),
    # Temporal state verbs — moderate
    (re.compile(r"\b(was updated|has changed|is now|have changed|were updated)\b", re.IGNORECASE), 0.65, "state_verb"),
    # Future anchors (lower weight — user wants upcoming, not past recency)
    (re.compile(r"\b(upcoming|next (week|month|quarter|sprint))\b", re.IGNORECASE), 0.4, "future_anchor"),
]

# Combined confidence ceiling — multiple weak signals can stack but are capped
_MAX_CONFIDENCE: float = 0.95
# Minimum confidence before we call the query temporal
_TEMPORAL_THRESHOLD: float = 0.5
# Maximum recency_weight we ever auto-suggest (keep user override in control)
_MAX_AUTO_RECENCY_WEIGHT: float = 0.6


def classify_temporal(query: str) -> TemporalClassification:
    """Classify whether *query* contains temporal language.

    Returns a :class:`TemporalClassification` with ``is_temporal``,
    ``confidence``, ``recency_weight`` (0.0–0.6), and which patterns fired.

    Confidence is the capped sum of all matching pattern contributions.
    ``recency_weight`` scales linearly from 0 at the threshold to
    ``_MAX_AUTO_RECENCY_WEIGHT`` at full confidence.

    Args:
        query: Free-text recall query to classify.

    Returns:
        :class:`TemporalClassification` — always succeeds, never raises.
    """
    matched: list[str] = []
    total_confidence: float = 0.0

    for pattern, contrib, label in _PATTERNS:
        if pattern.search(query):
            matched.append(label)
            total_confidence += contrib

    confidence = min(total_confidence, _MAX_CONFIDENCE)
    is_temporal = confidence >= _TEMPORAL_THRESHOLD

    if not is_temporal:
        return TemporalClassification(
            is_temporal=False,
            confidence=confidence,
            recency_weight=0.0,
            matched_patterns=matched,
        )

    # Scale recency_weight linearly from threshold to max
    scale = (confidence - _TEMPORAL_THRESHOLD) / (_MAX_CONFIDENCE - _TEMPORAL_THRESHOLD)
    recency_weight = round(_MAX_AUTO_RECENCY_WEIGHT * scale, 3)
    recency_weight = min(recency_weight, _MAX_AUTO_RECENCY_WEIGHT)

    return TemporalClassification(
        is_temporal=True,
        confidence=round(confidence, 3),
        recency_weight=recency_weight,
        matched_patterns=matched,
    )
