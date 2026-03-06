"""Importance scoring and result types for tiered memory lifecycle.

Provides the composite importance score formula and the TierSweepResult
named tuple used to report sweep outcomes.
"""

from __future__ import annotations

import math
from datetime import date
from typing import NamedTuple

from trw_memory.lifecycle._utils import days_since_access as _days_since_access
from trw_memory.models.config import MemoryConfig
from trw_memory.retrieval.dense import cosine_similarity


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class TierSweepResult(NamedTuple):
    """Outcome of a single sweep() pass across all tiers.

    Attributes:
        promoted: Entries moved up a tier (Cold->Warm).
        demoted: Entries moved down a tier (Hot->Warm, Warm->Cold).
        purged: Entries deleted from Cold tier (retention expired).
        errors: Per-entry failures that were logged and skipped.
    """

    promoted: int
    demoted: int
    purged: int
    errors: int

    @property
    def total(self) -> int:
        """Total number of entries affected by this sweep."""
        return self.promoted + self.demoted + self.purged + self.errors


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------


def compute_importance_score(
    entry: dict[str, object],
    query_tokens: list[str],
    query_embedding: list[float] | None = None,
    entry_embedding: list[float] | None = None,
    *,
    config: MemoryConfig | None = None,
) -> float:
    """Compute a composite importance score for a memory entry.

    Formula: score = w1*relevance + w2*recency + w3*importance

    Weights are normalized if they don't sum to 1.0.

    Args:
        entry: Memory entry as a dict (from YAML or model_dump).
        query_tokens: Tokenized query for token-overlap fallback.
        query_embedding: Optional dense query vector for cosine similarity.
        entry_embedding: Optional dense entry vector for cosine similarity.
        config: MemoryConfig for weights and decay settings.

    Returns:
        Composite importance score in [0.0, 1.0].
    """
    cfg = config or MemoryConfig()

    w1 = cfg.score_relevance_weight
    w2 = cfg.score_recency_weight
    w3 = cfg.score_importance_weight

    # Normalize weights if they don't sum to 1.0
    total_w = w1 + w2 + w3
    if total_w > 0 and abs(total_w - 1.0) > 1e-9:
        w1 /= total_w
        w2 /= total_w
        w3 /= total_w

    # Relevance: cosine similarity when both embeddings present, else token overlap
    if query_embedding is not None and entry_embedding is not None:
        relevance = max(0.0, cosine_similarity(query_embedding, entry_embedding))
    else:
        # Token overlap ratio fallback
        entry_text = (
            str(entry.get("content", "")).lower()
            + " "
            + str(entry.get("detail", "")).lower()
        )
        entry_tokens = set(entry_text.split())
        query_set = {t.lower() for t in query_tokens}
        if query_set:
            relevance = len(query_set & entry_tokens) / len(query_set)
        else:
            relevance = 0.0

    # Recency: exponential decay based on days since access
    today = date.today()
    days = _days_since_access(entry, today)
    half_life = cfg.decay_half_life_days
    decay_rate = math.log(2) / half_life if half_life > 0 else 0.0
    recency = math.exp(-decay_rate * days)

    # Importance: the entry's importance field (was 'impact' in LearningEntry)
    # Support both 'importance' (MemoryEntry) and 'impact' (legacy LearningEntry)
    raw_importance = entry.get("importance", entry.get("impact", 0.5))
    importance = float(str(raw_importance))
    importance = max(0.0, min(1.0, importance))

    score = w1 * relevance + w2 * recency + w3 * importance
    return max(0.0, min(1.0, score))
