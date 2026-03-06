"""Tiered memory storage: Hot (LRU) / Warm (sqlite-vec sidecar) / Cold (YAML archive).

Implements lifecycle management for memory entries with automatic tier
transitions based on recency and importance scores.

Tier definitions:
- Hot: in-memory LRU cache (OrderedDict, O(1) ops)
- Warm: sqlite-vec backed persistent index or JSONL keyword sidecar
- Cold: YAML archive partitioned by {YYYY}/{MM}/
"""

from trw_memory.lifecycle.tiers._manager import TierManager
from trw_memory.lifecycle.tiers._scoring import TierSweepResult, compute_importance_score

__all__ = [
    "TierManager",
    "TierSweepResult",
    "compute_importance_score",
]
