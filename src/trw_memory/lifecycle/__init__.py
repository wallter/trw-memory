"""Lifecycle management — scoring, tiers, dedup, consolidation."""

from trw_memory.lifecycle.consolidation import consolidate_cycle, find_clusters
from trw_memory.lifecycle.dedup import (
    DedupResult,
    batch_dedup,
    check_duplicate,
    merge_entries,
)
from trw_memory.lifecycle.scoring import (
    apply_time_decay,
    bayesian_calibrate,
    compute_utility_score,
    enforce_tier_distribution,
    entry_utility,
    rank_by_utility,
    update_q_value,
    utility_based_prune_candidates,
)
from trw_memory.lifecycle.tiers import TierManager, TierSweepResult

__all__ = [
    "DedupResult",
    "TierManager",
    "TierSweepResult",
    "apply_time_decay",
    "batch_dedup",
    "bayesian_calibrate",
    "check_duplicate",
    "compute_utility_score",
    "consolidate_cycle",
    "enforce_tier_distribution",
    "entry_utility",
    "find_clusters",
    "merge_entries",
    "rank_by_utility",
    "update_q_value",
    "utility_based_prune_candidates",
]
