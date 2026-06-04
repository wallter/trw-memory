"""Lifecycle management — scoring, tiers, dedup, consolidation."""

from trw_memory.lifecycle.consolidation import (
    _redact_paths,
    complete_linkage_cluster,
    consolidate_cycle,
    find_clusters,
)
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
    converge_tier_distribution,
    enforce_tier_distribution,
    entry_utility,
    persist_tier_convergence,
    rank_by_utility,
    update_q_value,
    utility_based_prune_candidates,
)
from trw_memory.lifecycle.tiers import TierManager, TierSweepResult

__all__ = [
    "DedupResult",
    "TierManager",
    "TierSweepResult",
    "_redact_paths",
    "apply_time_decay",
    "batch_dedup",
    "bayesian_calibrate",
    "check_duplicate",
    "complete_linkage_cluster",
    "compute_utility_score",
    "consolidate_cycle",
    "converge_tier_distribution",
    "enforce_tier_distribution",
    "entry_utility",
    "find_clusters",
    "merge_entries",
    "persist_tier_convergence",
    "rank_by_utility",
    "update_q_value",
    "utility_based_prune_candidates",
]
