"""Lifecycle management — scoring, tiers, dedup, consolidation."""

from trw_memory.lifecycle._recall import rank_by_utility, utility_based_prune_candidates
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
from trw_memory.lifecycle.protection import (
    EXEMPT_PROTECTION_TIERS,
    entry_protection_tier,
    is_removal_exempt,
    prune_threshold_multiplier,
)
from trw_memory.lifecycle.scoring import (
    apply_time_decay,
    bayesian_calibrate,
    compute_utility_score,
    enforce_tier_distribution,
    entry_utility,
    update_q_value,
)
from trw_memory.lifecycle.tiers import TierManager, TierSweepResult

__all__ = [
    "EXEMPT_PROTECTION_TIERS",
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
    "enforce_tier_distribution",
    "entry_protection_tier",
    "entry_utility",
    "find_clusters",
    "is_removal_exempt",
    "merge_entries",
    "prune_threshold_multiplier",
    "rank_by_utility",
    "update_q_value",
    "utility_based_prune_candidates",
]
