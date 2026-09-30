"""The fields a consolidated entry derives from its cluster (PRD-CORE-099, PRD-FIX-114).

Split out of ``consolidation`` (pure move, no behaviour change). ``_create_consolidated_entry`` adds the
content and detail, builds the entry and writes it.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from trw_memory.lifecycle.dedup import _stronger_protection_tier, _union_assertions
from trw_memory.models.memory import MemoryEntry, MemoryStatus, ProtectionTier

#: The most tags a consolidated entry carries: the most frequent across its cluster, ties by name.
CONSOLIDATION_TAGS_MAX = 20


def capped_tags(cluster: list[MemoryEntry]) -> list[str]:
    """The ``CONSOLIDATION_TAGS_MAX`` tags most members share, sorted (ties break by name)."""
    counts = Counter(tag for entry in cluster for tag in set(entry.tags))
    keep = sorted(counts, key=lambda tag: (-counts[tag], tag))[:CONSOLIDATION_TAGS_MAX]
    return sorted(keep)


def merged_entry_fields(cluster: list[MemoryEntry]) -> dict[str, Any]:
    """Provenance, counts, tags, evidence, protection and assertions merged from *cluster*."""
    # Inherit provenance from highest-importance source (PRD-CORE-099)
    best_source = max(cluster, key=lambda e: e.importance)

    # Preserve maintenance evidence, not verification of the newly changed claim.
    # Reuse dedup's tier ordering and assertion identity rather than allowing
    # the two maintenance paths to evolve incompatible preservation policies.
    protection_tier: ProtectionTier | str = cluster[0].protection_tier
    assertions = cluster[0].assertions
    for source_entry in cluster[1:]:
        protection_tier = _stronger_protection_tier(protection_tier, source_entry.protection_tier)
        assertions = _union_assertions(assertions, source_entry.assertions)
    assertions = [
        assertion.model_copy(
            update={"last_result": None, "last_verified_at": None, "last_evidence": "", "first_failed_at": None}
        )
        for assertion in assertions
    ]
    return {
        "source": "consolidated",
        "source_identity": best_source.source_identity,
        "client_profile": best_source.client_profile,
        "model_id": best_source.model_id,
        "consolidated_from": [e.id for e in cluster],
        "importance": max(e.importance for e in cluster),
        "tags": capped_tags(cluster),
        "evidence": list(dict.fromkeys(ev for e in cluster for ev in e.evidence)),
        "recurrence": len(cluster),
        "access_count": sum(e.access_count for e in cluster),
        "recall_count": sum(e.recall_count for e in cluster),
        "protection_tier": protection_tier,
        "assertions": assertions,
        "status": MemoryStatus.ACTIVE,
    }
