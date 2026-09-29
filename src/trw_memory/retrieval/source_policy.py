"""Source-aware recall policy helpers for multi-source retrieval.

Also the one home of the distilled-lesson weight (PRD-CORE-336 FR01):
``weight_distilled`` is the single step that multiplies ``git_distilled`` rows,
called by ``hybrid_search_scored`` for the library and daemon routes alike, and
``SourcePolicy.resolve`` is the only reader of its env override (FR03).
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import structlog

from trw_memory.retrieval.validity_prior import expiry_has_passed

if TYPE_CHECKING:
    from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger(__name__)

SourceFamily = str
#: Operator override for the ``git_distilled`` weight, validated to [0, 1].
DISTILLED_WEIGHT_ENV = "TRW_MEMORY_DISTILLED_RECALL_WEIGHT"
_TRANSIENT_SOURCE_FAMILIES = frozenset({"lifecycle", "episodic"})

DEFAULT_SOURCE_WEIGHTS: dict[SourceFamily, float] = {
    "git_distilled": 0.75,
    "instruction_rule": 0.95,
    "semantic_memory": 0.9,
    "lifecycle": 0.55,
    "episodic": 0.45,
    "unknown": 1.0,
}


def classify_source_family(result: Mapping[str, object]) -> SourceFamily:
    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        explicit = str(metadata.get("source_kind", "")).strip()
        if explicit == "git":
            return "git_distilled"
        if explicit in {"instruction_rule", "semantic_memory", "lifecycle", "episodic"}:
            return explicit
        source = str(metadata.get("source", "")).strip()
        if source.startswith("distilled:git:"):
            return "git_distilled"
        if source.startswith("distilled:bulletin:"):
            return "lifecycle"
    tags = result.get("tags", []) or []
    for tag in cast("Iterable[object]", tags):
        if not isinstance(tag, str):
            continue
        if tag.startswith("source_kind:"):
            family = tag.split(":", 1)[1]
            if family == "git":
                return "git_distilled"
            if family in {"instruction_rule", "semantic_memory", "lifecycle", "episodic"}:
                return family
        if tag.startswith(("distill:", "distilled:")):
            return "git_distilled"
        if tag == "change_bulletin":
            return "lifecycle"
    return "unknown"


def distilled_weight_from_env() -> float | None:
    """The env override, or ``None`` when unset or invalid (logged; the default then applies)."""
    raw = os.environ.get(DISTILLED_WEIGHT_ENV, "").strip()
    if not raw:
        return None
    try:
        weight = float(raw)
    except ValueError:  # trw-fail-silent-allow: a bad operator override is logged; the default applies
        logger.warning("distilled_recall_weight_invalid", raw=raw)
        return None
    if not (math.isfinite(weight) and 0.0 <= weight <= 1.0):
        logger.warning("distilled_recall_weight_out_of_range", raw=raw)
        return None
    return weight


def weight_distilled(
    entries: Sequence[MemoryEntry], scores: Sequence[float], weight: float
) -> list[tuple[MemoryEntry, float, bool]]:
    """Multiply ONLY ``git_distilled`` rows' scores by *weight*, then stably re-sort.

    Returns ``(entry, score, weighted)`` triples by descending score; ties keep
    the input order, and every other row keeps its score, so non-distilled rows
    never change order relative to each other. Inputs are not mutated.
    """
    weighted: list[tuple[MemoryEntry, float, bool]] = []
    for entry, score in zip(entries, scores, strict=True):
        if classify_source_family({"metadata": entry.metadata, "tags": entry.tags}) == "git_distilled":
            weighted.append((entry, score * weight, True))
        else:
            weighted.append((entry, score, False))
    weighted.sort(key=lambda row: row[1], reverse=True)
    return weighted


def resolve_expiry(result: Mapping[str, object]) -> str:
    raw = result.get("expires")
    if isinstance(raw, str) and raw:
        return raw
    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        meta_expiry = metadata.get("expires")
        if isinstance(meta_expiry, str):
            return meta_expiry
    return ""


def is_expired_result(result: Mapping[str, object], *, now: datetime | None = None) -> bool:
    """Use the same day-exclusive expiry contract as temporal selection."""
    return expiry_has_passed(resolve_expiry(result), reference_time=now)


@dataclass(frozen=True)
class SourcePolicy:
    """Immutable per-invocation source admission and ranking policy.

    Construct with ``resolve`` to snapshot caller options and the clock once.
    Admission is score-independent, so acquisition and ranking can share it.
    """

    include_distilled: bool
    include_kinds: frozenset[str]
    exclude_kinds: frozenset[str]
    weights: Mapping[str, float]
    explicit_weight_overrides: frozenset[str]
    exclude_expired: bool
    reference_time: datetime
    # Preserve presence even when the caller explicitly chooses the default.
    explicit_distilled_weight: bool = False

    @classmethod
    def resolve(
        cls,
        *,
        include_distilled: bool = True,
        distilled_weight: float | None = None,
        include_source_kinds: list[str] | None = None,
        exclude_source_kinds: list[str] | None = None,
        source_weights: dict[str, float] | None = None,
        exclude_expired: bool = True,
        reference_time: datetime | None = None,
    ) -> SourcePolicy:
        weights = dict(DEFAULT_SOURCE_WEIGHTS)
        if source_weights:
            weights.update(source_weights)
        if distilled_weight is not None:
            weights["git_distilled"] = distilled_weight
        elif "git_distilled" not in (source_weights or {}):
            env_weight = distilled_weight_from_env()
            if env_weight is not None:
                weights["git_distilled"] = env_weight
        return cls(
            include_distilled=include_distilled,
            include_kinds=frozenset(include_source_kinds or ()),
            exclude_kinds=frozenset(exclude_source_kinds or ()),
            weights=MappingProxyType(weights),
            explicit_weight_overrides=frozenset(source_weights or ()),
            exclude_expired=exclude_expired,
            reference_time=reference_time or datetime.now(timezone.utc),
            explicit_distilled_weight=distilled_weight is not None,
        )

    def allows(self, result: Mapping[str, object]) -> bool:
        """Apply hard source/expiry/weight exclusions without inspecting score."""
        family = classify_source_family(result)
        if family == "git_distilled" and not self.include_distilled:
            return False
        if self.include_kinds and family not in self.include_kinds:
            return False
        if family in self.exclude_kinds:
            return False
        if (
            self.exclude_expired
            and family in _TRANSIENT_SOURCE_FAMILIES
            and is_expired_result(result, now=self.reference_time)
        ):
            return False
        return not self.weights.get(family, 1.0) <= 0.0

    def rank_key(self, result: Mapping[str, object], *, pipeline_weighted: bool = False) -> tuple[int, float]:
        """Ascending containment and weighted-score key for an admitted raw result.

        *pipeline_weighted* marks a row the pipeline already ranked through
        ``weight_distilled``: its ``git_distilled`` weight is in its order, so it
        is not applied a second time here (PRD-CORE-336 FR01).
        """
        family = classify_source_family(result)
        bucket = 0
        if family in _TRANSIENT_SOURCE_FAMILIES and family not in self.explicit_weight_overrides:
            bucket = 2
        elif str(result.get("source", "")) in {"org", "shared"}:
            bucket = 1
        weight = 1.0 if pipeline_weighted and family == "git_distilled" else self.weights.get(family, 1.0)
        return bucket, -float(cast("float", result.get("score", 0.0))) * weight


__all__ = [
    "DEFAULT_SOURCE_WEIGHTS",
    "DISTILLED_WEIGHT_ENV",
    "SourcePolicy",
    "classify_source_family",
    "distilled_weight_from_env",
    "is_expired_result",
    "resolve_expiry",
    "weight_distilled",
]
