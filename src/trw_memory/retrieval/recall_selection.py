"""Invocation-scoped policy and authoritative recall candidates.

No client or transport dependency: storage and tiers consume the same policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.retrieval.admission_policy import apply_admission_filter
from trw_memory.retrieval.source_policy import SourcePolicy
from trw_memory.retrieval.temporal_selection import TemporalSelection
from trw_memory.retrieval.validity_prior import _is_open_at, expiry_has_passed
from trw_memory.security.namespace_scope import NamespaceScopeError


def entry_policy_fields(entry: MemoryEntry, source: str = "local", score: float = 0.0) -> dict[str, object]:
    """A policy view, never a replacement for the authoritative entry."""
    return {
        "metadata": entry.metadata,
        "tags": entry.tags,
        "expires": entry.expires,
        "importance": entry.importance,
        "source": source,
        "score": score,
    }


@dataclass(frozen=True, slots=True)
class LocalCandidate:
    entry: MemoryEntry
    raw_score: float
    source: str = "local"
    relevance_hint: float | None = None
    cold: bool = False
    # Request-local ordering only; never replace raw relevance or persist this flag.
    tier_fallback: bool = False


@dataclass(frozen=True, slots=True)
class RemoteCandidate:
    result: Mapping[str, object]

    def temporal_eligibility(self, temporal: TemporalSelection) -> bool | None:
        """False means contradicted supplied evidence; None is missing evidence."""
        raw_expiry = self.result.get("expires")
        expires = raw_expiry if isinstance(raw_expiry, str) else ""
        if expiry_has_passed(expires, reference_time=temporal.as_of or temporal.reference_time):
            return False
        valid_from = _instant(self.result.get("valid_from"))
        invalid_from = _instant(self.result.get("invalid_from"))
        as_of = temporal.as_of
        if as_of is not None and as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        if invalid_from is not None and (as_of is None or as_of >= invalid_from):
            return False
        if valid_from is not None and as_of is not None and as_of < valid_from:
            return False
        if (
            valid_from is not None
            and "invalid_from" in self.result
            and (self.result["invalid_from"] is None or invalid_from is not None)
        ):
            return _is_open_at(
                _RemoteWindow(valid_from, invalid_from, expires), as_of, reference_time=temporal.reference_time
            )
        return None


@dataclass(frozen=True, slots=True)
class _RemoteWindow:
    valid_from: datetime
    invalid_from: datetime | None
    expires: str


def _instant(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:  # trw-fail-silent-allow: feeds temporal_eligibility's deliberate tri-state, where None is documented as "missing evidence" and is distinct from False ("contradicted") -- the opposite of collapsing the two
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class RecallInvocation:
    source: SourcePolicy
    temporal: TemporalSelection
    namespace: str
    tags: frozenset[str] = frozenset()
    confidence_floor: float | None = None
    exclude_historical_only: bool = False

    def allows_entry(self, entry: MemoryEntry) -> bool:
        if entry.namespace != self.namespace:
            raise NamespaceScopeError("recall candidate is outside its authorized namespace")
        if entry.status != MemoryStatus.ACTIVE:
            return False
        if self.temporal.exclude_system_canaries and entry.metadata.get("system_canary") == "true":
            return False
        if self.tags and not self.tags.issubset(entry.tags):
            return False
        fields = entry_policy_fields(entry)
        return self.source.allows(fields) and bool(
            apply_admission_filter(
                [fields],
                confidence_floor=self.confidence_floor,
                exclude_historical_only=self.exclude_historical_only,
                namespace=self.namespace,
            )
        )

    def acquire(
        self, fetch: Callable[[Callable[[MemoryEntry], bool], int], list[MemoryEntry]], *, limit: int
    ) -> list[MemoryEntry]:
        """Fill the existing cap in policy-priority order, not arrival order.

        Partitions are disjoint. Each underlying storage query retains its own
        ordering within a partition; this does not promise globally optimal
        relevance. At most four queries may scan the underlying collection.
        """
        if limit < 1:
            raise ValueError("acquisition limit must be positive")
        entries: list[MemoryEntry] = []
        for eligible in (True, False) if self.temporal.include_superseded else (True,):
            for bucket in (0, 2):

                def accepts(entry: MemoryEntry, eligible: bool = eligible, bucket: int = bucket) -> bool:
                    return (
                        self.allows_entry(entry)
                        and self.temporal.eligible(entry) is eligible
                        and self.source.rank_key(entry_policy_fields(entry))[0] == bucket
                    )

                entries.extend(fetch(accepts, limit - len(entries)))
                if len(entries) >= limit:
                    return entries
        return entries

    def rank_key(self, entry: MemoryEntry, raw_score: float, *, source: str = "local") -> tuple[int, int, float]:
        bucket, negative_score = self.source.rank_key(entry_policy_fields(entry, source=source, score=raw_score))
        return (int(not self.temporal.eligible(entry)), bucket, negative_score)
