"""Request-scoped temporal eligibility and bounded candidate selection.

Generic storage callers opt in explicitly; absent selection must retain raw
maintenance visibility. Acquisition callers share one instance across stores.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from heapq import nlargest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.validity_prior import ValidityFields, _is_open_at


@dataclass(frozen=True, slots=True)
class TemporalSelection:
    """One request's canonical validity policy, independent of model readiness.

    ``as_of=None`` retains the existing open-window policy, not an implicit
    historical query at ``reference_time``. The captured reference is used for
    expiry only. Explicit as-of instants retain canonical predicate semantics.
    """

    as_of: datetime | None = None
    include_superseded: bool = False
    exclude_system_canaries: bool = False
    reference_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def eligible(self, entry: ValidityFields) -> bool:
        """Evaluate the shared policy without consulting a changing wall clock."""
        return _is_open_at(entry, self.as_of, reference_time=self.reference_time)

    def select(self, entries: Iterable[MemoryEntry], *, limit: int) -> list[MemoryEntry]:
        """Select eligible-first, preserving order within each class.

        Working lists retain at most ``2 * limit`` references, plus the returned
        list. Stops at ``limit`` eligible entries. An all-ineligible iterable must be exhausted to establish that
        no eligible entries remain; this is a memory bound, not a work deadline.
        The caller owns and closes any cursor backing the iterable.
        """
        if limit <= 0:
            raise ValueError("Temporal selection limit must be positive")
        eligible: list[MemoryEntry] = []
        deferred: list[MemoryEntry] = []
        for entry in entries:
            if self.exclude_system_canaries and entry.metadata.get("system_canary") == "true":
                continue
            if self.eligible(entry):
                eligible.append(entry)
                if len(eligible) == limit:
                    break
            elif self.include_superseded and len(deferred) < limit:
                deferred.append(entry)
        return eligible + deferred[: limit - len(eligible)]

    def select_ranked(
        self,
        entries: Iterable[MemoryEntry],
        *,
        limit: int,
        rank_key: Callable[[MemoryEntry], tuple[float, str, str]],
    ) -> list[MemoryEntry]:
        """Rank an unordered stream with O(limit) retained candidates.

        All input entries are visited. Eligible records rank ahead of deferred
        ones; ordering within a class comes from the caller's existing key.
        """
        if limit <= 0:
            raise ValueError("Temporal selection limit must be positive")

        def candidates() -> Iterator[tuple[bool, tuple[float, str, str], MemoryEntry]]:
            for entry in entries:
                if self.exclude_system_canaries and entry.metadata.get("system_canary") == "true":
                    continue
                eligible = self.eligible(entry)
                if eligible or self.include_superseded:
                    yield eligible, rank_key(entry), entry

        selected = nlargest(limit, candidates(), key=lambda item: (item[0], item[1]))
        return [item[2] for item in selected]
