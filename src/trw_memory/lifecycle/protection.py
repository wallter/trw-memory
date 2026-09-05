"""``protection_tier`` semantics for every automatic-removal path (PRD-CORE-244 FR10).

``protection_tier`` was advertised by ``trw_learn`` for its whole life with two
consumers, both dedup merge tiebreaks. Every *destructive* path — utility prune,
excess-entry auto-prune, tier demotion, cold purge — ignored it, so an entry an
agent marked ``permanent`` was nominated for removal on exactly the same schedule
as a ``normal`` one.

This module is the single place that turns the tier vocabulary into a decision.
It is deliberately pure: no config import, no I/O, no entry model import, so both
packages can call it from inside a hot scan loop and trw-mcp can pass its own
``TRWConfig.protection_tier_prune_discount`` table without a MemoryConfig round
trip.

Two shapes of answer:

* :func:`is_removal_exempt` — ``protected`` and ``permanent`` are never
  auto-removed. Not "harder to remove": exempt.
* :func:`prune_threshold_multiplier` — every other tier multiplies the utility
  threshold a candidate must fall *below*, so ``critical`` at 0.25 must be four
  times less useful than ``normal`` before it is nominated.

Manual deletion (``trw_forget``, an operator naming an entry) is explicitly out
of scope: this governs automatic removal only.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "EXEMPT_PROTECTION_TIERS",
    "entry_protection_tier",
    "is_removal_exempt",
    "prune_threshold_multiplier",
]

#: Tiers whose promise is absolute. An entry carrying one of these is never
#: nominated by an automatic prune, demotion or purge, regardless of utility.
EXEMPT_PROTECTION_TIERS = frozenset({"protected", "permanent"})

#: Applied when a tier is absent from the configured discount table (including
#: the empty-string / missing-field case). Neutral by construction: an unknown
#: tier must not silently become easier OR harder to remove.
NEUTRAL_MULTIPLIER = 1.0


def entry_protection_tier(entry: Mapping[str, object]) -> str:
    """Read an entry dict's ``protection_tier`` as a lowercase string.

    Accepts the serialized string form and the ``ProtectionTier`` enum form
    (``use_enum_values`` is not set on ``MemoryEntry``, so a live model dump can
    still carry the enum). Anything unreadable degrades to ``"normal"`` — the
    same default the model applies — rather than to an exemption, so a malformed
    field can never manufacture protection.
    """
    raw = entry.get("protection_tier", "")
    value = getattr(raw, "value", raw)
    if not isinstance(value, str) or not value:
        return "normal"
    return value.strip().lower()


def is_removal_exempt(entry: Mapping[str, object]) -> bool:
    """True when *entry*'s tier forbids automatic removal outright."""
    return entry_protection_tier(entry) in EXEMPT_PROTECTION_TIERS


def prune_threshold_multiplier(
    entry: Mapping[str, object],
    discounts: Mapping[str, float],
) -> float | None:
    """Return the multiplier for *entry*'s removal threshold, or ``None`` if exempt.

    ``None`` is the exemption signal and callers MUST branch on it — returning
    0.0 instead would look like "threshold zero", which a ``<`` comparison
    against a negative utility could still satisfy.

    Args:
        entry: Serialized entry dict (YAML, SQLite row, or ``model_dump()``).
        discounts: The configured tier -> multiplier table.

    Returns:
        The multiplier to apply to the nomination threshold, or ``None`` when
        the entry must never be nominated.
    """
    tier = entry_protection_tier(entry)
    if tier in EXEMPT_PROTECTION_TIERS:
        return None
    return float(discounts.get(tier, NEUTRAL_MULTIPLIER))
