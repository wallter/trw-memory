"""Temporal query classifier and prefix stripper for auto-recency boost.

Detects temporal language in queries so callers can auto-populate
``recency_weight`` and ``as_of`` without user configuration.

The classifier is intentionally conservative (high precision, not recall):
false positives incorrectly degrade exact-match precision for non-temporal
queries, which is worse than missing a temporal query.

Patterns captured:
- Relative recency adverbs: recently, lately, just, now
- Relative time phrases: last week/month/year, past N days/weeks
- Superlative recency: newest, most recent, latest, current
- Present-state anchors: today, this week/month/year
- Temporal verbs: was updated, has changed, is now
- Temporal arithmetic: N days/weeks/months ago, last Tuesday
- Prior context: previous conversation/chat references

Temporal queries often start with boilerplate prefixes like
"latest guidance on X" or "what is the current state of X".  These
prefixes are high-df noise for BM25 (common across many queries) and
add semantic drift for dense embedders.  ``strip_temporal_prefix``
removes them, leaving just the meaningful topic tokens (X).

"Previous conversation" boilerplate ("I was looking back at our previous chat
about X, can you remind me Y?") is similarly stripped — the meaningful topic
is X, not the conversational framing.

Returns a dataclass with the detected ``is_temporal`` flag plus a suggested
``recency_weight`` scaled by detection confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TemporalClassification:
    is_temporal: bool
    confidence: float
    recency_weight: float
    matched_patterns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TemporalQueryRewrite:
    retrieval_query: str
    recency_weight: float
    classification: TemporalClassification | None = None
    prefix_stripped: bool = False


# Pattern groups ordered from most specific (high confidence) to broadest.
# Each tuple is (regex, confidence_contribution, label).
_PATTERNS: list[tuple[re.Pattern[str], float, str]] = [
    # Superlative recency — very strong signal
    (re.compile(r"\b(latest|newest|most recent|current|up[- ]to[- ]date)\b", re.IGNORECASE), 0.9, "superlative"),
    # Explicit relative window — strong
    (re.compile(r"\blast\s+\d+\s+(day|week|month|year)s?\b", re.IGNORECASE), 0.85, "relative_window_n"),
    # Named recent windows — strong
    (re.compile(r"\b(last|past|previous)\s+(week|month|year|quarter|sprint|cycle|session)\b", re.IGNORECASE), 0.8, "relative_window"),
    # Today / this-period anchors — moderate
    (re.compile(r"\b(today|this (week|month|year|quarter|sprint|session))\b", re.IGNORECASE), 0.7, "present_anchor"),
    # Recency adverbs — moderate
    (re.compile(r"\b(recently|lately|just now|just\b|right now)\b", re.IGNORECASE), 0.65, "recency_adverb"),
    # Temporal state verbs — moderate
    (re.compile(r"\b(was updated|has changed|is now|have changed|were updated)\b", re.IGNORECASE), 0.65, "state_verb"),
    # Future anchors (lower weight — user wants upcoming, not past recency)
    (re.compile(r"\b(upcoming|next (week|month|quarter|sprint))\b", re.IGNORECASE), 0.4, "future_anchor"),
    # Temporal arithmetic — pinpoints a specific past moment (is_temporal=True;
    # recency_weight stays low: the caller wants the entry FROM that moment, not
    # the most-recent entries). Future code can branch on "temporal_arithmetic"
    # to apply date-range filtering instead of recency blending.
    (
        re.compile(
            r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|twenty|thirty|forty|fifty|sixty|ninety|hundred)\s+"
            r"(day|days|week|weeks|month|months|year|years)\s+ago\b",
            re.IGNORECASE,
        ),
        0.75,
        "temporal_arithmetic",
    ),
    # Named weekday anchors ("last Tuesday", "last Monday") — specific day
    (
        re.compile(
            r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            re.IGNORECASE,
        ),
        0.75,
        "temporal_arithmetic",
    ),
    # Prior-context references — "previous conversation/chat/session" without a
    # named window (no recency boost needed; the memory IS the prior context).
    (
        re.compile(
            r"\b(previous|prior|earlier|last)\s+(conversation|chat|session|discussion|talk)\b",
            re.IGNORECASE,
        ),
        0.6,
        "prior_context",
    ),
    # Vague recency adverbs — "a while ago", "the other day"
    # (lower confidence: broad temporal reference with no time bound)
    (
        re.compile(r"\ba while (ago|back)\b|\bthe other day\b", re.IGNORECASE),
        0.6,
        "recency_adverb",
    ),
    # "some time ago" / "sometime ago" — vague past reference, similar to
    # temporal arithmetic but without a specific count.
    (
        re.compile(r"\bsome\s*time\s+ago\b", re.IGNORECASE),
        0.65,
        "temporal_arithmetic",
    ),
    # "last time" standalone — "the last time I X" / "last time we discussed"
    # (slightly weaker than named windows because "last time" can be non-temporal
    #  in constructions like "last time I'll say this").
    (
        re.compile(r"\blast time\b", re.IGNORECASE),
        0.6,
        "relative_window",
    ),
    # Memory-recall opener — "do you remember when I X" / "remember that time"
    (
        re.compile(r"\b(do you |you )?(remember|recall)\s+(when|that time|me (saying|telling|mentioning))\b", re.IGNORECASE),
        0.6,
        "prior_context",
    ),
]

# Combined confidence ceiling — multiple weak signals can stack but are capped
_MAX_CONFIDENCE: float = 0.95
# Minimum confidence before we call the query temporal
_TEMPORAL_THRESHOLD: float = 0.5
# Maximum recency_weight we ever auto-suggest (keep user override in control)
_MAX_AUTO_RECENCY_WEIGHT: float = 0.6


# Boilerplate temporal prefixes that carry no topical signal.
# Ordered longest-first so the greedier match is tried first (prevents
# "latest" stripping before "latest guidance on" has a chance to match).
_TEMPORAL_PREFIXES: list[re.Pattern[str]] = [
    re.compile(
        r"^(what is |what('s| is) |tell me )?the (latest|current|most recent|newest|up-to-date) "
        r"(guidance|information|info|update|status|state|version|rules?|docs?|documentation|"
        r"advice|recommendation|summary|overview) (on|for|about|regarding) ",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(what('s| is) |what are )?the (latest|current|most recent|newest) ",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(latest|current|most recent|newest|up-to-date) (guidance|information|info|update|"
        r"status|state|rules?|docs?|documentation|advice|recommendation) (on|for|about|regarding) ",
        re.IGNORECASE,
    ),
    re.compile(r"^(latest|most recent|current|newest) (on|for|about|regarding) ", re.IGNORECASE),
    re.compile(r"^(what('s| is| are) (the )?)?(latest|current|most recent) ", re.IGNORECASE),
    # Prior-context boilerplate: "I was looking back at our previous conversation about X"
    # → strip conversational frame, leave topic X.
    # "I was looking back at our previous conversation about X" → X
    re.compile(
        r"^(I('m| am| was| have been))?\s*(looking back at|going back to|thinking about|"
        r"checking)\s+(our\s+)?(previous|prior|earlier|last)\s+"
        r"(conversation|chat|session|discussion|talk)\s+(about|on|regarding|re:)\s+",
        re.IGNORECASE,
    ),
    # "I was looking back at our previous chat and I wanted to confirm, X" → X
    re.compile(
        r"^(I('m| am| was| have been))?\s*(looking back at|going back to|thinking about|"
        r"checking)\s+(our\s+)?(previous|prior|earlier|last)\s+"
        r"(conversation|chat|session|discussion|talk)\s+and\s+I\s+wanted\s+to\s+"
        r"(confirm|check|verify|ask|know|revisit)[.,]?\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(I\s+)?(wanted to follow up on|wanted to check back on|"
        r"was thinking about)\s+(our\s+)?(previous|prior|last)\s+"
        r"(conversation|chat|discussion)\s+(about|on|regarding)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(I\s+)?(remember|recall)\s+(you\s+)?(told|mentioned|said|suggested|recommended)\s+"
        r"(me\s+)?(about\s+)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(I\s+)?(think|thought)\s+we\s+discussed\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^in (our\s+)?(previous|prior|earlier|last)\s+(conversation|chat|session|discussion),?\s*"
        r"(you\s+)?(mentioned|said|suggested|told me|recommended)?\s*",
        re.IGNORECASE,
    ),
    # "How many days/weeks/months ago did I X?" → "X"
    re.compile(
        r"^how many\s+(day|days|week|weeks|month|months|year|years)\s+ago\s+(did|have|has)\s+I\s+",
        re.IGNORECASE,
    ),
    # "Do you remember when I X?" → "X"
    re.compile(
        r"^do you remember when (I|we)\s+",
        re.IGNORECASE,
    ),
    # "The last time I/we X" → "X"
    re.compile(
        r"^(the\s+)?last time (I|we)\s+",
        re.IGNORECASE,
    ),
    # "A while ago/back (I|you) X" → "X"
    re.compile(
        r"^a while (ago|back)[,]?\s*(I|you|we)?\s*",
        re.IGNORECASE,
    ),
]


def strip_temporal_prefix(query: str) -> str:
    """Remove leading boilerplate temporal phrases from *query*.

    Strips prefixes like "latest guidance on X" → "X" so BM25 and dense
    search focus on the meaningful topic tokens rather than high-df noise.
    Returns the original query unchanged when no prefix matches.

    Args:
        query: Raw recall query string.

    Returns:
        Query with the temporal boilerplate prefix removed, or the
        original query if no prefix pattern matches.  Leading/trailing
        whitespace is stripped from the result.
    """
    for pattern in _TEMPORAL_PREFIXES:
        m = pattern.match(query)
        if m:
            remainder = query[m.end():].strip()
            if remainder:
                return remainder
    return query


# Inline temporal arithmetic phrases: these appear anywhere in a query and
# add no lexical value (sessions store absolute dates, not relative ones).
# Ordered longest-first to avoid partial matches.
_TEMPORAL_ARITHMETIC_INLINE: list[re.Pattern[str]] = [
    # "on the Wednesday two months ago" — named-weekday + time-ago compound
    re.compile(
        r"\bon the\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
        r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|twenty|thirty|forty|fifty|sixty|ninety|hundred)\s+"
        r"(day|days|week|weeks|month|months|year|years)\s+ago\b",
        re.IGNORECASE,
    ),
    # "N days/weeks/months/years ago"
    re.compile(
        r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|twenty|thirty|forty|fifty|sixty|ninety|hundred)\s+"
        r"(day|days|week|weeks|month|months|year|years)\s+ago\b",
        re.IGNORECASE,
    ),
    # "last Tuesday / last Monday" etc.
    re.compile(
        r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    ),
    # "during the lunch last Tuesday" — preposition phrase before named weekday
    re.compile(
        r"\bduring\s+the\s+\w+\s+last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    ),
    # "a while ago" / "a while back" / "some time ago" — vague past references
    re.compile(r"\ba while (ago|back)\b|\bsome\s*time\s+ago\b", re.IGNORECASE),
    # "the other day" — informal recent reference
    re.compile(r"\bthe other day\b", re.IGNORECASE),
]


def strip_temporal_arithmetic(query: str) -> str:
    """Remove embedded temporal arithmetic phrases from *query*.

    Strips relative-time expressions like "10 days ago", "last Tuesday", or
    "on the Wednesday two months ago" from ANYWHERE in the query (not just
    the prefix), leaving the topical content for BM25 / dense retrieval.

    Sessions store absolute dates ("session_date: 2023/05/20") so relative
    phrases ("10 days ago") never produce lexical matches; removing them lets
    the retriever focus on what actually appears in session content.

    Returns the query with temporal arithmetic phrases removed, whitespace
    normalised, and leading/trailing whitespace stripped.  Returns the original
    query unchanged when no phrase matches.

    Args:
        query: Raw recall query string.

    Returns:
        Query with embedded temporal arithmetic phrases removed, or the
        original query if no phrase pattern matches.
    """
    result = query
    for pattern in _TEMPORAL_ARITHMETIC_INLINE:
        result = pattern.sub(" ", result)
    # Collapse runs of whitespace and strip
    result = " ".join(result.split())
    return result if result else query


def prepare_temporal_query(
    query: str,
    *,
    current_recency_weight: float,
    auto_temporal: bool,
    strip_prefix: bool,
) -> TemporalQueryRewrite:
    """Return the retrieval query and effective recency weight for *query*.

    This keeps SDK recall and MCP-tool recall on the same temporal-query
    contract: auto-recency only fills the zero/default weight, prefix stripping
    is opt-in by config, and non-temporal queries pass through unchanged.
    """
    if not auto_temporal:
        return TemporalQueryRewrite(retrieval_query=query, recency_weight=current_recency_weight)

    temporal = classify_temporal(query)
    if not temporal.is_temporal:
        return TemporalQueryRewrite(
            retrieval_query=query,
            recency_weight=current_recency_weight,
            classification=temporal,
        )

    effective_recency_weight = current_recency_weight
    if effective_recency_weight == 0.0:
        effective_recency_weight = temporal.recency_weight

    retrieval_query = query
    if strip_prefix:
        retrieval_query = strip_temporal_prefix(query)
        # Also strip embedded temporal arithmetic so BM25 / dense focus on topic
        # tokens rather than relative-time phrases ("10 days ago", "last Tuesday").
        retrieval_query = strip_temporal_arithmetic(retrieval_query)

    return TemporalQueryRewrite(
        retrieval_query=retrieval_query,
        recency_weight=effective_recency_weight,
        classification=temporal,
        prefix_stripped=retrieval_query != query,
    )


def classify_temporal(query: str) -> TemporalClassification:
    """Classify whether *query* contains temporal language.

    Returns a :class:`TemporalClassification` with ``is_temporal``,
    ``confidence``, ``recency_weight`` (0.0–0.6), and which patterns fired.

    Confidence is the capped sum of all matching pattern contributions.
    ``recency_weight`` scales linearly from 0 at the threshold to
    ``_MAX_AUTO_RECENCY_WEIGHT`` at full confidence.

    Args:
        query: Free-text recall query to classify.

    Returns:
        :class:`TemporalClassification` — always succeeds, never raises.
    """
    matched: list[str] = []
    total_confidence: float = 0.0

    for pattern, contrib, label in _PATTERNS:
        if pattern.search(query):
            matched.append(label)
            total_confidence += contrib

    confidence = min(total_confidence, _MAX_CONFIDENCE)
    is_temporal = confidence >= _TEMPORAL_THRESHOLD

    if not is_temporal:
        return TemporalClassification(
            is_temporal=False,
            confidence=confidence,
            recency_weight=0.0,
            matched_patterns=matched,
        )

    # Scale recency_weight linearly from threshold to max
    scale = (confidence - _TEMPORAL_THRESHOLD) / (_MAX_CONFIDENCE - _TEMPORAL_THRESHOLD)
    recency_weight = round(_MAX_AUTO_RECENCY_WEIGHT * scale, 3)
    recency_weight = min(recency_weight, _MAX_AUTO_RECENCY_WEIGHT)

    return TemporalClassification(
        is_temporal=True,
        confidence=round(confidence, 3),
        recency_weight=recency_weight,
        matched_patterns=matched,
    )
