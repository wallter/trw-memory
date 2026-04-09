"""Token-aware recall budgeting for trw-memory.

Provides utilities to estimate the token cost of memory entries and apply
a token budget to a list of recall results, truncating to fit within the
budget while guaranteeing at least one result is returned.

Constants:
    TOKEN_MULTIPLIER: Ratio of tokens per whitespace-delimited word (1.3).
    METADATA_OVERHEAD: Fixed token cost for entry metadata fields (20).
"""

from __future__ import annotations

__all__ = [
    "METADATA_OVERHEAD",
    "TOKEN_MULTIPLIER",
    "apply_token_budget",
    "estimate_entry_tokens",
    "estimate_tokens",
]

# Average tokens per whitespace-delimited word.  English text typically
# tokenises to ~1.3 tokens/word across GPT-family tokenisers.
TOKEN_MULTIPLIER: float = 1.3

# Fixed overhead per entry for metadata fields (id, timestamps, namespace,
# importance, score, etc.) that are always included in the response.
METADATA_OVERHEAD: int = 20


def estimate_tokens(text: str | None) -> int:
    """Estimate the token count of *text* using a word-count heuristic.

    Args:
        text: The input text.  ``None`` and whitespace-only strings are
            treated as empty (returning 0).

    Returns:
        Estimated token count.  Always >= 1 for non-empty text, 0 otherwise.
    """
    if text is None:
        return 0
    words = text.split()
    if not words:
        return 0
    return max(1, round(len(words) * TOKEN_MULTIPLIER))


def estimate_entry_tokens(entry: dict[str, object]) -> int:
    """Estimate the total token cost of a memory entry dict.

    Combines the token cost of ``content``, ``detail``, and ``tags`` fields
    with a fixed :data:`METADATA_OVERHEAD`.

    Args:
        entry: A dict with optional ``content`` (str), ``detail`` (str),
            and ``tags`` (list[str]) keys.

    Returns:
        Estimated token count including metadata overhead.
    """
    content = str(entry.get("content", "") or "")
    detail = str(entry.get("detail", "") or "")
    raw_tags = entry.get("tags")
    if isinstance(raw_tags, list):
        tags_text = " ".join(str(t) for t in raw_tags)
    else:
        tags_text = ""

    combined = f"{content} {detail} {tags_text}".strip()
    return estimate_tokens(combined) + METADATA_OVERHEAD


def apply_token_budget(
    results: list[dict[str, object]],
    token_budget: int,
) -> tuple[list[dict[str, object]], int, bool]:
    """Filter *results* to fit within *token_budget*.

    Iterates results in order, accumulating the token cost of each entry.
    An entry is included if the cumulative total does not exceed the budget.
    If no entries fit within the budget, the first entry is always included
    (minimum-one guarantee).

    Args:
        results: Ordered list of memory result dicts.
        token_budget: Maximum token budget.  Must be a positive integer.

    Returns:
        A 3-tuple of ``(filtered_results, tokens_used, was_truncated)``.

    Raises:
        ValueError: If *token_budget* is not a positive integer.
    """
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")

    if not results:
        return [], 0, False

    filtered: list[dict[str, object]] = []
    tokens_used = 0
    truncated = False

    for entry in results:
        cost = estimate_entry_tokens(entry)
        if tokens_used + cost > token_budget and filtered:
            truncated = True
            break
        filtered.append(entry)
        tokens_used += cost

    return filtered, tokens_used, truncated
