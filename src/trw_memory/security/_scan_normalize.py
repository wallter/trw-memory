"""Invisible-character split redaction for egress (PII-INVISIBLE-SPLIT).

Belongs to ``trw_memory.security``; used by ``credentials.mask_credentials`` and ``pii.strip_pii``.

Redaction matches patterns, so an invisible format character inside a secret (a zero-width space, soft
hyphen, bidi override, word joiner or BOM) breaks the match and the secret leaves the box intact, or only
its head is masked. When the text contains such a character (Unicode category Cf), it is masked twice:
as written (A, exactly what the normal pass emits) and with every format character removed (B). Both
results are mapped back to character ranges of the ORIGINAL text, and the output masks the UNION of the
two. So nothing the normal pass masks can ever be exposed (A is a subset), a split secret is masked with
its full surrounding context (B sees the whole text, quotes and all), no compatibility normalisation runs,
and unmasked text is emitted exactly as written. Text without a format character takes the normal pass
unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

# Every placeholder the egress maskers emit. A placeholder never matches the text it replaced, which is
# what lets the unchanged literal stretches of a masked string be located in the original.
_PLACEHOLDER_RE = re.compile(r"<REDACTED:[a-z_]+>|<(?:email|api_key|phone|ssn|credit_card|ip|project)>|\$HOME")


# Invisible characters that are NOT category Cf but render as nothing: combining grapheme joiner, Hangul
# fillers, Khmer inherent vowels, Mongolian free variation selectors, variation selectors.
_EXTRA_INVISIBLE = frozenset(
    "\u034f\u115f\u1160\u17b4\u17b5\u180b\u180c\u180d\u180f\u3164\uffa0"
    + "".join(chr(c) for c in range(0xFE00, 0xFE10))
)


def _is_format(ch: str) -> bool:
    """An invisible character that can split a secret: Unicode category Cf, or one of ``_EXTRA_INVISIBLE``."""
    return ch in _EXTRA_INVISIBLE or unicodedata.category(ch) == "Cf"


def _masked_ranges(source: str, masked: str) -> list[tuple[int, int, str]]:
    """The ``(start, end, placeholder)`` ranges of *source* that *masked* replaced.

    *masked* is *source* with some non-empty stretches replaced by placeholders, so the literal stretches
    between placeholders appear in *source* in order. Where a literal could sit in more than one place,
    the range covers both the earliest and the latest placement, so it may over-mask but never under-mask.
    """
    parts = _PLACEHOLDER_RE.split(masked)
    holders = _PLACEHOLDER_RE.findall(masked)
    if not holders:
        return []
    last = len(parts) - 1
    full = [(0, len(source), holders[0])]  # not a pure substitution of source: mask everything rather than guess
    if not source.startswith(parts[0]) or not source.endswith(parts[last]):
        return full
    # The first literal is a prefix and the last a suffix (anchored: an empty edge literal is not "found" mid-text).
    early, pos = [0], len(parts[0])
    for i in range(1, last + 1):
        at = len(source) - len(parts[i]) if i == last else source.find(parts[i], pos)
        if at < pos:
            return full
        early.append(at)
        pos = at + len(parts[i])
    late, end = [0] * len(parts), len(source)
    for i in range(last, -1, -1):
        at = 0 if i == 0 else (len(source) - len(parts[i]) if i == last else source.rfind(parts[i], 0, end))
        if at < 0 or at + len(parts[i]) > end:
            return full
        late[i], end = at, at
    earliest = [(early[i] + len(parts[i]), early[i + 1], h) for i, h in enumerate(holders)]
    latest = [(late[i] + len(parts[i]), late[i + 1], h) for i, h in enumerate(holders)]
    # Both extreme alignments must reproduce *masked* exactly from *source*: the first literal anchored as a
    # prefix, the last as a suffix, every stretch lined up (auditor B1: a decoy literal later in the text must
    # never decide the alignment). Otherwise *masked* is not a pure substitution of *source*: mask everything.
    # ASSUMED contract of every egress masker (tested, not proven here): it only SUBSTITUTES placeholders for
    # stretches of its input (possibly empty, e.g. an empty JSON secret) and changes nothing else -- see
    # test_every_as_written_masker_only_substitutes. Under that contract both extreme alignments rebuild *masked*
    # exactly. This check is the fail-safe for a masker that breaks it (it fired once: the decisions redactor's
    # Bearer rule collapsed whitespace): mask everything rather than guess.
    for candidate in (earliest, latest):
        if any(end < start for start, end, _ in candidate) or _rebuild(source, candidate) != masked:
            return [(0, len(source), holders[0])]
    # Every valid alignment lies between the two, so the per-placeholder hull may over-mask, never under-mask.
    # A range that hides nothing (a placeholder inserted into an empty stretch) is dropped: it masks no text.
    # Cosmetic consequence: in text that contains an invisible character, an empty value's placeholder (e.g.
    # ``"password": "<REDACTED:json_secret>"`` for an empty password) is not emitted; nothing is exposed.
    hull = [(earliest[i][0], latest[i][1], h) for i, h in enumerate(holders)]
    return [r for r in hull if r[1] > r[0]]


def _rebuild(source: str, ranges: list[tuple[int, int, str]]) -> str:
    out, pos = [], 0
    for start, end, holder in ranges:
        if start < pos:
            return ""
        out.append(source[pos:start])
        out.append(holder)
        pos = end
    out.append(source[pos:])
    return "".join(out)


def _merge(ranges: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    merged: list[tuple[int, int, str]] = []
    for start, end, holder in sorted(ranges):
        if merged and start <= merged[-1][1]:
            prev_start, prev_end, prev_holder = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_holder)
        else:
            merged.append((start, end, holder))
    return merged


def _without_invisible_splits(text: str) -> tuple[str, list[int]]:
    """Return the invisible-stripped view and each kept character's source index."""
    kept = [index for index, ch in enumerate(text) if not _is_format(ch)]
    return "".join(text[index] for index in kept), kept


def _union_ranges(text: str, mask: Callable[[str], str]) -> list[tuple[int, int, str]]:
    """The merged ranges of *text* to mask: *mask*'s own ranges plus those it finds once invisible characters
    are removed. The output is exactly ``_rebuild(text, _union_ranges(...))``."""
    joined, kept = _without_invisible_splits(text)
    ranges = _masked_ranges(text, mask(text))
    for start, end, holder in _masked_ranges(joined, mask(joined)):
        ranges.append((kept[start], kept[end - 1] + 1, holder))
    return _merge(ranges)


def spans_with_invisible_splits(
    text: str,
    find_spans: Callable[[str], list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """Return detector spans in *text* plus spans found after removing invisible split characters.

    Matches in the normalized view map back to the original character offsets. This is the detection
    counterpart of :func:`mask_with_invisible_splits`: normalization only discovers candidates; returned
    spans always refer to the untouched source text.
    """
    spans = list(find_spans(text))
    if not text or not any(_is_format(ch) for ch in text):
        return spans

    joined, kept = _without_invisible_splits(text)
    for start, end in find_spans(joined):
        if 0 <= start < end <= len(kept):
            spans.append((kept[start], kept[end - 1] + 1))

    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def mask_with_invisible_splits(text: str, mask: Callable[[str], str]) -> str:
    """*mask* applied to *text*, plus every range that *mask* finds once invisible characters are removed."""
    if not text or not any(_is_format(ch) for ch in text):
        return mask(text)
    return _rebuild(text, _union_ranges(text, mask))
