"""The label levels and the places a row can go (PRD-SEC-023)."""

from __future__ import annotations

import unicodedata
from bisect import bisect_right
from enum import IntEnum, StrEnum
from functools import lru_cache

from trw_memory.labels._default_ignorable import DEFAULT_IGNORABLE_RANGES


class Level(IntEnum):
    """A confidentiality level; a higher number is more sensitive. ``PUBLIC`` is reserved: no phase-0 rule or sink uses it."""

    PUBLIC = 0
    TEAM = 1
    PERSONAL = 2
    SENSITIVE = 3


class Surface(StrEnum):
    """Where a recalled row is shown: ``auto`` is everything the agent did not ask for; ``agent`` is ``trw_recall``."""

    AUTO = "auto"
    AGENT = "agent"


class Sink(StrEnum):
    """Where a row can leave the machine or land in a tracked file; both clear ``team`` only in phase 0."""

    PLATFORM = "platform"
    PROJECT_FILES = "project_files"


#: The only stamp values; any other value in ``metadata['trw_label']`` reads as ``SENSITIVE`` (fail closed).
STAMP_VALUES: dict[str, Level] = {level.name.lower(): level for level in Level}


@lru_cache(maxsize=4096)  # on the admit hot path; a store's tags repeat
def tag_key(tag: str) -> str:
    """The form every label match compares a tag in, on the rule side and the row side alike (LABEL-TAG-STRIP).

    NFKC (fullwidth and compatibility forms fold to their plain letters), invisible characters dropped (format
    characters and the other default-ignorables, so a zero-width space cannot split a tag away from its rule),
    casefolded, whitespace trimmed and collapsed. A tag spelled ``"finance "`` or in fullwidth letters is the
    ``finance`` a rule names, so it cannot fall back to ``team``.
    """
    if tag.isascii():  # NFKC leaves ASCII alone, ASCII holds no invisible character, and casefold is lower
        return " ".join(tag.lower().split())
    text = "".join(ch for ch in unicodedata.normalize("NFKC", tag) if not _ignorable(ch))
    return " ".join(text.casefold().split())


#: The start of each ``DEFAULT_IGNORABLE_RANGES`` range, for a bisect lookup.
_IGNORABLE_STARTS = tuple(low for low, _high in DEFAULT_IGNORABLE_RANGES)


def _ignorable(ch: str) -> bool:
    """A character that is there but not seen: Unicode Default_Ignorable_Code_Point, or any format character (Cf).

    The property comes from the vendored, generated ``_default_ignorable`` table, not a hand-kept list: U+2065 is
    default-ignorable but category Cn, and a Cf check plus a hand list missed it (C1 red team, LABEL-TAG-STRIP-a).
    """
    if unicodedata.category(ch) == "Cf":
        return True
    point = ord(ch)
    index = bisect_right(_IGNORABLE_STARTS, point) - 1
    return index >= 0 and point <= DEFAULT_IGNORABLE_RANGES[index][1]
