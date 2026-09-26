"""The length cap on an assertion's caller-written text (rc6 C12).

A leaf module: the write paths (store, update, import) and verification all read it, and
none of them may import the others.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trw_memory.models.memory import Assertion

#: Longest pattern or target an assertion is written with. Verification refuses a longer one before
#: any regex or walk touches it; rows stored before the cap still load, and verify unverified.
MAX_PATTERN_LEN: int = 1024
OVERLONG = f"an assertion's pattern or target is longer than {MAX_PATTERN_LEN} characters"


def overlong(assertion: Assertion | Mapping[str, object]) -> bool:
    """Whether *assertion* (a model, or the raw dict a bulk or sync caller sends) is over the cap."""
    if isinstance(assertion, Mapping):
        return any(len(str(assertion.get(key) or "")) > MAX_PATTERN_LEN for key in ("pattern", "target"))
    return max(len(assertion.pattern), len(assertion.target)) > MAX_PATTERN_LEN
