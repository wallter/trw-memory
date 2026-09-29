"""The EMAIL detector scans in linear time and finds exactly what ``finditer`` found.

A plain ``finditer`` of the email pattern retried every position inside a long
``@``-free local-part run and rescanned the rest of the run each time, so one
64 KiB field cost 0.85-4.3 s in ``detect_pii`` (M6, 2026-09-28).
"""

from __future__ import annotations

import random
import time

import pytest

from tests._timing import assert_budget
from trw_memory.models.memory import MAX_TEXT_FIELD_CHARS
from trw_memory.security.pii import _EMAIL_PATTERN, PIIType, _email_matches, detect_pii

pytestmark = pytest.mark.unit


def _spans(matches: object) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in matches]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "mail alice@example.com today",
        "a@b.com.x@c.org",  # finditer resumes INSIDE a local-part run
        "a@b.com9x@c.org",
        "a@b.com-x@c.org",
        "x" * 500 + "@example.com",
        "x" * 500 + " no at sign",
        "first.last+tag@sub.example.co.uk, second@example.io",
        "@@a@b.cc@@d@e.ff",
    ],
)
def test_email_scan_matches_finditer_on_edge_cases(text: str) -> None:
    assert _spans(_email_matches(text)) == _spans(_EMAIL_PATTERN.finditer(text))


def test_email_scan_matches_finditer_on_random_text() -> None:
    alphabet = "ab.-_%+@ 9Zc\n/"
    rng = random.Random(0)
    for _ in range(5_000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert _spans(_email_matches(text)) == _spans(_EMAIL_PATTERN.finditer(text)), repr(text)


def test_detect_pii_still_reports_an_email_after_a_long_run() -> None:
    text = "y" * (MAX_TEXT_FIELD_CHARS - 40) + " contact bob@example.com"
    emails = [m.value for m in detect_pii(text) if m.pii_type == PIIType.EMAIL]
    assert emails == ["bob@example.com"]


@pytest.mark.requires_local_timing
def test_detect_pii_on_a_64k_field_is_fast_budget() -> None:
    text = "y" * MAX_TEXT_FIELD_CHARS
    started = time.perf_counter()
    detect_pii(text)
    elapsed = time.perf_counter() - started
    # 0.011 s after the fix at load ~19; 0.85-4.3 s before it.
    assert_budget("detect_pii_64k_field", elapsed, 0.1, "s")
