"""``strip_pii`` and ``mask_query_credentials`` mask emails in linear time, with the old output.

Each carried its own copy of the email pattern and called ``re.sub``: 5.9 s and 5.7 s on one 64 KiB
``@``-free field (M6 fixed only ``detect_pii``'s scan). Both now call ``_mask_emails``, built on the one
pattern and the linear scan ``detect_pii`` uses.
"""

from __future__ import annotations

import random
import time

import pytest

from tests._timing import assert_budget
from trw_memory.models.memory import MAX_TEXT_FIELD_CHARS
from trw_memory.security import pii
from trw_memory.security.pii import _EMAIL_PATTERN, _mask_emails, mask_query_credentials, strip_pii

pytestmark = pytest.mark.unit


def _old_mask(text: str) -> str:
    """The behaviour both functions had before: ``re.sub`` of the shared pattern."""
    return _EMAIL_PATTERN.sub("<email>", text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "mail alice@example.com today",
        "a@b.com.x@c.org",  # re.sub resumes INSIDE a local-part run
        "a@b.com9x@c.org",
        "a@b.com-x@c.org",
        "x" * 500 + "@example.com",
        "x" * 500 + " no at sign",
        "first.last+tag@sub.example.co.uk, second@example.io",
        "@@a@b.cc@@d@e.ff",
        "trailing a@b.co",
    ],
)
def test_mask_emails_equals_the_old_re_sub_on_edge_cases(text: str) -> None:
    assert _mask_emails(text) == _old_mask(text)


def test_mask_emails_equals_the_old_re_sub_on_random_text() -> None:
    alphabet = "ab.-_%+@ 9Zc\n/"
    rng = random.Random(1)
    for _ in range(5_000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert _mask_emails(text) == _old_mask(text), repr(text)


def test_both_egress_functions_equal_their_old_output_on_random_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whole-function equivalence: swap the old ``re.sub`` in, and every fuzz string gives the same result."""
    alphabet = "ab.-_%+@ 9Zc\nsk-_key"
    rng = random.Random(2)
    texts = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60))) for _ in range(2_000)]
    new = [(strip_pii(t), mask_query_credentials(t)) for t in texts]

    monkeypatch.setattr(pii, "_mask_emails", _old_mask)

    assert new == [(strip_pii(t), mask_query_credentials(t)) for t in texts]


def test_an_email_after_a_long_run_is_still_masked_by_both() -> None:
    text = "y" * (MAX_TEXT_FIELD_CHARS - 40) + " contact bob@example.com"
    assert strip_pii(text).endswith(" contact <email>")
    assert mask_query_credentials(text).endswith(" contact <email>")


@pytest.mark.requires_local_timing
@pytest.mark.parametrize("egress", [strip_pii, mask_query_credentials], ids=["strip_pii", "mask_query_credentials"])
def test_egress_email_masking_on_a_64k_field_is_fast_budget(egress) -> None:  # type: ignore[no-untyped-def]
    text = "y" * MAX_TEXT_FIELD_CHARS
    started = time.perf_counter()
    egress(text)
    elapsed = time.perf_counter() - started
    # 0.013 s and 0.002 s after the fix at load ~15; 5.9 s and 5.7 s before it.
    assert_budget(f"{egress.__name__}_64k_field", elapsed, 0.1, "s")
