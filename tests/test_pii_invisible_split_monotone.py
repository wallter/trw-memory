"""PII-INVISIBLE-SPLIT monotonicity: the invisible-character handling may only ADD redaction.

``*_as_written`` are the maskers exactly as before the change (trunk behaviour); the public entry points add
the invisible-character-stripped pass on top. Checked character by character: every character of the input
that the as-written pass masks must also be masked by the public entry point, for fuzzed inputs that mix
credentials, PII, quoted and JSON values, PEM blocks, decoy keywords, varied separators and invisible
characters (auditor M2: a tail-only check could not see a decoy shifting the alignment).
"""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest

from trw_memory.decisions._redaction import _default_redactor_as_written, default_redactor
from trw_memory.security._scan_normalize import _PLACEHOLDER_RE, _is_format, _rebuild, _union_ranges
from trw_memory.security.credentials import mask_credentials, mask_credentials_as_written
from trw_memory.security.pii import (
    mask_query_credentials,
    mask_query_credentials_as_written,
    strip_pii,
    strip_pii_as_written,
)

INVISIBLE = ["\u200b", "\u00ad", "\u202e", "\u200d", "\u2060", "\ufeff", "\u034f", "\u3164", "\ufe0f"]
SEPARATORS = [" ", " ", "\t", "\n", "\u00a0", ""]
PIECES = [
    "AKIAIOSFODNN7EXAMPLE",
    "ghp_" + "a1B2c3D4e5F6g7H8i9J0" * 2,
    "sk_live_" + "Z9y8X7w6V5u4T3s2R1q0P9o8",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJl",
    "123-45-6789",
    "123 45 6789",
    "jane.doe@example.com",
    "+1 (415) 555-0123",
    'PASSWORD="first SensitiveSuffixABC"',
    '{"password": "hunter2hunter2", "user": "x"}',
    "API_TOKEN=abcdEFGH1234ijkl",
    "Authorization: Bearer abcdef123456789",
    "Bearer abcdef123456",
    "Bearer\tabcdef123456",
    "Token\n\tzyxwvu987654",
    'PASSWORD="Bearer x y"',
    '"Bearer x"',
    "-----BEGIN PRIVATE KEY-----\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEA\n-----END PRIVATE KEY-----",
    "postgres://admin:s3cretPass@db.internal",
    "?token=abc123def456ghi",
    # decoys and plain prose
    "Bearer",
    "PASSWORD=",
    '"',
    "token",
    "José",
    "東京",
    "x",
    "abcde",
    "x y",
]


def _case(rng: random.Random) -> str:
    text = ""
    for _ in range(rng.randint(2, 7)):
        text += rng.choice(PIECES) + rng.choice(SEPARATORS)
    chars = list(text)
    for _ in range(rng.randint(1, 3)):
        chars.insert(rng.randint(0, len(chars)), rng.choice(INVISIBLE))
    return "".join(chars)


def _alignments(text: str, out: str) -> tuple[set[int], set[int]] | None:
    """Independent of production code: the characters of *text* that *out* masks under the earliest and under
    the latest alignment of its literal stretches, or None when *out* is not a substitution of *text* at all.
    No fallback: a measure that reported "everything masked" on misalignment would hide exactly the bug."""
    parts, holders = _PLACEHOLDER_RE.split(out), _PLACEHOLDER_RE.findall(out)
    if not text.startswith(parts[0]) or not text.endswith(parts[-1]):
        return None
    last = len(parts) - 1
    early, pos = [0], len(parts[0])  # the first literal is a prefix, the last a suffix: both anchored
    for i in range(1, len(parts)):
        at = len(text) - len(parts[i]) if i == last else text.find(parts[i], pos)
        if at < pos:
            return None
        early.append(at)
        pos = at + len(parts[i])
    late, end = [0] * len(parts), len(text)
    for i in range(last, -1, -1):
        at = 0 if i == 0 else (len(text) - len(parts[i]) if i == last else text.rfind(parts[i], 0, end))
        if at < 0 or at + len(parts[i]) > end:
            return None
        late[i], end = at, at

    def covered(starts: list[int]) -> set[int]:
        return {
            k
            for i in range(len(holders))
            for k in range(starts[i] + len(parts[i]), starts[i + 1])
            if not _is_format(text[k])
        }

    return covered(early), covered(late)


def _exposed(text: str, as_written: Callable[[str], str], public: Callable[[str], str]) -> set[int]:
    """Characters the as-written output may mask (measured independently, see ``_alignments``) that the public
    output leaves visible. The public side uses the ranges production built its output from, after checking
    they reproduce that output exactly, so every character outside them is literally in the output."""
    written = _alignments(text, as_written(text))
    assert written is not None, ("as-written output is not a substitution of the input", text, as_written(text))
    ranges = _union_ranges(text, as_written)
    assert public(text) == _rebuild(text, ranges), ("public output is not built from its ranges", text)
    hidden = {k for start, end, _ in ranges for k in range(start, end)}
    return (written[0] | written[1]) - hidden


@pytest.mark.parametrize(
    ("public", "as_written"),
    [
        (mask_credentials, mask_credentials_as_written),
        (strip_pii, strip_pii_as_written),
        (mask_query_credentials, mask_query_credentials_as_written),
        (default_redactor, _default_redactor_as_written),
    ],
    ids=["mask_credentials", "strip_pii", "mask_query_credentials", "decisions_default_redactor"],
)
def test_the_public_masker_masks_every_character_the_as_written_pass_masks(
    public: Callable[[str], str], as_written: Callable[[str], str]
) -> None:
    rng = random.Random(20261002)
    for _ in range(500):
        text = _case(rng)
        exposed = _exposed(text, as_written, public)
        assert not exposed, (text, as_written(text), public(text), sorted(exposed))


def test_the_auditor_decoy_case_stays_masked() -> None:
    """Auditor B1: a collapsed scheme separator let a later decoy decide the alignment; the token leaked."""
    out = mask_credentials('Bearer\tabcdef123456 PASSWORD="Bearer x y"\u200b')
    assert "abcdef123456" not in out and "x y" not in out, out


def test_a_split_key_in_a_recall_query_is_masked() -> None:
    """Auditor M1: mask_query_credentials (sync/_remote_fetch recall queries) was not wrapped."""
    assert "w6V5u4T3s2R1q0P9o8" not in mask_query_credentials("why sk_live_Z9y8X7\u200bw6V5u4T3s2R1q0P9o8 401s")


def test_a_split_inside_a_bearer_value_does_not_hide_an_ssn_from_strip_pii() -> None:
    """codex r3 P0: the inner credential pass consumed '123' before SSN detection saw '123 45 6789'."""
    assert "45 6789" not in strip_pii("Bearer abcde\u200b123 45 6789")


def test_the_ssn_case_through_the_publish_payload() -> None:
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.sync._remote_publish import _anonymize_entry

    leak = "Bearer abcde\u200b123 45 6789"
    payload = _anonymize_entry(MemoryEntry(id="L-3", content=leak, detail=leak, tags=[leak]), "")
    assert "45 6789" not in str(payload), payload


@pytest.mark.parametrize("mark", ["\u034f", "\u3164", "\ufe0f"], ids=["cgj", "hangul-filler", "vs16"])
def test_a_non_cf_invisible_character_cannot_split_a_secret(mark: str) -> None:
    """Auditor m1: the combining grapheme joiner, Hangul filler and variation selectors also render as nothing."""
    assert "IOSFODNN7EXAMPLE" not in mask_credentials(f"key AKIA{mark}IOSFODNN7EXAMPLE")


def test_the_decisions_redactor_masks_a_split_bearer_tail() -> None:
    """Auditor round 2 M3: decisions/_redaction's own pre-pass consumed the head before strip_pii saw the token."""
    from trw_memory.decisions import default_redactor

    out = default_redactor("Bearer abcdefgh12\u200bZQXWVUTSRQPONMLK9876")
    assert "ZQXWVUTSRQPONMLK9876" not in out, out


def test_an_empty_secret_beside_an_emoji_keeps_the_surrounding_text() -> None:
    """Auditor round 2 M4: a placeholder over an empty stretch forced the mask-everything fallback. An empty
    secret hides nothing, so the text comes back whole instead of as one placeholder."""
    text = '\u26a0\ufe0f {"password": ""} done'
    assert strip_pii(text) == text


def test_two_adjacent_placeholders_map_to_their_own_stretches() -> None:
    """Auditor round 2 M4: an empty literal between two placeholders must not force the fallback."""
    from trw_memory.security._scan_normalize import _masked_ranges

    ranges = _masked_ranges("ab XY cd", "ab <api_key><email> cd")
    # Exactly the 'XY' stretch is masked (each placeholder's hull covers the ambiguous split): no fallback.
    assert {k for start, end, _ in ranges for k in range(start, end)} == {3, 4}, ranges


@pytest.mark.parametrize(
    "as_written",
    [
        mask_credentials_as_written,
        strip_pii_as_written,
        mask_query_credentials_as_written,
        _default_redactor_as_written,
    ],
    ids=["mask_credentials", "strip_pii", "mask_query_credentials", "decisions_default_redactor"],
)
def test_every_as_written_masker_only_substitutes(as_written: Callable[[str], str]) -> None:
    """The contract _masked_ranges assumes: an as-written masker replaces stretches of its input with
    placeholders and changes nothing else (no whitespace collapse, no rewrite), so its output re-aligns onto
    its input with the first literal as a prefix and the last as a suffix. A future masker that rewrites text
    fails here instead of silently driving the union into its mask-everything fallback."""
    rng = random.Random(20261004)
    for _ in range(500):
        text = _case(rng)
        out = as_written(text)
        assert _alignments(text, out) is not None, ("not a pure substitution", text, out)
        # With no placeholder nothing may change at all (codex PII-SCAN-HARDENING r1: a deletion with no
        # placeholder could otherwise still pass the prefix/suffix check).
        if not _PLACEHOLDER_RE.search(out):
            assert out == text, ("text changed without a placeholder", text, out)
