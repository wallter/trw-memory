"""PII-INVISIBLE-SPLIT: a secret split by an invisible character must not leave the box.

Egress redaction matched patterns on the raw text, so a zero-width space, soft hyphen or bidi control
inside an email, key or token broke the match and the secret went out intact (or, for a bearer token,
half-redacted). The maskers now scan a normalised copy (format characters dropped, NFKC) and emit it.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from trw_memory.security.credentials import mask_credentials
from trw_memory.security.pii import strip_pii

ZWSP, SHY, RLO, ZWJ, WJ, BOM = "\u200b", "\u00ad", "\u202e", "\u200d", "\u2060", "\ufeff"
AWS = "AKIAIOSFODNN7EXAMPLE"
GH = "ghp_" + "a1B2c3D4e5F6g7H8i9J0" * 2

SPLIT_SECRETS = [
    pytest.param(f"key {AWS[:4]}{ZWSP}{AWS[4:]} here", AWS, id="aws-zero-width-space"),
    pytest.param(f"key {AWS[:6]}{SHY}{AWS[6:]} here", AWS, id="aws-soft-hyphen"),
    pytest.param(f"key {AWS[:8]}{RLO}{AWS[8:]} here", AWS, id="aws-bidi-override"),
    pytest.param(f"token {GH[:20]}{ZWSP}{GH[20:]}", GH, id="github-token-zero-width"),
    pytest.param(f"token {GH[:12]}{SHY}{GH[12:30]}{RLO}{GH[30:]}", GH, id="github-token-soft-hyphen-and-bidi"),
    pytest.param(f"mail jane{ZWJ}.doe@exam{WJ}ple.com now", "jane.doe@example.com", id="email-zwj-word-joiner"),
    pytest.param(f"mail {BOM}jane.doe@example{SHY}.com now", "jane.doe@example.com", id="email-bom-soft-hyphen"),
]


def _pieces(secret: str) -> list[str]:
    """Every 8-character window of the secret: no fragment that long may survive anywhere in the output."""
    return [secret[i : i + 8] for i in range(0, len(secret) - 7, 4)]


CASES = [(redact, case) for redact in (mask_credentials, strip_pii) for case in SPLIT_SECRETS]


@pytest.mark.parametrize(
    ("redact", "text", "secret"),
    [
        pytest.param(redact, *case.values, id=f"{redact.__name__}-{case.id}")
        for redact, case in CASES
        # Emails are strip_pii's job; mask_credentials masks credentials only.
        if not (redact is mask_credentials and "@" in case.values[1])
    ],
)
def test_a_secret_split_by_an_invisible_character_is_redacted(
    redact: Callable[[str], str], text: str, secret: str
) -> None:
    out = redact(text)
    leaked = [piece for piece in _pieces(secret) if piece in out]
    assert leaked == [], (out, leaked)
    # Invisible characters OUTSIDE a masked range are emitted as written (unmasked text is never rewritten).


@pytest.mark.parametrize(
    "prose",
    ["José met Zoë at the café", "東京で会いましょう", "naïve façade — résumé", "Ελληνικά και русский"],
)
def test_ordinary_non_ascii_prose_passes_through_unchanged(prose: str) -> None:
    """NFKC must not create a false match or rewrite ordinary composed text."""
    assert strip_pii(prose) == prose
    assert mask_credentials(prose) == prose


def test_normalisation_is_idempotent_and_keeps_existing_placeholders() -> None:
    text = f"key {AWS[:4]}{ZWSP}{AWS[4:]} mail jane.doe@example.com"
    once = strip_pii(text)
    assert strip_pii(once) == once
    assert AWS[4:] not in once and "api_key>" in once and "<email>" in once


def test_a_decomposed_unicode_project_root_is_still_anonymised_on_publish() -> None:
    """codex r1 P0-3: normalising the content before redact_paths broke the exact match on the root."""
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.sync._remote_publish import _anonymize_entry

    root = "/Users/José/project"
    entry = MemoryEntry(id="L-1", content=root + "/private/file.py", detail="")
    payload = _anonymize_entry(entry, root)
    assert payload["summary"] == "<project>/private/file.py", payload["summary"]


def test_a_split_quoted_password_never_reaches_the_publish_payload() -> None:
    """codex r2 P0 through the outbound publisher: summary, detail and tags."""
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.sync._remote_publish import _anonymize_entry

    leak = 'PASSWORD\u200b="first SensitiveSuffixABC"'
    entry = MemoryEntry(id="L-2", content=leak, detail=leak, tags=[leak])
    payload = _anonymize_entry(entry, "")
    assert "SensitiveSuffixABC" not in str(payload), payload
