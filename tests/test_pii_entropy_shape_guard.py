"""Precision guard for the Shannon-entropy PII backstop.

Regression suite for the 2026-07-25 data-loss incident: the unguarded entropy
backstop redacted technical identifiers — repo paths, dotted module paths,
snake_case symbols, kebab-case doc slugs, version ranges, ruff rule lists — out of
stored learnings on the write path, before persistence, irreversibly. Measured on
this project's corpus: 83 of 6,197 stored learning files damaged, and the sampled
true-positive rate of those redactions was zero.

The fix is candidate selection only (``_is_structured_technical_token``). It does
not touch ``BLOCKING_PII_TYPES``, the API_KEY detector, or any other PII type, so
every recognised-credential defence is unchanged. These tests hold BOTH halves of
that claim to account:

* precision — the six sentences that were actually destroyed survive intact, and
  each redacted token is proven to have fired the raw entropy rule before the fix,
  so the assertions are non-vacuous;
* recall — randomly generated secrets of the shapes the backstop exists for are
  still detected, demonstrated over a generated population rather than asserted on
  one hand-picked example.
"""

from __future__ import annotations

import random
import string

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security._runtime_pii import BLOCKING_PII_TYPES, apply_runtime_pii_policy
from trw_memory.security.pii import (
    _DEFAULT_ENTROPY_THRESHOLD,
    _MIN_ENTROPY_TOKEN_LEN,
    PIIType,
    _is_structured_technical_token,
    detect_pii,
    shannon_entropy,
)

# ---------------------------------------------------------------------------
# The real destroyed samples
# ---------------------------------------------------------------------------
# Each entry is (context_template, token). The context is the surviving text of a
# learning that the backstop damaged, quoted from the stored corpus. The token is
# the identifier that occupied the redacted span: the raw text is unrecoverable —
# redaction runs on MemoryClient.store BEFORE persistence — so each token is
# reconstructed from the surrounding sentence and drawn from the same corpus.
# ``test_destroyed_tokens_would_have_been_redacted`` pins every one of them to the
# raw entropy rule, so none of these fixtures can silently become vacuous.
DESTROYED_SAMPLES: list[tuple[str, str]] = [
    (
        "records receipt-verified raw-evidence paths as {token} then "
        "delete_candidates(root, cands, gate_via_registry=True,",
        ".trw/runs/independent-prd-consolidation-review/20260711T190521Z-150684bb/meta/events.jsonl",
    ),
    (
        "{token} items #1-#9, #12, #13, #16a/b, #24 shipped across v0.3.0-v0.6.1",
        "docs/research/trw-distill/TRW-DISTILL-ROADMAP-SUPERVISOR-2026-05-25.md#35",  # trw-leak-allow: internal_docs synthetic fixture string, not a real location
    ),
    (
        "-> _update_agents at all - meaning _is_user_modified {token} was dead",
        "trw-mcp/src/trw_mcp/sync/_agents_dispatch.py:88-104,203-247",
    ),
    (
        "metaharness for post-hoc scoring/journal/gates. Plan doc: {token}",
        "docs/requirements-aare-f/exec-plans/EXECUTION-PLAN-PRD-EVAL-048.md",
    ),
    (
        "salvage): (1) fd-based TOCTOU-safe file ops {token} re-verify) -> naive",
        "openat2(RESOLVE_NO_SYMLINKS)+fstat+st_dev/st_ino",
    ),
    (
        "so add {token} to pyproject ruff per-file-ignores (mirrors cli.py's).",
        "src/trw_swarm/_cli_loaders.py:S404,S603,T201,PLR0913",
    ),
]

# Secret shapes the backstop exists for: unrecognised, uniformly random material.
# Hex and UUID are deliberately absent — see test_hex_blobs_are_out_of_reach.
_BASE64 = string.ascii_letters + string.digits + "+/"
_BASE64URL = string.ascii_letters + string.digits + "-_"
_ALNUM = string.ascii_letters + string.digits

_SECRET_FAMILIES: list[tuple[str, str, int]] = [
    ("base64", _BASE64, 32),
    ("base64", _BASE64, 44),
    ("base64", _BASE64, 64),
    ("base64-pem-line", _BASE64, 76),
    ("base64", _BASE64, 88),
    ("base64url", _BASE64URL, 32),
    ("base64url", _BASE64URL, 44),
    ("base64url", _BASE64URL, 64),
    ("mixed-alnum", _ALNUM, 32),
    ("mixed-alnum", _ALNUM, 44),
    ("mixed-alnum", _ALNUM, 64),
]

_SAMPLES_PER_FAMILY = 300


def _generate_secrets(alphabet: str, length: int, count: int, seed: int) -> list[str]:
    """Deterministically generate *count* uniform random tokens."""
    rng = random.Random(seed)
    return ["".join(rng.choice(alphabet) for _ in range(length)) for _ in range(count)]


def _would_fire_raw_entropy_rule(token: str) -> bool:
    """Whether the pre-fix heuristic (length + Shannon entropy) selects *token*."""
    return len(token) >= _MIN_ENTROPY_TOKEN_LEN and shannon_entropy(token) >= _DEFAULT_ENTROPY_THRESHOLD


def _high_entropy_matches(text: str) -> list[str]:
    return [m.value for m in detect_pii(text) if m.pii_type == PIIType.HIGH_ENTROPY]


# ---------------------------------------------------------------------------
# Precision: the destroyed learnings survive
# ---------------------------------------------------------------------------


class TestDestroyedSamplesSurvive:
    """The six real damaged learnings must now round-trip intact."""

    @pytest.mark.parametrize(("context", "token"), DESTROYED_SAMPLES)
    def test_destroyed_tokens_would_have_been_redacted(self, context: str, token: str) -> None:
        """Non-vacuity pin: every fixture token fires the raw pre-fix heuristic.

        Without this, a future edit could weaken a fixture into a token the
        backstop never selected and the survival tests would pass for free.
        """
        assert _would_fire_raw_entropy_rule(token), f"fixture no longer exercises the backstop: {token!r}"

    @pytest.mark.parametrize(("context", "token"), DESTROYED_SAMPLES)
    def test_token_is_recognised_as_a_technical_identifier(self, context: str, token: str) -> None:
        assert _is_structured_technical_token(token) is True

    @pytest.mark.parametrize(("context", "token"), DESTROYED_SAMPLES)
    def test_no_high_entropy_match_on_destroyed_sample(self, context: str, token: str) -> None:
        assert _high_entropy_matches(context.format(token=token)) == []

    @pytest.mark.parametrize(("context", "token"), DESTROYED_SAMPLES)
    def test_store_path_preserves_the_sentence_verbatim(self, context: str, token: str) -> None:
        """The store path — where the loss actually happened — must not mutate it."""
        text = context.format(token=token)
        entry = MemoryEntry(id="M-destroyed", content=text, detail=text)
        result, _matches = apply_runtime_pii_policy(entry, MemoryConfig())
        assert result.content == text
        assert result.detail == text
        assert token in result.content
        assert "contains_high_entropy_token" not in result.metadata


class TestStructuredTokenShapes:
    """The identifier classes the guard must recognise."""

    @pytest.mark.parametrize(
        "token",
        [
            "/home/dev/projects/example-repo/trw-memory/src/trw_memory/security/pii.py",  # trw-leak-allow: machine_path synthetic fixture string, not a real location
            "docs/requirements-aare-f/prds/PRD-INFRA-054.md",
            "trw_memory.security._runtime_pii.apply_runtime_pii_policy",
            "HIGH_ENTROPY_ELISION_PREFIX_LEN",
            "_is_structured_technical_token",
            "docs/research/trw-distill/SC2-C806-PRESERVE-HYBRID-ORDER-2026-05-19.md",  # trw-leak-allow: internal_docs synthetic fixture string, not a real location
            "https://github.com/wallter/trw-framework/blob/357e6e163b5d/.github/workflows/eval-ci.yml",
            "357e6e163b5d0d5c1158e14d606d425058620f5e..9fe8bcfd7fc109e66294798052cc21bda3dbc667",
            "v0.3.0-v0.6.1",
            "20260711T190521Z-150684bb",
        ],
    )
    def test_recognised_as_structured(self, token: str) -> None:
        assert _is_structured_technical_token(token) is True

    @pytest.mark.parametrize(
        "token",
        [
            "C:\\Users\\Tyler\\Desktop\\trw-framework\\docs\\index.md",
            "platform/src/app/(marketing)/about/AboutContent.tsx:98-104",  # trw-leak-allow: proprietary_path synthetic fixture string, not a real location
            "OllamaSynthesisClient(SynthesisConfig(primary_model='qwen3.6:35b-a3b')))",
        ],
    )
    def test_known_residual_camelcase_tokens_are_not_excluded(self, token: str) -> None:
        """Documents the deliberate limit of the guard, so it stays visible.

        CamelCase runs mix case inside a single segment, which is exactly the
        signature the guard uses to identify random material, so they are NOT
        excluded. A CamelCase-tolerant variant was measured against generated
        secrets and rejected: it raised corpus false-positive suppression from
        88.9% to 95.6% but lost 13.7% of true positives, because a random
        mixed-case run is frequently a valid CamelCase parse.

        This costs less than it appears: Windows drive paths are matched by the
        FILE_PATH detector first, so ``already_matched`` keeps the entropy branch
        off them regardless. The residual is 92 of the 832 corpus tokens.
        """
        assert _is_structured_technical_token(token) is False

    def test_windows_paths_are_claimed_by_the_file_path_detector(self) -> None:
        """Why the CamelCase residual does not re-damage Windows paths."""
        windows_path = "C:\\Users\\Tyler\\Desktop\\trw-framework\\docs\\index.md"
        types = {m.pii_type for m in detect_pii(windows_path)}
        assert PIIType.FILE_PATH in types
        assert PIIType.HIGH_ENTROPY not in types

    @pytest.mark.parametrize(
        "token",
        [
            # Undelimited blobs: a single alphanumeric run is never excluded.
            "aB3cD9eF2gH5iJ8kL1mN4oP7qR6sT0",
            "JBSWY3DPEHPK3PXPJBSWY3DPEHPK3PXP",
            # Delimited, but the runs mix case at random - i.e. not identifiers.
            "kR8x/Qm2LpZa9WvTn4Yb+Hd7Fj1Ug6Es3Cw0Ix5Ok2Mz",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dQw4w9WgXcQpNvB7yTkLmR",
        ],
    )
    def test_not_recognised_as_structured(self, token: str) -> None:
        assert _is_structured_technical_token(token) is False

    def test_single_segment_blob_is_never_excluded(self) -> None:
        """The load-bearing safety property: no separators means no exclusion."""
        for _name, alphabet, length in _SECRET_FAMILIES:
            for token in _generate_secrets(alphabet.replace("/", "").replace("+", ""), length, 50, seed=11):
                assert _is_structured_technical_token(token) is False


# ---------------------------------------------------------------------------
# Recall: genuine secrets are still caught
# ---------------------------------------------------------------------------


class TestGenuineSecretsStillDetected:
    """Demonstrated over a generated population, not asserted on one example."""

    @pytest.mark.parametrize(("family", "alphabet", "length"), _SECRET_FAMILIES)
    def test_random_secrets_are_never_excluded_by_the_guard(self, family: str, alphabet: str, length: int) -> None:
        """Direct proof of zero recall cost: of every generated secret the pre-fix
        heuristic would have selected, the shape guard excludes none."""
        secrets = _generate_secrets(alphabet, length, _SAMPLES_PER_FAMILY, seed=20260725)
        selected = [token for token in secrets if _would_fire_raw_entropy_rule(token)]
        assert selected, f"{family}/{length} generated no detectable secrets - fixture is vacuous"
        excluded = [token for token in selected if _is_structured_technical_token(token)]
        assert excluded == [], f"{family}/{length}: guard swallowed {len(excluded)}/{len(selected)} real secrets"

    @pytest.mark.parametrize(("family", "alphabet", "length"), _SECRET_FAMILIES)
    def test_random_secrets_still_reach_a_detector(self, family: str, alphabet: str, length: int) -> None:
        """End-to-end through detect_pii: no generated secret is stored silently."""
        secrets = _generate_secrets(alphabet, length, _SAMPLES_PER_FAMILY, seed=20260725)
        selected = [token for token in secrets if _would_fire_raw_entropy_rule(token)]
        undetected = [token for token in selected if not detect_pii(token)]
        assert undetected == [], f"{family}/{length}: {len(undetected)} secrets now pass undetected"

    def test_jwt_shaped_secret_still_detected(self) -> None:
        rng = random.Random(4242)
        jwt = ".".join("".join(rng.choice(_BASE64URL) for _ in range(size)) for size in (36, 60, 43))
        assert _would_fire_raw_entropy_rule(jwt)
        assert _high_entropy_matches(jwt) == [jwt]

    def test_hex_blobs_are_out_of_reach_of_this_backstop(self) -> None:
        """Documents a PRE-EXISTING limit, unchanged by the guard.

        Shannon entropy over a 16-symbol alphabet cannot exceed 4.0 bits/char, so a
        hex digest can never reach the 4.5 default threshold — it was never caught
        before this change either. Recorded here so the gap is visible rather than
        mistaken for a regression introduced by the shape guard.
        """
        rng = random.Random(99)
        for length in (32, 40, 64):
            blob = "".join(rng.choice("0123456789abcdef") for _ in range(length))
            assert shannon_entropy(blob) < 4.0 + 1e-9
            assert not _would_fire_raw_entropy_rule(blob)


class TestBackstopSignalPreserved:
    """The metadata flag and the recognised-secret defences are untouched."""

    def test_high_entropy_metadata_flag_still_fires(self) -> None:
        rng = random.Random(7)
        secret = "".join(rng.choice(_BASE64URL) for _ in range(64))
        assert _would_fire_raw_entropy_rule(secret)
        entry = MemoryEntry(id="M-secret", content=f"pasted token {secret} into the config")
        result, matches = apply_runtime_pii_policy(entry, MemoryConfig())
        assert result.metadata["contains_high_entropy_token"] == "true"
        assert "high_entropy" in result.metadata["pii_types"]
        assert any(m.pii_type == PIIType.HIGH_ENTROPY for m in matches)
        # 2026-07-25: the SIGNAL is what this class pins, not a mutation. The
        # store path no longer rewrites content for HIGH_ENTROPY — the flag and
        # the match list are the whole observable effect, and the user's text
        # stays exactly as written. Sanitization happens at egress (strip_pii).
        assert secret in result.content

    def test_blocking_pii_types_unchanged(self) -> None:
        assert BLOCKING_PII_TYPES == frozenset({PIIType.API_KEY})

    @pytest.mark.parametrize(
        "credential",
        [
            "sk-abcdefghijklmnopqrstuvwxyz012345",
            "secret_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123",
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "AKIAIOSFODNN7EXAMPLE",
        ],
    )
    def test_recognised_credentials_still_detected_as_api_key(self, credential: str) -> None:
        """These shapes are case-uniform per segment, so the guard could only have
        weakened them if it were wired anywhere other than the entropy branch."""
        matches = detect_pii(f"leaked {credential} in the log")
        assert any(m.pii_type == PIIType.API_KEY for m in matches), credential

    @pytest.mark.parametrize(
        "credential",
        [
            "sk-abcdefghijklmnopqrstuvwxyz012345",
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        ],
    )
    def test_recognised_credentials_still_block_the_store(self, credential: str) -> None:
        from trw_memory.exceptions import PIIBlockError

        entry = MemoryEntry(id="M-cred", content=f"token {credential} leaked")
        with pytest.raises(PIIBlockError):
            apply_runtime_pii_policy(entry, MemoryConfig())
