"""PII detection pipeline using regex patterns and Shannon entropy analysis.

Detects common PII types (email, phone, SSN, credit card, API keys) and
high-entropy strings that may contain secrets.  Supports three actions:
block (reject the entry), redact (mask and allow), and warn (log but allow).

Also provides pure anonymization helpers (strip_pii, redact_paths,
anonymize_installation_id) for telemetry data that must not contain PII.
"""

from __future__ import annotations

import hashlib
import math
import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from trw_memory.exceptions import ConfigError, MemoryError
from trw_memory.models.memory import MemoryEntry


class PIIType(str, Enum):
    """Categories of personally identifiable information."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    API_KEY = "api_key"
    IP_ADDRESS = "ip_address"
    FILE_PATH = "file_path"
    CUSTOM = "custom"
    HIGH_ENTROPY = "high_entropy"


class PIIMatch(BaseModel):
    """A single PII detection match with location and confidence."""

    model_config = ConfigDict(strict=True, use_enum_values=True)

    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float = Field(ge=0.0, le=1.0)


class PIIAction(str, Enum):
    """Action to take when PII is detected."""

    BLOCK = "block"
    REDACT = "redact"
    WARN = "warn"


# ---------------------------------------------------------------------------
# Shared API-key / secret pattern sources (single source of truth)
# ---------------------------------------------------------------------------
# These raw pattern strings are referenced by BOTH the detection set
# (``_PII_PATTERNS``, used by detect_pii / the live store-time gate) AND the
# anonymization helper ``strip_pii`` (used for telemetry + shadow-quarantine
# scrubbing). Keeping one source of truth prevents the divergence found in the
# 2026-06-17 audit, where strip_pii's inlined copy omitted the ``secret`` prefix
# so a ``secret-<token>`` credential was blocked at store time but written
# verbatim to the shadow-quarantine JSONL. Add a new credential shape HERE and
# both paths stay in sync.

# Generic ``<prefix>[-_]<20+ alnum>`` credential shape.
_SECRET_PREFIX_PATTERN = r"(?:sk|pk|api|key|token|secret)[-_][a-zA-Z0-9]{20,}"  # noqa: S105 — regex, not a credential
# Provider-specific shapes that lack a "<prefix>[-_]" separator and fall below
# the Shannon-entropy backstop (GitHub PATs, AWS access key IDs). Anchored +
# bounded (no nested quantifiers) so they stay ReDoS-free.
_PROVIDER_SECRET_PATTERN = (
    r"\b(?:gh[posru]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})\b|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"  # noqa: S105 — regex, not a credential
)

# ---------------------------------------------------------------------------
# Regex patterns for each PII type
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[PIIType, re.Pattern[str], float]] = [
    (
        PIIType.EMAIL,
        re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        0.95,
    ),
    (
        PIIType.PHONE,
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        0.8,
    ),
    (
        PIIType.SSN,
        re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
        0.9,
    ),
    (
        PIIType.CREDIT_CARD,
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        0.85,
    ),
    (
        PIIType.API_KEY,
        re.compile(
            r"\b" + _SECRET_PREFIX_PATTERN + r"\b",
            re.IGNORECASE,
        ),
        0.9,
    ),
    # Provider-specific secret shapes that lack a "<prefix>[-_]" separator and
    # fall below the Shannon-entropy backstop (e.g. a 40-char GitHub PAT scores
    # ~4.1 bits/char, under the 4.5 default). These leaked silently before — a
    # token in `content`/`detail` matched neither the generic API_KEY pattern
    # nor the high-entropy path. See _PROVIDER_SECRET_PATTERN (shared with
    # strip_pii).
    (
        PIIType.API_KEY,
        re.compile(_PROVIDER_SECRET_PATTERN),
        0.95,
    ),
    (
        PIIType.IP_ADDRESS,
        # Octet-range-validated IPv4 (each octet 0-255) so version strings like
        # "3.11.0.2" or "999.300.1.500" are NOT redacted (closure re-audit #3:
        # the old ``\b(?:\d{1,3}\.){3}\d{1,3}\b`` over-matched any 4 dotted
        # digit-runs, corrupting content with false-positive redaction). The
        # ``(?<![\d.])`` / ``(?![\d.])`` guards stop a valid 4-octet subspan
        # from being carved out of a longer dotted-number run (e.g. the leading
        # "3." or a trailing ".5" in a 5-segment version).
        re.compile(
            r"(?<![\d.])"
            r"(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
            r"(?![\d.])"
        ),
        0.85,
    ),
    (
        PIIType.FILE_PATH,
        # A POSIX absolute path must start at a boundary (start-of-string or
        # whitespace) — NOT mid-token. The negative lookbehind ``(?<![\w:/.])``
        # stops matching URL path components such as ``example.com/api/users``
        # or ``https://host/a/b`` (where the leading ``/`` is preceded by a
        # host char or ``:``/``/``). Windows drive paths are matched separately.
        re.compile(r"(?:[A-Za-z]:\\[^\s]+|(?<![\w:/.])/(?:[^/\s]+/)+[^/\s]+)"),
        0.8,
    ),
]

# Minimum token length and entropy threshold for high-entropy detection
_MIN_ENTROPY_TOKEN_LEN = 20
_DEFAULT_ENTROPY_THRESHOLD = 4.5

# ---------------------------------------------------------------------------
# Shape guard for the high-entropy backstop
# ---------------------------------------------------------------------------
# The entropy branch is a BACKSTOP for credential shapes no detector recognises.
# Recognised credentials are PIIType.API_KEY and BLOCK the store outright; EMAIL,
# IP_ADDRESS, SSN, CREDIT_CARD and PHONE have their own detectors. So the entropy
# branch's only job is unrecognised, high-randomness material — and its candidate
# set should therefore exclude anything that is recognisably NOT random.
#
# Measured on this project's stored-learning corpus (6,197 entries, 2026-07-25),
# the unguarded heuristic fired on 832 distinct tokens and every sampled one was a
# technical identifier — relative repo paths, dotted module paths, snake_case and
# SCREAMING_SNAKE symbols, kebab-case doc slugs, URLs, version ranges and ruff rule
# lists. Those tokens ARE the substance of an engineering learning, and redacting
# them destroyed the sentence the learning existed to record.
#
# The discriminator is internal structure. A real random secret is drawn uniformly
# from its alphabet, so within any run of alphanumerics it mixes upper- and
# lower-case at random. A human-authored identifier does not: split it on its
# separators (``/``, ``.``, ``_``, ``-``, ``:``, punctuation) and every resulting
# run is case-uniform — all-lower (``requirements``, ``py``, a git SHA), all-upper
# (``PRD``, ``INFRA``, ``T190521Z``), or purely numeric (versions, line numbers).
#
# We therefore skip a token only when it decomposes into >= 2 alphanumeric runs
# that are ALL case-uniform. Requiring >= 2 runs is what keeps the backstop honest:
# a bare undelimited blob is never skipped, so a pasted secret in its native shape
# still gets caught no matter what its alphabet is.
#
# Measured effect of this guard (see tests/test_pii_entropy_shape_guard.py):
#   * corpus false positives 832 -> 92 distinct tokens (88.9% of the damage removed)
#   * true positives 0 lost out of 97,702 detections across random base64, base64url,
#     mixed alphanumeric, JWT and PEM-line families at 24/32/44/64/76/88 chars.
# A CamelCase-tolerant variant was measured and REJECTED: it reached 95.6% FP
# suppression but lost 13.7% of true positives, because a random mixed-case run is
# frequently a valid CamelCase parse. Precision is not worth recall here.
_ENTROPY_SEGMENT_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")
_MIN_STRUCTURED_SEGMENTS = 2

# Closure re-audit #3: a 4-segment dotted run whose octets are all in range is
# ambiguous between a real IPv4 address and a software version string (e.g.
# "3.11.0.2"). When the dotted run is introduced by a version-context word
# ("python 3.11.0.2", "version 1.2.3.4", "v2.0.0.1") we treat it as a version
# and suppress the IP redaction so it does not corrupt content. The context
# word is matched case-insensitively immediately before the match (allowing a
# single ``v``/``V`` prefix attached to the number).
_VERSION_CONTEXT_WORDS = frozenset(
    {
        "v",
        "version",
        "ver",
        "release",
        "rel",
        "build",
        "python",
        "py",
        "node",
        "nodejs",
        "ruby",
        "go",
        "rust",
        "java",
        "php",
        "perl",
        "gcc",
        "clang",
        "kernel",
        "linux",
        "ubuntu",
        "debian",
        "package",
        "pkg",
        "upgraded",
        "downgraded",
        "bumped",
        "tag",
        # Common package / tool names that precede a dotted version string. Without
        # these, "mysql 3.0.0.5" / "openssl 1.1.1.2" had their version octet-valid
        # dotted run false-positive-redacted as an IPv4 address (trw-memory-9).
        "mysql",
        "postgres",
        "postgresql",
        "redis",
        "nginx",
        "apache",
        "openssl",
        "docker",
        "kubernetes",
        "k8s",
        "npm",
        "pip",
        "cargo",
        "gradle",
        "maven",
        "django",
        "flask",
        "react",
        "vue",
        "angular",
        "typescript",
        "deno",
        "dotnet",
        "kotlin",
        "swift",
        "scala",
        "elixir",
        "erlang",
    }
)
_VERSION_PREFIX_RE = re.compile(r"([A-Za-z]+)\s*v?$")

# ReDoS hardening for caller-supplied custom patterns. Python's ``re`` engine
# has no execution timeout, so a single pathological pattern could hang the
# store path on attacker-influenced text. We bound count + length and reject
# the classic catastrophic-backtracking shapes (a quantifier applied to a group
# that itself ends in a quantifier, e.g. ``(a+)+`` / ``(a*)*`` / ``(a+)*``).
_MAX_CUSTOM_PATTERNS = 50
_MAX_CUSTOM_PATTERN_LEN = 1000
# Matches a quantifier ( * + {m,n} ) applied to a ``)`` that closes a group
# whose final atom is itself quantified ( * + {m,n} ) — the nested-quantifier
# ReDoS trigger. Examples caught: (a+)+ (a*)* (\d+)* (x{1,3}){2,} (ab+)+
_INNER_QUANTIFIER = r"(?:[*+]|\{\d+(?:,\d*)?\})"
_OUTER_QUANTIFIER = r"(?:[*+]|\{\d+(?:,\d*)?\})"
_NESTED_QUANTIFIER = re.compile(r"\([^)]*" + _INNER_QUANTIFIER + r"\)\s*" + _OUTER_QUANTIFIER)
# Matches a quantifier ( * + {m,n} ) applied to a ``)`` that closes a group
# containing an alternation ( ``|`` ). Quantified alternation with overlapping
# branches (the classic ``(a|a)*`` / ``(a|ab)*`` / ``(x|x|y)+``) is a second
# catastrophic-backtracking family the nested-quantifier heuristic misses, since
# none of the branches need carry its own quantifier. We can't cheaply prove the
# branches overlap, so we conservatively reject any quantified group that holds a
# top-level ``|``. Examples caught: (a|a)* (foo|bar)+ (\d|\d){2,}
_ALTERNATION_QUANTIFIER = re.compile(r"\([^)|]*\|[^)]*\)\s*" + _OUTER_QUANTIFIER)


def _validate_custom_pattern(raw_pattern: str) -> None:
    """Reject custom patterns that risk catastrophic backtracking (ReDoS).

    Raises:
        ConfigError: If the pattern is too long or contains a nested-quantifier
            backtracking trigger.
    """
    if len(raw_pattern) > _MAX_CUSTOM_PATTERN_LEN:
        raise ConfigError(
            f"custom PII pattern exceeds {_MAX_CUSTOM_PATTERN_LEN} chars "
            f"(got {len(raw_pattern)}); rejected as a ReDoS safety measure"
        )
    if _NESTED_QUANTIFIER.search(raw_pattern):
        raise ConfigError(
            "custom PII pattern contains a nested quantifier "
            "(catastrophic-backtracking risk); rejected as a ReDoS safety measure"
        )
    if _ALTERNATION_QUANTIFIER.search(raw_pattern):
        raise ConfigError(
            "custom PII pattern contains a quantified alternation group "
            "(catastrophic-backtracking risk); rejected as a ReDoS safety measure"
        )
    # Trial-compile so a syntactically-invalid caller pattern (e.g. ``[``) is
    # rejected as a documented ConfigError at validation time rather than raising
    # an uncaught re.error from re.compile in detect_pii — which previously
    # crashed every store for the session. The length + backtracking guards above
    # run first so we never trial-compile a known-pathological pattern.
    try:
        re.compile(raw_pattern)
    except re.error as exc:
        raise ConfigError(f"custom PII pattern is not a valid regular expression: {exc}") from exc


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy in bits per character.

    Args:
        text: The string to analyze.

    Returns:
        Entropy in bits/char.  Returns ``0.0`` for empty strings.
    """
    if not text:
        return 0.0

    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    length = len(text)
    entropy = 0.0
    for count in freq.values():
        prob = count / length
        if prob > 0:
            entropy -= prob * math.log2(prob)

    return entropy


def _is_case_uniform(segment: str) -> bool:
    """Return True when *segment* never mixes upper- and lower-case letters.

    Segments come from ``_ENTROPY_SEGMENT_SPLIT_RE`` so they are ASCII
    alphanumerics only. Digits are case-neutral and never disqualify a segment,
    which is what lets ``T190521Z``, ``2149`` and ``0f2a9c`` read as identifier
    material. Scans left to right and stops at the first mixed-case proof, so it
    is linear in the segment length with no backtracking.
    """
    has_lower = False
    has_upper = False
    for char in segment:
        if char.islower():
            has_lower = True
        elif char.isupper():
            has_upper = True
        if has_lower and has_upper:
            return False
    return True


def _is_structured_technical_token(token: str) -> bool:
    """Return True when *token* is recognisably a technical identifier.

    Such tokens are excluded from the Shannon-entropy backstop before it fires —
    see the shape-guard rationale above ``_ENTROPY_SEGMENT_SPLIT_RE``. This does
    NOT relax any recognised-secret defence: API_KEY still blocks the store, and
    EMAIL / IP_ADDRESS / SSN / CREDIT_CARD / PHONE keep their own detectors.

    A token qualifies when it splits into at least ``_MIN_STRUCTURED_SEGMENTS``
    alphanumeric runs on non-alphanumeric separators AND every run is
    case-uniform. Covers filesystem paths, dotted module paths, snake_case,
    SCREAMING_SNAKE, kebab-case slugs, URLs, git ranges and version strings.
    An undelimited blob yields a single run and is therefore never excluded.
    """
    segments = [segment for segment in _ENTROPY_SEGMENT_SPLIT_RE.split(token) if segment]
    if len(segments) < _MIN_STRUCTURED_SEGMENTS:
        return False
    return all(_is_case_uniform(segment) for segment in segments)


def _is_version_context(text: str, ip_start: int) -> bool:
    """Return True when the dotted run at *ip_start* is preceded by a version word.

    Used to suppress IPv4 false positives on octet-valid version strings such as
    ``python 3.11.0.2`` or ``v1.2.3.4`` (closure re-audit #3). Matches the word
    immediately before the run (case-insensitive, tolerating a trailing ``v``
    prefix attached to the number) against ``_VERSION_CONTEXT_WORDS``.
    """
    prefix = _VERSION_PREFIX_RE.search(text[:ip_start])
    if prefix is None:
        return False
    return prefix.group(1).lower() in _VERSION_CONTEXT_WORDS


def detect_pii(
    text: str,
    entropy_threshold: float = _DEFAULT_ENTROPY_THRESHOLD,
    custom_patterns: list[str] | None = None,
) -> list[PIIMatch]:
    """Scan *text* for PII using regex patterns and entropy analysis.

    Args:
        text: The text to scan.
        entropy_threshold: Minimum Shannon entropy (bits/char) for a token
            to be flagged as a high-entropy secret.

    Returns:
        List of :class:`PIIMatch` instances for every detection.
    """
    matches: list[PIIMatch] = []

    # Regex-based detection
    for pii_type, pattern, confidence in _PII_PATTERNS:
        for m in pattern.finditer(text):
            if pii_type == PIIType.IP_ADDRESS and _is_version_context(text, m.start()):
                # Suppress a version string masquerading as an IPv4 address so
                # we never false-positive-redact (closure re-audit #3).
                continue
            matches.append(
                PIIMatch(
                    pii_type=pii_type,
                    value=m.group(),
                    start=m.start(),
                    end=m.end(),
                    confidence=confidence,
                )
            )

    patterns = custom_patterns or []
    if len(patterns) > _MAX_CUSTOM_PATTERNS:
        raise ConfigError(
            f"too many custom PII patterns: {len(patterns)} > {_MAX_CUSTOM_PATTERNS} "
            "(rejected as a ReDoS / resource-exhaustion safety measure)"
        )
    for raw_pattern in patterns:
        _validate_custom_pattern(raw_pattern)
        pattern = re.compile(raw_pattern)
        matches.extend(
            PIIMatch(
                pii_type=PIIType.CUSTOM,
                value=m.group(),
                start=m.start(),
                end=m.end(),
                confidence=0.9,
            )
            for m in pattern.finditer(text)
        )

    # High-entropy token detection
    # Split on whitespace and common delimiters to get tokens
    for m in re.finditer(r"\S+", text):
        token = m.group()
        if len(token) < _MIN_ENTROPY_TOKEN_LEN:
            continue
        if _is_structured_technical_token(token):
            # Shape guard: a path / dotted module / snake_case / kebab-case /
            # URL / version token, not a secret. Skipping it here is candidate
            # selection only — every recognised credential shape is matched by
            # the regex passes above and is unaffected.
            continue
        # Skip tokens already matched by regex patterns
        token_start = m.start()
        token_end = m.end()
        already_matched = any(pm.start <= token_start and pm.end >= token_end for pm in matches)
        if already_matched:
            continue
        ent = shannon_entropy(token)
        if ent >= entropy_threshold:
            matches.append(
                PIIMatch(
                    pii_type=PIIType.HIGH_ENTROPY,
                    value=token,
                    start=token_start,
                    end=token_end,
                    confidence=min(ent / 6.0, 1.0),
                )
            )

    return matches


def redact_text(text: str, matches: list[PIIMatch]) -> str:
    """Replace PII matches with redaction markers.

    Matches are processed in reverse order of position so that earlier
    offsets remain valid after replacement.

    Args:
        text: Original text.
        matches: PII matches to redact.

    Returns:
        Text with PII replaced by ``[REDACTED:<type>]`` markers.
    """
    if not matches:
        return text

    # Sort by start position descending so replacements don't shift offsets
    sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)
    result = text
    for match in sorted_matches:
        marker = f"[REDACTED:{match.pii_type}]"
        result = result[: match.start] + marker + result[match.end :]
    return result


def check_entry_pii(
    entry: MemoryEntry,
    action: PIIAction = PIIAction.WARN,
    entropy_threshold: float = _DEFAULT_ENTROPY_THRESHOLD,
) -> tuple[MemoryEntry, list[PIIMatch]]:
    """Check ``content``, ``detail`` and ``tags`` fields for PII and apply *action*.

    Args:
        entry: The memory entry to check.
        action: What to do when PII is found: block, redact, or warn.
        entropy_threshold: Minimum Shannon entropy for high-entropy detection.

    Returns:
        A tuple of ``(possibly_modified_entry, all_matches)``.

    Raises:
        MemoryError: If *action* is ``BLOCK`` and PII is detected.
    """
    # Scan content, detail AND tags. Security audit 2026-06-09 (v0.9.2): the
    # internal runtime path (apply_runtime_pii_policy) scanned tags in v0.9.1, but
    # this PUBLIC API still ignored them — so a credential or PII hidden in a tag
    # returned a false-clean result to direct callers and was surfaced verbatim at
    # recall time. Scan tags here for parity with the runtime path.
    content_matches = detect_pii(entry.content, entropy_threshold)
    detail_matches = detect_pii(entry.detail, entropy_threshold)
    tag_matches_by_index: list[list[PIIMatch]] = [detect_pii(tag, entropy_threshold) for tag in entry.tags]
    tag_matches = [match for matches in tag_matches_by_index for match in matches]

    all_matches = content_matches + detail_matches + tag_matches

    if not all_matches:
        return (entry, [])

    if action == PIIAction.BLOCK:
        pii_types = {m.pii_type for m in all_matches}
        raise MemoryError(
            f"PII detected in entry {entry.id!r}: types={sorted(pii_types)}. Entry blocked by PII policy."
        )

    if action == PIIAction.REDACT:
        new_content = redact_text(entry.content, content_matches)
        new_detail = redact_text(entry.detail, detail_matches)
        new_tags = [redact_text(tag, matches) for tag, matches in zip(entry.tags, tag_matches_by_index, strict=True)]
        updated = entry.model_copy(
            update={"content": new_content, "detail": new_detail, "tags": new_tags},
        )
        return (updated, all_matches)

    # PIIAction.WARN — return entry unchanged with matches for logging
    return (entry, all_matches)


# ---------------------------------------------------------------------------
# Anonymization helpers (telemetry-safe, non-reversible)
# ---------------------------------------------------------------------------


# Markers for the detector-driven second pass in strip_pii. EMAIL and API_KEY
# are NOT listed here — they are handled by the dedicated re.sub calls below,
# which must run first (see the ordering note in strip_pii).
_EGRESS_MARKERS: dict[PIIType, str] = {
    PIIType.PHONE: "<phone>",
    PIIType.SSN: "<ssn>",
    PIIType.CREDIT_CARD: "<credit_card>",
    PIIType.IP_ADDRESS: "<ip>",
}


def strip_pii(text: str) -> str:
    """Remove personal identifiers and credentials from text leaving the machine.

    Replaces recognised PII patterns with safe placeholders:

    * Email addresses -> ``<email>``
    * Common API key / token patterns -> ``<api_key>``
    * Phone / SSN / credit-card / IPv4 shapes -> ``<phone>`` / ``<ssn>`` /
      ``<credit_card>`` / ``<ip>``

    This is an EGRESS helper — it runs at the publish boundary
    (``sync/_remote_publish``) and on the shadow-quarantine record, never on the
    stored row. The store path deliberately keeps the user's text verbatim
    (see ``security/_runtime_pii``), so sanitizing here is reversible: the
    unmasked original is still on the user's disk. The last four types were
    added 2026-07-25 to keep egress coverage at parity with what the store path
    used to mask, now that the store path no longer mutates anything.
    """
    # Email addresses
    text = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "<email>",
        text,
    )
    # API key / token patterns (prefix followed by 20+ alphanumeric chars).
    # Shares _SECRET_PREFIX_PATTERN with _PII_PATTERNS so the ``secret`` prefix
    # (and any future addition) stays in sync — the 2026-06-17 audit found this
    # inlined copy omitted ``secret``, leaking ``secret-<token>`` credentials to
    # the shadow-quarantine JSONL though detect_pii blocked them at store time.
    text = re.sub(
        _SECRET_PREFIX_PATTERN,
        "<api_key>",
        text,
        flags=re.IGNORECASE,
    )
    # Provider-specific secret shapes without a "<prefix>[-_]" separator
    # (GitHub PATs, AWS access key IDs) — shared with _PII_PATTERNS.
    text = re.sub(
        _PROVIDER_SECRET_PATTERN,
        "<api_key>",
        text,
    )
    # Second pass, detector-driven so it inherits detect_pii's quality guards
    # (octet-validated IPv4 + version-context suppression) rather than re-inlining
    # weaker regexes. It MUST run after the credential subs above: a long digit run
    # inside an API key matches the SSN shape, and masking that first would break
    # the credential into a fragment the api_key pattern no longer recognises,
    # leaking the remainder. Scrubbing credentials first leaves no digits behind.
    matches = [match for match in detect_pii(text) if match.pii_type in _EGRESS_MARKERS]
    applied_start = len(text)
    for match in sorted(matches, key=lambda item: item.start, reverse=True):
        if match.end > applied_start:
            continue  # overlaps an already-masked span; a second splice would corrupt it
        text = text[: match.start] + _EGRESS_MARKERS[PIIType(match.pii_type)] + text[match.end :]
        applied_start = match.start
    return text


def redact_paths(text: str, project_root: str = "") -> str:
    """Replace absolute project paths with ``<project>/relative/path``.

    Scans *text* for occurrences of *project_root* and replaces each with
    the ``<project>`` placeholder so that machine-specific filesystem layouts
    are not transmitted.  When *project_root* is empty, returns *text*
    unchanged.
    """
    if not project_root:
        return text
    return text.replace(project_root, "<project>")


def anonymize_installation_id(raw_id: str) -> str:
    """Double SHA-256 hash for non-reversible anonymization.

    Applies two rounds of SHA-256 hashing so that the original value
    cannot be recovered even with rainbow tables of common inputs.
    Returns the first 16 hex characters of the second hash.
    """
    first = hashlib.sha256(raw_id.encode()).hexdigest()
    return hashlib.sha256(first.encode()).hexdigest()[:16]
