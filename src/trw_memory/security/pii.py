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

import structlog
from pydantic import BaseModel, ConfigDict, Field

from trw_memory.exceptions import MemoryError
from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger(__name__)


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
            r"\b(?:sk|pk|api|key|token|secret)[-_][a-zA-Z0-9]{20,}\b",
            re.IGNORECASE,
        ),
        0.9,
    ),
    # Provider-specific secret shapes that lack a "<prefix>[-_]" separator and
    # fall below the Shannon-entropy backstop (e.g. a 40-char GitHub PAT scores
    # ~4.1 bits/char, under the 4.5 default). These leaked silently before — a
    # token in `content`/`detail` matched neither the generic API_KEY pattern
    # nor the high-entropy path. Patterns are anchored + bounded (no nested
    # quantifiers) so they stay ReDoS-free.
    (
        PIIType.API_KEY,
        re.compile(
            # GitHub: ghp_/gho_/ghu_/ghs_/ghr_ + 36 base62, or github_pat_ + long body
            r"\b(?:gh[posru]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})\b"
            # AWS access key IDs: AKIA/ASIA + 16 uppercase base32
            r"|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        ),
        0.95,
    ),
    (
        PIIType.IP_ADDRESS,
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        0.85,
    ),
    (
        PIIType.FILE_PATH,
        re.compile(r"(?:[A-Za-z]:\\[^\s]+|/(?:[^/\s]+/)+[^/\s]+)"),
        0.8,
    ),
]

# Minimum token length and entropy threshold for high-entropy detection
_MIN_ENTROPY_TOKEN_LEN = 20
_DEFAULT_ENTROPY_THRESHOLD = 4.5


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
        matches.extend(
            PIIMatch(
                pii_type=pii_type,
                value=m.group(),
                start=m.start(),
                end=m.end(),
                confidence=confidence,
            )
            for m in pattern.finditer(text)
        )

    for raw_pattern in custom_patterns or []:
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
    """Check ``content`` and ``detail`` fields for PII and apply *action*.

    Args:
        entry: The memory entry to check.
        action: What to do when PII is found: block, redact, or warn.
        entropy_threshold: Minimum Shannon entropy for high-entropy detection.

    Returns:
        A tuple of ``(possibly_modified_entry, all_matches)``.

    Raises:
        MemoryError: If *action* is ``BLOCK`` and PII is detected.
    """
    # Scan both content and detail fields
    content_matches = detect_pii(entry.content, entropy_threshold)
    detail_matches = detect_pii(entry.detail, entropy_threshold)

    # Adjust offsets for detail matches to note they come from the detail field
    all_matches = content_matches + detail_matches

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
        updated = entry.model_copy(
            update={"content": new_content, "detail": new_detail},
        )
        return (updated, all_matches)

    # PIIAction.WARN — return entry unchanged with matches for logging
    return (entry, all_matches)


# ---------------------------------------------------------------------------
# Anonymization helpers (telemetry-safe, non-reversible)
# ---------------------------------------------------------------------------


def strip_pii(text: str) -> str:
    """Remove email addresses and API key patterns from text.

    Replaces recognised PII patterns with safe placeholders:

    * Email addresses -> ``<email>``
    * Common API key / token patterns -> ``<api_key>``
    """
    # Email addresses
    text = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "<email>",
        text,
    )
    # API key / token patterns (prefix followed by 20+ alphanumeric chars)
    text = re.sub(
        r"(sk|pk|api|key|token)[-_][a-zA-Z0-9]{20,}",
        "<api_key>",
        text,
        flags=re.IGNORECASE,
    )
    # Provider-specific secret shapes without a "<prefix>[-_]" separator
    # (GitHub PATs, AWS access key IDs) — see _PII_PATTERNS above.
    text = re.sub(
        r"\b(?:gh[posru]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})\b"
        r"|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        "<api_key>",
        text,
    )
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
