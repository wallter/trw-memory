"""PII redaction helpers for the runtime store path.

Belongs to ``security/runtime.py``. Re-exported there for back-compat.

5 helpers + 3 constants covering PII redaction + code-snippet flagging:

- ``apply_runtime_pii_policy`` — block on API_KEY, redact emails/IPs,
  hash file paths, set high-entropy metadata flag.
- ``replace_pii`` — apply per-match redaction in reverse-end order.
- ``hash_path_components`` — sha256-hash each component of a file
  path (Windows + POSIX).
- ``redaction_marker`` — return ``<email>``/``<ip>``/etc. token.
- ``flag_code_snippet`` — authoritatively set
  ``SYSTEM_CODE_FLAG_KEY`` metadata + ``code_snippet_flagged`` tag.

Constants:

- ``BLOCKING_PII_TYPES`` — frozenset of types that raise
  ``PIIBlockError``.
- ``REDACTED_PII_TYPES`` — frozenset of types that get a marker
  replacement.
- ``CODE_SNIPPET_PATTERNS`` — compiled regex tuple for code-snippet
  heuristic.

Extracted as PRD-DIST-245 Phase 3 batch 99.
"""

from __future__ import annotations

import hashlib
import re

import structlog

from trw_memory.exceptions import PIIBlockError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.pii import PIIMatch, PIIType, detect_pii

logger = structlog.get_logger(__name__)

BLOCKING_PII_TYPES = frozenset({PIIType.API_KEY})
REDACTED_PII_TYPES = frozenset(
    {
        PIIType.EMAIL,
        PIIType.IP_ADDRESS,
        PIIType.CUSTOM,
        # Security audit 2026-06-09 (v0.9.2): the v0.9.1 tag-scan added detection
        # of SSN/PHONE/CREDIT_CARD in tags, but redaction was never wired up — so
        # these were detected-but-stored-verbatim (matched neither BLOCKING_PII_TYPES
        # nor REDACTED_PII_TYPES, so replace_pii() fell through its else: continue).
        # Policy is redact-not-block (parity with EMAIL/IP): preserve the entry while
        # masking the value rather than rejecting the whole store like a credential.
        PIIType.PHONE,
        PIIType.SSN,
        PIIType.CREDIT_CARD,
        # Security audit 2026-06-17: HIGH_ENTROPY tokens (entropy-backstop secrets
        # without a recognized API_KEY prefix) were detected — metadata flag set —
        # but never redacted: replace_pii() fell through its else: continue, storing
        # the raw secret verbatim in content/detail/tags. The public check_entry_pii
        # path already redacts HIGH_ENTROPY generically via redact_text; this aligns
        # the runtime path with it. Redact-not-block (parity with EMAIL/IP).
        PIIType.HIGH_ENTROPY,
    }
)
CODE_SNIPPET_PATTERNS = (
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bclass\s+\w+"),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bfunction\s+\w+\s*\("),
)


def apply_runtime_pii_policy(
    entry: MemoryEntry,
    config: MemoryConfig,
) -> tuple[MemoryEntry, list[PIIMatch]]:
    if not config.pii_enabled:
        return entry, []

    content_matches = detect_pii(
        entry.content,
        entropy_threshold=config.pii_entropy_threshold,
        custom_patterns=config.pii_custom_patterns,
    )
    detail_matches = detect_pii(
        entry.detail,
        entropy_threshold=config.pii_entropy_threshold,
        custom_patterns=config.pii_custom_patterns,
    )
    # Security audit 2026-06-09: scan tags too. A credential (API key) or other
    # PII placed in a tag previously bypassed BOTH the block gate and redaction
    # while still being surfaced at recall time.
    tag_matches_by_index: list[list[PIIMatch]] = [
        detect_pii(
            tag,
            entropy_threshold=config.pii_entropy_threshold,
            custom_patterns=config.pii_custom_patterns,
        )
        for tag in entry.tags
    ]
    tag_matches = [match for matches in tag_matches_by_index for match in matches]
    # Security audit 2026-07-17 (SEC-001 release-blocker): evidence[] and each
    # Assertion.last_evidence are publicly reachable via memory_store and were
    # persisted verbatim — a credential/email/SSN placed there bypassed BOTH the
    # block gate and redaction while still surfacing at recall. Scan + redact them
    # with the same policy as content/detail/tags.
    evidence_matches_by_index: list[list[PIIMatch]] = [
        detect_pii(
            item,
            entropy_threshold=config.pii_entropy_threshold,
            custom_patterns=config.pii_custom_patterns,
        )
        for item in entry.evidence
    ]
    evidence_matches = [match for matches in evidence_matches_by_index for match in matches]
    assertion_matches_by_index: list[list[PIIMatch]] = [
        detect_pii(
            assertion.last_evidence,
            entropy_threshold=config.pii_entropy_threshold,
            custom_patterns=config.pii_custom_patterns,
        )
        for assertion in entry.assertions
    ]
    assertion_matches = [match for matches in assertion_matches_by_index for match in matches]
    all_matches = content_matches + detail_matches + tag_matches + evidence_matches + assertion_matches
    if not all_matches:
        return entry, []

    blocking = [match for match in all_matches if match.pii_type in BLOCKING_PII_TYPES]
    if blocking:
        detected_type = str(blocking[0].pii_type)
        logger.warning(
            "memory_store_pii_blocked",
            detected_type=detected_type,
            namespace=entry.namespace,
            entry_id=entry.id,
        )
        raise PIIBlockError(
            f"memory entry blocked by PII policy: {detected_type}",
            detected_type=detected_type,
        )

    new_content = replace_pii(entry.content, content_matches)
    new_detail = replace_pii(entry.detail, detail_matches)
    new_tags = [replace_pii(tag, matches) for tag, matches in zip(entry.tags, tag_matches_by_index, strict=True)]
    new_evidence = [
        replace_pii(item, matches) for item, matches in zip(entry.evidence, evidence_matches_by_index, strict=True)
    ]
    # Assertion.last_evidence is a nested-model string field; model_copy does not
    # re-run validation, so redacting it in place is safe (no risk of tripping the
    # grep-pattern validator). Only touch assertions that actually matched.
    new_assertions = [
        (
            assertion.model_copy(update={"last_evidence": replace_pii(assertion.last_evidence, matches)})
            if matches
            else assertion
        )
        for assertion, matches in zip(entry.assertions, assertion_matches_by_index, strict=True)
    ]
    metadata = dict(entry.metadata)
    metadata["pii_types"] = ",".join(sorted({match.pii_type for match in all_matches}))
    if any(match.pii_type == PIIType.HIGH_ENTROPY for match in all_matches):
        metadata["contains_high_entropy_token"] = "true"  # noqa: S105 — flag value, not a credential
    return (
        entry.model_copy(
            update={
                "content": new_content,
                "detail": new_detail,
                "tags": new_tags,
                "evidence": new_evidence,
                "assertions": new_assertions,
                "metadata": metadata,
            }
        ),
        all_matches,
    )


def replace_pii(text: str, matches: list[PIIMatch]) -> str:
    if not matches:
        return text
    result = text
    for match in sorted(matches, key=lambda item: item.start, reverse=True):
        if match.pii_type == PIIType.FILE_PATH:
            replacement = hash_path_components(match.value)
        elif match.pii_type in REDACTED_PII_TYPES:
            replacement = redaction_marker(match.pii_type)
        else:
            continue
        result = result[: match.start] + replacement + result[match.end :]
    return result


def hash_path_components(path_value: str) -> str:
    is_windows = ":\\" in path_value
    separator = "\\" if is_windows else "/"
    components = [component for component in re.split(r"[\\/]+", path_value) if component]
    hashed = [hashlib.sha256(component.encode("utf-8")).hexdigest()[:8] for component in components]
    prefix = "C:\\" if is_windows and path_value[:2].isalpha() else ("/" if path_value.startswith("/") else "")
    return prefix + separator.join(hashed)


def redaction_marker(pii_type: PIIType) -> str:
    if pii_type == PIIType.EMAIL:
        return "<email>"
    if pii_type == PIIType.IP_ADDRESS:
        return "<ip>"
    if pii_type == PIIType.CUSTOM:
        return "<custom_pii>"
    if pii_type == PIIType.PHONE:
        return "<phone>"
    if pii_type == PIIType.SSN:
        return "<ssn>"
    if pii_type == PIIType.CREDIT_CARD:
        return "<credit_card>"
    if pii_type == PIIType.HIGH_ENTROPY:
        return "<high_entropy_secret>"  # redaction marker, not a credential
    return f"<{pii_type}>"


def flag_code_snippet(entry: MemoryEntry) -> MemoryEntry:
    """Authoritatively set the system code-flag metadata key.

    Strips any caller-provided value of ``SYSTEM_CODE_FLAG_KEY`` and sets
    it to "true" iff the combined content matches a code-snippet pattern.
    This guarantees callers cannot pre-seed the bypass flag — see security
    audit 2026-04-18 H2. The descriptive ``"code_snippet_flagged"`` tag is
    still appended for backward-compat visibility in UI listings.
    """
    from trw_memory.security.poisoning import SYSTEM_CODE_FLAG_KEY

    combined = f"{entry.content}\n{entry.detail}"
    is_code = any(pattern.search(combined) for pattern in CODE_SNIPPET_PATTERNS)

    metadata = {k: v for k, v in entry.metadata.items() if k != SYSTEM_CODE_FLAG_KEY}
    tags = list(entry.tags)
    updates: dict[str, object] = {}

    if is_code:
        metadata[SYSTEM_CODE_FLAG_KEY] = "true"
        if "code_snippet_flagged" not in tags:
            tags.append("code_snippet_flagged")

    if metadata != entry.metadata:
        updates["metadata"] = metadata
    if tags != list(entry.tags):
        updates["tags"] = tags

    if updates:
        return entry.model_copy(update=updates)
    return entry
