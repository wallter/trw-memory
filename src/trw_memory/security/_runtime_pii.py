"""PII policy for the runtime store path.

Belongs to ``security/runtime.py``. Re-exported there for back-compat.

3 helpers + 3 constants covering the store-time PII gate + code-snippet
flagging:

- ``apply_runtime_pii_policy`` — block on API_KEY, record detection
  metadata, mask operator-configured custom patterns.
- ``replace_pii`` — mask operator-configured CUSTOM matches only.
- ``flag_code_snippet`` — authoritatively set
  ``SYSTEM_CODE_FLAG_KEY`` metadata + ``code_snippet_flagged`` tag.

Constants:

- ``BLOCKING_PII_TYPES`` — frozenset of types that raise
  ``PIIBlockError``.
- ``REDACTED_PII_TYPES`` — frozenset of types that get a marker
  replacement (operator-configured CUSTOM patterns only).
- ``CODE_SNIPPET_PATTERNS`` — compiled regex tuple for code-snippet
  heuristic.

Extracted as PRD-DIST-245 Phase 3 batch 99.
"""

from __future__ import annotations

import re

import structlog

from trw_memory.exceptions import PIIBlockError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.pii import PIIMatch, PIIType, detect_pii

logger = structlog.get_logger(__name__)

BLOCKING_PII_TYPES = frozenset({PIIType.API_KEY})

# ---------------------------------------------------------------------------
# Write-path posture (2026-07-25): DETECT + BLOCK, do not destroy
# ---------------------------------------------------------------------------
# This path used to mutate stored content for EMAIL, IP_ADDRESS, PHONE, SSN,
# CREDIT_CARD, FILE_PATH and HIGH_ENTROPY. It ran BEFORE persistence, so the
# original text never reached disk and the loss was irreversible.
#
# Two facts retire that behaviour:
#
# 1. It was redundant against its own threat model. The only boundary where a
#    memory leaves the machine is the publish path, and that path ALREADY
#    sanitizes independently — ``sync/_remote_publish._anonymize_entry`` calls
#    ``redact_paths(strip_pii(...))`` on content and detail. Write-path
#    redaction protected nothing the egress path did not already protect; its
#    only unique effect was destroying the user's own engineering knowledge on
#    the user's own machine, next to the source tree it describes.
#
# 2. The detectors are 8 regexes with no NER, and their precision does not
#    justify irreversible mutation of local text. Measured on this project's
#    corpus: SSN matches any 9 consecutive digits ("build 123456789"),
#    CREDIT_CARD any 16, IP_ADDRESS fires on version strings, PHONE is US-only,
#    FILE_PATH did not redact but sha256-hashed every path component, and the
#    HIGH_ENTROPY backstop had a measured true-positive rate of ZERO over 832
#    flagged tokens (92 remain after the case-uniformity shape guard in
#    e936f79d7b).
#
# What is deliberately UNCHANGED:
#   * API_KEY still BLOCKS the store (``BLOCKING_PII_TYPES`` above). Refusing to
#     persist a recognized credential is high-precision and non-destructive — it
#     errors loudly instead of silently mutating.
#   * Detection still runs over every writable field; ``pii_types`` and
#     ``contains_high_entropy_token`` metadata are still set. Observability is
#     free and non-destructive.
#   * The egress path in ``sync/`` is untouched and is where sanitization
#     belongs — it is reversible there, because the local copy keeps the truth.
#
# CUSTOM stays because it is not one of our heuristics: it is the operator's own
# ``pii_custom_patterns`` (default empty), the only way to opt in to local
# masking. Removing it would delete operator control, not restore user data.
REDACTED_PII_TYPES = frozenset({PIIType.CUSTOM})

CUSTOM_PII_MARKER = "<custom_pii>"

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
    # Security audit 2026-06-09: scan tags too. A credential (API key) placed in
    # a tag previously bypassed the block gate entirely.
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
    # Assertion.last_evidence are publicly reachable via memory_store, so a
    # credential smuggled there bypassed the block gate. Scan them with the same
    # policy as content/detail/tags so API_KEY blocks wherever it is hidden.
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
    # re-run validation, so masking it in place is safe (no risk of tripping the
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
    """Mask operator-configured CUSTOM matches; leave every other type verbatim.

    Built-in detector types are intentionally NOT masked here — see the
    write-path posture note above ``REDACTED_PII_TYPES``. A caller who wants
    local masking configures ``pii_custom_patterns`` and gets exactly the
    pattern they wrote, with no heuristic guessing at what looks like a secret.
    """
    custom = [match for match in matches if match.pii_type in REDACTED_PII_TYPES]
    if not custom:
        return text
    result = text
    for match in sorted(custom, key=lambda item: item.start, reverse=True):
        result = result[: match.start] + CUSTOM_PII_MARKER + result[match.end :]
    return result


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
