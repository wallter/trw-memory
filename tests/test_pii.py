"""Tests for trw_memory.security.pii — PII detection and redaction."""

from __future__ import annotations

import pytest

from trw_memory.exceptions import MemoryError
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.pii import (
    PIIAction,
    PIIType,
    anonymize_installation_id,
    check_entry_pii,
    detect_pii,
    redact_paths,
    redact_text,
    shannon_entropy,
    strip_pii,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str = "safe content",
    detail: str = "",
    entry_id: str = "M-test",
) -> MemoryEntry:
    """Create a MemoryEntry for testing."""
    return MemoryEntry(id=entry_id, content=content, detail=detail)


# ---------------------------------------------------------------------------
# Shannon entropy tests
# ---------------------------------------------------------------------------


class TestShannonEntropy:
    """Tests for Shannon entropy calculation."""

    def test_empty_string_returns_zero(self) -> None:
        """Empty string has zero entropy."""
        assert shannon_entropy("") == 0.0

    def test_single_char_returns_zero(self) -> None:
        """A string of identical characters has zero entropy."""
        assert shannon_entropy("aaaaaaa") == 0.0

    def test_known_entropy_value(self) -> None:
        """Binary string 'ab' repeated has exactly 1.0 bit/char."""
        # 'abababab' — two chars, equal frequency => 1.0 bit/char
        result = shannon_entropy("abababab")
        assert abs(result - 1.0) < 0.001

    def test_higher_diversity_means_higher_entropy(self) -> None:
        """More diverse characters produce higher entropy."""
        low = shannon_entropy("aabbcc")
        # Use hex chars for higher diversity
        high = shannon_entropy("0123456789abcdef")
        assert high > low

    def test_random_hex_has_high_entropy(self) -> None:
        """A 32-char hex string should have entropy near 4.0 bits/char."""
        # All 16 hex chars equally distributed
        hex_str = "0123456789abcdef" * 2
        ent = shannon_entropy(hex_str)
        assert ent > 3.5


# ---------------------------------------------------------------------------
# detect_pii tests
# ---------------------------------------------------------------------------


class TestDetectPII:
    """Tests for PII regex detection."""

    def test_normal_text_no_matches(self) -> None:
        """Regular text produces no PII matches."""
        matches = detect_pii("The quick brown fox jumps over the lazy dog.")
        assert matches == []

    def test_detect_email(self) -> None:
        """Email addresses are detected."""
        matches = detect_pii("Contact us at user@example.com for help.")
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL
        assert matches[0].value == "user@example.com"
        assert matches[0].confidence == 0.95

    def test_detect_phone(self) -> None:
        """US phone numbers are detected."""
        matches = detect_pii("Call me at 555-123-4567 today.")
        assert len(matches) >= 1
        phone_matches = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phone_matches) >= 1
        assert "555" in phone_matches[0].value

    def test_detect_phone_with_parens(self) -> None:
        """Phone numbers with parentheses are detected."""
        matches = detect_pii("Call (555) 123-4567.")
        phone_matches = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phone_matches) >= 1

    def test_detect_ssn(self) -> None:
        """Social security numbers are detected."""
        matches = detect_pii("SSN is 123-45-6789.")
        ssn_matches = [m for m in matches if m.pii_type == PIIType.SSN]
        assert len(ssn_matches) >= 1
        assert "123" in ssn_matches[0].value

    def test_detect_credit_card(self) -> None:
        """Credit card numbers are detected."""
        matches = detect_pii("Card: 4111-1111-1111-1111")
        cc_matches = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cc_matches) >= 1
        assert "4111" in cc_matches[0].value

    def test_detect_credit_card_no_dashes(self) -> None:
        """Credit card numbers without dashes are detected."""
        matches = detect_pii("Card: 4111111111111111")
        cc_matches = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cc_matches) >= 1

    def test_detect_api_key(self) -> None:
        """API keys with common prefixes are detected."""
        matches = detect_pii("Use sk-abcdefghijklmnopqrstuvwxyz to authenticate.")
        key_matches = [m for m in matches if m.pii_type == PIIType.API_KEY]
        assert len(key_matches) >= 1
        assert key_matches[0].value.startswith("sk-")

    def test_detect_api_key_token_prefix(self) -> None:
        """Token-prefixed keys are detected."""
        matches = detect_pii("Use token-abcdefghijklmnopqrstuvwxyz for auth.")
        key_matches = [m for m in matches if m.pii_type == PIIType.API_KEY]
        assert len(key_matches) >= 1

    def test_detect_github_pat_classic(self) -> None:
        """Classic GitHub PATs (ghp_ + 36 chars) are detected.

        Regression: a 40-char GitHub PAT scores ~4.1 bits/char Shannon
        entropy, below the 4.5 default threshold, and has no
        "<prefix>[-_]" separator, so it matched NEITHER the generic
        API_KEY pattern NOR the high-entropy backstop — it leaked
        silently. Without the provider-specific pattern this asserts
        len() == 0.
        """
        leaked = "ghp_16C7e42F292c6912E7710c838347Ae178B4a"
        matches = detect_pii(f"token is {leaked}")
        key_matches = [m for m in matches if m.pii_type == PIIType.API_KEY]
        assert len(key_matches) == 1
        assert key_matches[0].value == leaked

    def test_detect_github_pat_fine_grained(self) -> None:
        """Fine-grained GitHub PATs (github_pat_...) are detected."""
        leaked = "github_pat_11ABCDEFG0abcdefghijkl_mnopQRSTUVwxyz0123456789ABCDEF"
        matches = detect_pii(f"export GH={leaked}")
        key_matches = [m for m in matches if m.pii_type == PIIType.API_KEY]
        assert len(key_matches) == 1
        assert leaked in key_matches[0].value

    def test_detect_aws_access_key_id(self) -> None:
        """AWS access key IDs (AKIA/ASIA + 16 base32) are detected."""
        leaked = "AKIAIOSFODNN7EXAMPLE"
        matches = detect_pii(f"aws_access_key_id={leaked}")
        key_matches = [m for m in matches if m.pii_type == PIIType.API_KEY]
        assert len(key_matches) == 1
        assert key_matches[0].value == leaked

    def test_detect_ip_address(self) -> None:
        """IPv4 addresses are detected."""
        matches = detect_pii("Server is reachable at 192.168.1.10")
        ip_matches = [m for m in matches if m.pii_type == PIIType.IP_ADDRESS]
        assert len(ip_matches) == 1

    def test_detect_file_path(self) -> None:
        """Absolute filesystem paths are detected."""
        matches = detect_pii("Read /home/alice/.ssh/config for setup")
        path_matches = [m for m in matches if m.pii_type == PIIType.FILE_PATH]
        assert len(path_matches) == 1

    def test_detect_custom_pattern(self) -> None:
        """Custom regex patterns are surfaced as custom PII."""
        matches = detect_pii("employee id EMP-12345", custom_patterns=[r"EMP-\d+"])
        custom_matches = [m for m in matches if m.pii_type == PIIType.CUSTOM]
        assert len(custom_matches) == 1

    def test_detect_high_entropy_string(self) -> None:
        """High-entropy tokens (mixed alphanumeric) are flagged."""
        # 30-char mixed-case+digits token — entropy ~4.9 bits/char
        high_entropy_token = "aB3cD9eF2gH5iJ8kL1mN4oP7qR6sT0"
        matches = detect_pii(f"Secret: {high_entropy_token}")
        he_matches = [m for m in matches if m.pii_type == PIIType.HIGH_ENTROPY]
        assert len(he_matches) >= 1
        assert he_matches[0].value == high_entropy_token

    def test_short_tokens_not_flagged_as_high_entropy(self) -> None:
        """Tokens shorter than 20 chars are not flagged as high entropy."""
        matches = detect_pii("Short token: abc123def456")
        he_matches = [m for m in matches if m.pii_type == PIIType.HIGH_ENTROPY]
        assert len(he_matches) == 0

    def test_low_entropy_long_string_not_flagged(self) -> None:
        """Long but low-entropy strings (repeated chars) are not flagged."""
        low_entropy = "a" * 30
        matches = detect_pii(f"Pattern: {low_entropy}")
        he_matches = [m for m in matches if m.pii_type == PIIType.HIGH_ENTROPY]
        assert len(he_matches) == 0

    def test_multiple_pii_types_in_one_text(self) -> None:
        """Multiple PII types in the same text are all detected."""
        text = "Email user@test.com, phone 555-123-4567, SSN 123-45-6789"
        matches = detect_pii(text)
        types_found = {m.pii_type for m in matches}
        assert PIIType.EMAIL in types_found
        assert PIIType.PHONE in types_found
        assert PIIType.SSN in types_found

    def test_match_positions_are_correct(self) -> None:
        """Match start and end positions correspond to the actual text."""
        text = "Hello user@example.com world"
        matches = detect_pii(text)
        assert len(matches) >= 1
        email_match = matches[0]
        assert text[email_match.start : email_match.end] == "user@example.com"


# ---------------------------------------------------------------------------
# redact_text tests
# ---------------------------------------------------------------------------


class TestRedactText:
    """Tests for PII redaction."""

    def test_no_matches_returns_original(self) -> None:
        """Text is unchanged when there are no matches."""
        text = "Clean text here."
        assert redact_text(text, []) == text

    def test_single_email_redacted(self) -> None:
        """A single email is replaced with a redaction marker."""
        text = "Contact user@example.com please."
        matches = detect_pii(text)
        result = redact_text(text, matches)
        assert "[REDACTED:email]" in result
        assert "user@example.com" not in result

    def test_multiple_redactions(self) -> None:
        """Multiple PII matches are all redacted."""
        text = "Email: a@b.com Phone: 555-123-4567"
        matches = detect_pii(text)
        result = redact_text(text, matches)
        assert "a@b.com" not in result
        assert "[REDACTED:" in result

    def test_redaction_preserves_surrounding_text(self) -> None:
        """Text before and after PII is preserved."""
        text = "START user@example.com END"
        matches = detect_pii(text)
        result = redact_text(text, matches)
        assert result.startswith("START ")
        assert result.endswith(" END")


# ---------------------------------------------------------------------------
# check_entry_pii tests
# ---------------------------------------------------------------------------


class TestCheckEntryPII:
    """Tests for the entry-level PII check with action handling."""

    def test_clean_entry_returns_unchanged(self) -> None:
        """Entry without PII is returned unchanged with empty matches."""
        entry = _make_entry("This is safe content.")
        result_entry, matches = check_entry_pii(entry)
        assert result_entry.content == "This is safe content."
        assert matches == []

    def test_warn_action_returns_entry_unchanged(self) -> None:
        """WARN action returns the entry as-is but reports matches."""
        entry = _make_entry("Contact user@example.com")
        result_entry, matches = check_entry_pii(entry, action=PIIAction.WARN)
        assert result_entry.content == "Contact user@example.com"
        assert len(matches) >= 1

    def test_redact_action_masks_pii(self) -> None:
        """REDACT action masks PII in content field."""
        entry = _make_entry("Contact user@example.com")
        result_entry, matches = check_entry_pii(entry, action=PIIAction.REDACT)
        assert "[REDACTED:email]" in result_entry.content
        assert "user@example.com" not in result_entry.content
        assert len(matches) >= 1

    def test_redact_action_masks_pii_in_detail(self) -> None:
        """REDACT action also masks PII in the detail field."""
        entry = _make_entry(
            content="Safe content",
            detail="Detail with user@example.com inside",
        )
        result_entry, matches = check_entry_pii(entry, action=PIIAction.REDACT)
        assert "user@example.com" not in result_entry.detail
        assert "[REDACTED:email]" in result_entry.detail

    def test_block_action_raises_memory_error(self) -> None:
        """BLOCK action raises MemoryError when PII is found."""
        entry = _make_entry("Contact user@example.com")
        with pytest.raises(MemoryError, match="PII detected"):
            check_entry_pii(entry, action=PIIAction.BLOCK)

    def test_block_action_with_clean_entry_succeeds(self) -> None:
        """BLOCK action does not raise when entry has no PII."""
        entry = _make_entry("Clean content, no PII here.")
        result_entry, matches = check_entry_pii(entry, action=PIIAction.BLOCK)
        assert matches == []
        assert result_entry.content == "Clean content, no PII here."

    def test_custom_entropy_threshold(self) -> None:
        """Custom entropy threshold is respected."""
        # Very low threshold should flag almost any diverse token
        entry = _make_entry("Token: abcdefghijklmnopqrst")
        _, matches_low = check_entry_pii(entry, action=PIIAction.WARN, entropy_threshold=2.0)
        _, matches_high = check_entry_pii(entry, action=PIIAction.WARN, entropy_threshold=6.0)
        # Lower threshold should produce more or equal matches
        he_low = [m for m in matches_low if m.pii_type == PIIType.HIGH_ENTROPY]
        he_high = [m for m in matches_high if m.pii_type == PIIType.HIGH_ENTROPY]
        assert len(he_low) >= len(he_high)


# ---------------------------------------------------------------------------
# strip_pii tests
# ---------------------------------------------------------------------------


class TestStripPII:
    """Tests for the strip_pii anonymization helper."""

    def test_email_replaced_with_placeholder(self) -> None:
        """Email addresses are replaced with <email>."""
        result = strip_pii("Contact admin@example.com for help.")
        assert "<email>" in result
        assert "admin@example.com" not in result

    def test_api_key_replaced_with_placeholder(self) -> None:
        """API key patterns are replaced with <api_key>."""
        result = strip_pii("Use sk-abcdefghijklmnopqrstuvwxyz to auth.")
        assert "<api_key>" in result
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in result

    def test_token_prefix_replaced(self) -> None:
        """token- prefixed keys are replaced."""
        result = strip_pii("token-abcdefghijklmnopqrstuvwxyz")
        assert "<api_key>" in result

    def test_github_pat_replaced(self) -> None:
        """GitHub PATs are scrubbed from telemetry text.

        Regression: strip_pii's API-key pattern required a
        "<prefix>[-_]" separator, so a ghp_ PAT survived telemetry
        anonymization. Without the provider-specific sub fails on the
        leaked-value assertion below.
        """
        leaked = "ghp_16C7e42F292c6912E7710c838347Ae178B4a"
        result = strip_pii(f"token is {leaked}")
        assert "<api_key>" in result
        assert leaked not in result

    def test_aws_access_key_replaced(self) -> None:
        """AWS access key IDs are scrubbed from telemetry text."""
        leaked = "AKIAIOSFODNN7EXAMPLE"
        result = strip_pii(f"key={leaked}")
        assert leaked not in result

    def test_clean_text_unchanged(self) -> None:
        """Text without PII is returned unchanged."""
        text = "This is a normal sentence with no secrets."
        assert strip_pii(text) == text

    def test_multiple_emails_all_replaced(self) -> None:
        """All email occurrences are replaced."""
        result = strip_pii("a@b.com and c@d.org")
        assert "a@b.com" not in result
        assert "c@d.org" not in result
        assert result.count("<email>") == 2


# ---------------------------------------------------------------------------
# redact_paths tests
# ---------------------------------------------------------------------------


class TestRedactPaths:
    """Tests for the redact_paths anonymization helper."""

    def test_project_root_replaced(self) -> None:
        """Occurrences of project_root are replaced with <project>."""
        result = redact_paths("/home/user/myproject/src/foo.py", "/home/user/myproject")
        assert "<project>" in result
        assert "/home/user/myproject" not in result

    def test_empty_root_returns_text_unchanged(self) -> None:
        """Empty project_root leaves text unchanged."""
        text = "/some/absolute/path/file.py"
        assert redact_paths(text, "") == text

    def test_default_root_returns_text_unchanged(self) -> None:
        """Default (empty) project_root leaves text unchanged."""
        text = "/some/path/file.py"
        assert redact_paths(text) == text

    def test_no_match_returns_text_unchanged(self) -> None:
        """Text not containing the root is returned unchanged."""
        text = "No path here at all."
        result = redact_paths(text, "/home/user/project")
        assert result == text

    def test_multiple_occurrences_all_replaced(self) -> None:
        """All occurrences of the root are replaced."""
        root = "/home/user/proj"
        text = f"{root}/a.py and {root}/b.py"
        result = redact_paths(text, root)
        assert root not in result
        assert result.count("<project>") == 2


# ---------------------------------------------------------------------------
# anonymize_installation_id tests
# ---------------------------------------------------------------------------


class TestAnonymizeInstallationId:
    """Tests for the double-SHA-256 anonymization helper."""

    def test_returns_16_hex_chars(self) -> None:
        """Output is exactly 16 hexadecimal characters."""
        result = anonymize_installation_id("my-installation-id")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_input_same_output(self) -> None:
        """Deterministic: same input always yields same output."""
        raw = "stable-id-12345"
        assert anonymize_installation_id(raw) == anonymize_installation_id(raw)

    def test_different_inputs_different_outputs(self) -> None:
        """Different inputs produce different outputs (no trivial collision)."""
        assert anonymize_installation_id("id-a") != anonymize_installation_id("id-b")

    def test_output_does_not_contain_input(self) -> None:
        """The raw ID is not present in the anonymized output."""
        raw = "supersecret-installation-xyz"
        result = anonymize_installation_id(raw)
        assert raw not in result

    def test_empty_string_produces_valid_hash(self) -> None:
        """Empty string input still produces a valid 16-char hex output."""
        result = anonymize_installation_id("")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)
