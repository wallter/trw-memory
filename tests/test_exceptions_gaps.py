"""Wave 14: coverage gap-fill for exceptions.py (lines 67-68, 75-76, 83-84, 91-92, 124-125)."""
from __future__ import annotations

from trw_memory.exceptions import (
    CorruptDatabaseUnsalvageableError,
    PIIBlockError,
    PoisoningError,
    RateLimitError,
    SchemaValidationError,
)


class TestSchemaValidationErrorInit:
    def test_failed_fields_stored(self) -> None:
        """SchemaValidationError.__init__ stores failed_fields (lines 67-68)."""
        exc = SchemaValidationError("field error", failed_fields=["content", "tags"])
        assert exc.failed_fields == ["content", "tags"]
        assert exc.path == ""

    def test_failed_fields_defaults_to_empty_list(self) -> None:
        """SchemaValidationError with no failed_fields defaults to []."""
        exc = SchemaValidationError("error")
        assert exc.failed_fields == []


class TestPIIBlockErrorInit:
    def test_detected_type_stored(self) -> None:
        """PIIBlockError.__init__ stores detected_type (lines 75-76)."""
        exc = PIIBlockError("pii blocked", detected_type="EMAIL")
        assert exc.detected_type == "EMAIL"
        assert exc.path == ""

    def test_detected_type_defaults_to_empty_string(self) -> None:
        """PIIBlockError with no detected_type defaults to empty string."""
        exc = PIIBlockError("blocked")
        assert exc.detected_type == ""


class TestPoisoningErrorInit:
    def test_reason_stored(self) -> None:
        """PoisoningError.__init__ stores reason (lines 83-84)."""
        exc = PoisoningError("injection detected", reason="prompt injection pattern")
        assert exc.reason == "prompt injection pattern"

    def test_reason_defaults_to_empty_string(self) -> None:
        """PoisoningError with no reason defaults to empty string."""
        exc = PoisoningError("blocked")
        assert exc.reason == ""


class TestRateLimitErrorInit:
    def test_retry_after_stored(self) -> None:
        """RateLimitError.__init__ stores retry_after (lines 91-92)."""
        exc = RateLimitError("rate limited", retry_after=30.0)
        assert exc.retry_after == 30.0

    def test_retry_after_defaults_to_zero(self) -> None:
        """RateLimitError with no retry_after defaults to 0.0."""
        exc = RateLimitError("limited")
        assert exc.retry_after == 0.0


class TestCorruptDatabaseUnsalvageableErrorInit:
    def test_backup_path_stored_and_embedded_in_message(self) -> None:
        """CorruptDatabaseUnsalvageableError.__init__ stores backup_path (lines 124-125)."""
        exc = CorruptDatabaseUnsalvageableError("corrupt DB", backup_path="/tmp/test.db.bak")
        assert exc.backup_path == "/tmp/test.db.bak"
        assert "/tmp/test.db.bak" in str(exc)

    def test_backup_path_defaults_to_empty_string(self) -> None:
        """CorruptDatabaseUnsalvageableError with no backup_path defaults to empty string."""
        exc = CorruptDatabaseUnsalvageableError("corrupt")
        assert exc.backup_path == ""
