"""Exception hierarchy for trw-memory."""

from __future__ import annotations


class MemoryError(Exception):
    """Base exception for all trw-memory errors.

    Args:
        message: Human-readable error description.
        path: Optional file path involved in the error.
    """

    def __init__(self, message: str, *, path: str = "") -> None:
        super().__init__(message)
        self.path = path


class StorageError(MemoryError):
    """Raised when a storage operation fails (read, write, lock)."""


class ConfigError(MemoryError):
    """Raised when configuration is invalid or cannot be loaded."""


class MemoryConnectionError(MemoryError):
    """Raised when no connection mode is available."""


class MemoryNotFoundError(MemoryError):
    """Raised when a memory entry is not found."""


class ToolAlreadyRegisteredError(MemoryError):
    """Raised when register_tools() is called twice."""


class AuthorizationError(MemoryError, PermissionError):
    """Raised when a permission check fails."""


class DimensionMismatchError(MemoryError):
    """Raised when vectors have incompatible dimensions."""


class LocalOnlyViolationError(MemoryError):
    """Raised when a network-capable operation is attempted in local-only mode."""


class MasterKeyNotFoundError(MemoryError):
    """Raised when no usable master key exists in any configured source."""


class EncryptionUnavailableError(MemoryError):
    """Raised when an encryption-required runtime dependency is unavailable."""


class KeyRotationError(MemoryError):
    """Raised when key rotation fails or cannot be safely completed."""


class SchemaValidationError(MemoryError):
    """Raised when a memory entry fails the write-time schema/policy contract."""

    def __init__(self, message: str, *, path: str = "", failed_fields: list[str] | None = None) -> None:
        super().__init__(message, path=path)
        self.failed_fields = failed_fields or []


class PIIBlockError(MemoryError):
    """Raised when a memory entry is blocked by the configured PII policy."""

    def __init__(self, message: str, *, path: str = "", detected_type: str = "") -> None:
        super().__init__(message, path=path)
        self.detected_type = detected_type


class PoisoningError(MemoryError):
    """Raised when content matches write-time poisoning or injection defenses."""

    def __init__(self, message: str, *, path: str = "", reason: str = "") -> None:
        super().__init__(message, path=path)
        self.reason = reason


class RateLimitError(MemoryError):
    """Raised when a caller exceeds the configured memory write rate limit."""

    def __init__(self, message: str, *, path: str = "", retry_after: float = 0.0) -> None:
        super().__init__(message, path=path)
        self.retry_after = retry_after


class Utf8ValidationError(SchemaValidationError):
    """Raised when a string field fails write-time UTF-8 validation.

    Carries the list of offending field names via failed_fields. Use this
    when a write would persist bytes that can't round-trip through
    str.encode('utf-8', errors='strict') — lone surrogates, invalid
    continuation bytes, etc.
    """


class StaleConnectionError(StorageError):
    """Raised when a connection was detected stale and could not be reopened.

    Normally SQLiteBackend._ensure_connection_fresh transparently reopens.
    This exception surfaces only when the reopen attempt itself fails.
    """


class CorruptDatabaseUnsalvageableError(StorageError):
    """Raised when a corrupt DB cannot be salvaged and strict policy refuses the empty fallback.

    Carries ``backup_path`` as a user-actionable breadcrumb so the forensic evidence
    is visible at the call site. Raised by :meth:`SQLiteBackend.recover_db` only when
    the primary SELECT salvage AND the ``sqlite3 .recover`` CLI fallback both yield
    zero rows from a non-empty ``.corrupt.bak`` file under the ``strict`` recovery
    policy (PRD-CORE-138).
    """

    def __init__(self, message: str, *, backup_path: str = "") -> None:
        super().__init__(f"{message} (backup preserved at: {backup_path})", path=backup_path)
        self.backup_path = backup_path


class SecurityDependencyError(MemoryError):
    """Base class for fail-loud SEC-001 dependency failures."""


class ScorerUnavailableError(SecurityDependencyError):
    """Raised when trust-scoring cannot be loaded or executed."""


class QuarantineUnreachableError(SecurityDependencyError):
    """Raised when the quarantine database cannot be reached."""


class CanaryFixturesMissingError(SecurityDependencyError):
    """Raised when canary fixtures cannot be resolved at startup/runtime."""


class ProvenanceKeyUnavailableError(SecurityDependencyError):
    """Raised when provenance signing is required but no signing key is available."""


class SecurityTelemetryUnavailableError(SecurityDependencyError):
    """Raised when required SEC-001 telemetry cannot be appended."""


class SecurityDefaultUnresolvableError(SecurityDependencyError):
    """Raised when a SEC-001 default path or fixture cannot resolve at boot."""


class CanaryTamperError(SecurityDependencyError):
    """Raised when a pinned canary hash drifts from the stored row."""
