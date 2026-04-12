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


class PIIBlockError(MemoryError):
    """Raised when a memory entry is blocked by the configured PII policy."""


class PoisoningError(MemoryError):
    """Raised when content matches write-time poisoning or injection defenses."""


class RateLimitError(MemoryError):
    """Raised when a caller exceeds the configured memory write rate limit."""
