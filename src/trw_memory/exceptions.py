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


class DimensionMismatchError(ValueError):
    """Raised when vectors have incompatible dimensions."""
