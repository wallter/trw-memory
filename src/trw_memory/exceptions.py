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
