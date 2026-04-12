"""Tests for trw-memory package metadata and exports."""

from __future__ import annotations


def test_package_importable() -> None:
    """Package is importable."""
    import trw_memory

    assert hasattr(trw_memory, "__version__")
    assert hasattr(trw_memory, "__all__")


def test_version_accessible() -> None:
    """Version string is accessible and well-formed."""
    from trw_memory import __version__

    assert isinstance(__version__, str)
    # Verify semantic version format: major.minor.patch
    parts = __version__.split(".")
    assert len(parts) >= 2, f"Version {__version__} does not look like semver"
    assert all(part.isdigit() for part in parts[:2])


def test_core_exports_exist() -> None:
    """All core exports from __all__ are importable."""
    from trw_memory import (
        ConfigError,
        EncryptionUnavailableError,
        KeyRotationError,
        MemoryConfig,
        MemoryEntry,
        MemoryError,
        MemoryEvent,
        MemoryEventType,
        MemoryIndex,
        MasterKeyNotFoundError,
        MemoryStatus,
        StorageError,
        namespace_to_path,
        validate_namespace,
    )

    assert issubclass(ConfigError, Exception)
    assert issubclass(EncryptionUnavailableError, Exception)
    assert issubclass(KeyRotationError, Exception)
    assert issubclass(MemoryConfig, object)
    assert issubclass(MemoryEntry, object)
    assert issubclass(MemoryError, Exception)
    assert issubclass(MemoryEvent, object)
    assert issubclass(MemoryEventType, object)
    assert issubclass(MemoryIndex, object)
    assert issubclass(MasterKeyNotFoundError, Exception)
    assert issubclass(MemoryStatus, object)
    assert issubclass(StorageError, Exception)
    assert callable(namespace_to_path)
    assert callable(validate_namespace)


def test_all_exports_valid() -> None:
    """Every name in __all__ actually exists in the module."""
    import trw_memory

    for name in trw_memory.__all__:
        assert hasattr(trw_memory, name), f"{name} listed in __all__ but not found"


def test_all_exports_complete() -> None:
    """Public names in __all__ match the declared set."""
    import trw_memory

    expected = {
        "AuthorizationError",
        "ConfigError",
        "DimensionMismatchError",
        "LocalOnlyViolationError",
        "EncryptionUnavailableError",
        "KeyRotationError",
        "MasterKeyNotFoundError",
        "MemoryClient",
        "MemoryConfig",
        "MemoryConnectionError",
        "MemoryEntry",
        "MemoryError",
        "MemoryEvent",
        "MemoryEventType",
        "MemoryIndex",
        "MemoryNotFoundError",
        "MemoryStatus",
        "StorageError",
        "ToolAlreadyRegisteredError",
        "__version__",
        "namespace_to_path",
        "validate_namespace",
    }
    assert set(trw_memory.__all__) == expected


def test_exceptions_inherit_properly() -> None:
    """Custom exceptions have correct hierarchy."""
    from trw_memory import (
        AuthorizationError,
        ConfigError,
        DimensionMismatchError,
        EncryptionUnavailableError,
        KeyRotationError,
        LocalOnlyViolationError,
        MasterKeyNotFoundError,
        MemoryConnectionError,
        MemoryError,
        MemoryNotFoundError,
        StorageError,
        ToolAlreadyRegisteredError,
    )

    assert issubclass(MemoryError, Exception)
    assert issubclass(StorageError, MemoryError)
    assert issubclass(ConfigError, MemoryError)
    assert issubclass(MemoryConnectionError, MemoryError)
    assert issubclass(MemoryNotFoundError, MemoryError)
    assert issubclass(ToolAlreadyRegisteredError, MemoryError)
    assert issubclass(AuthorizationError, MemoryError)
    assert issubclass(DimensionMismatchError, MemoryError)
    assert issubclass(LocalOnlyViolationError, MemoryError)
    assert issubclass(EncryptionUnavailableError, MemoryError)
    assert issubclass(KeyRotationError, MemoryError)
    assert issubclass(MasterKeyNotFoundError, MemoryError)


def test_memory_status_is_enum() -> None:
    """MemoryStatus is an enum with expected values."""
    from trw_memory import MemoryStatus

    assert hasattr(MemoryStatus, "ACTIVE")
    assert hasattr(MemoryStatus, "RESOLVED")
    assert hasattr(MemoryStatus, "OBSOLETE")
