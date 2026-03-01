"""Tests for trw-memory package metadata and exports."""

from __future__ import annotations


def test_package_importable() -> None:
    """Package is importable."""
    import trw_memory

    assert trw_memory is not None


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
        MemoryConfig,
        MemoryEntry,
        MemoryError,
        MemoryEvent,
        MemoryEventType,
        MemoryIndex,
        MemoryStatus,
        StorageError,
        namespace_to_path,
        validate_namespace,
    )

    assert ConfigError is not None
    assert MemoryConfig is not None
    assert MemoryEntry is not None
    assert MemoryError is not None
    assert MemoryEvent is not None
    assert MemoryEventType is not None
    assert MemoryIndex is not None
    assert MemoryStatus is not None
    assert StorageError is not None
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
        "ConfigError",
        "MemoryConfig",
        "MemoryEntry",
        "MemoryError",
        "MemoryEvent",
        "MemoryEventType",
        "MemoryIndex",
        "MemoryStatus",
        "StorageError",
        "__version__",
        "namespace_to_path",
        "validate_namespace",
    }
    assert set(trw_memory.__all__) == expected


def test_exceptions_inherit_properly() -> None:
    """Custom exceptions have correct hierarchy."""
    from trw_memory import ConfigError, MemoryError, StorageError

    assert issubclass(MemoryError, Exception)
    assert issubclass(StorageError, MemoryError)
    assert issubclass(ConfigError, MemoryError)


def test_memory_status_is_enum() -> None:
    """MemoryStatus is an enum with expected values."""
    from trw_memory import MemoryStatus

    assert hasattr(MemoryStatus, "ACTIVE")
    assert hasattr(MemoryStatus, "RESOLVED")
    assert hasattr(MemoryStatus, "OBSOLETE")
