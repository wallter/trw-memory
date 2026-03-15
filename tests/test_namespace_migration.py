"""Tests for namespace package migration — backward compat and new imports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.exceptions import ConfigError
from trw_memory.storage.sqlite_backend import SQLiteBackend


class TestBackwardCompatImports:
    """Old imports from trw_memory.namespace still work."""

    def test_validate_namespace_from_old_module(self) -> None:
        """from trw_memory.namespace import validate_namespace works."""
        from trw_memory.namespace import validate_namespace

        assert validate_namespace("default") == "default"

    def test_namespace_to_path_from_old_module(self) -> None:
        """from trw_memory.namespace import namespace_to_path works."""
        from trw_memory.namespace import namespace_to_path

        assert namespace_to_path("global") == Path("global")

    def test_top_level_exports_still_work(self) -> None:
        """from trw_memory import validate_namespace still works."""
        from trw_memory import namespace_to_path, validate_namespace

        assert validate_namespace("project:abc") == "project:abc"
        assert namespace_to_path("project:abc") == Path("project/abc")


class TestNewPackageImports:
    """New imports from trw_memory.namespaces work."""

    def test_validate_namespace_from_package(self) -> None:
        """from trw_memory.namespaces import validate_namespace works."""
        from trw_memory.namespaces import validate_namespace

        assert validate_namespace("default") == "default"

    def test_namespace_to_path_from_package(self) -> None:
        """from trw_memory.namespaces import namespace_to_path works."""
        from trw_memory.namespaces import namespace_to_path

        assert namespace_to_path("team:research") == Path("team/research")

    def test_validate_from_validation_module(self) -> None:
        """from trw_memory.namespaces.validation import validate_namespace works."""
        from trw_memory.namespaces.validation import validate_namespace

        assert validate_namespace("org:acme") == "org:acme"

    def test_path_from_path_mapping_module(self) -> None:
        """from trw_memory.namespaces.path_mapping import namespace_to_path works."""
        from trw_memory.namespaces.path_mapping import namespace_to_path

        assert namespace_to_path("project:repo-a") == Path("project/repo-a")

    def test_invalid_raises_in_new_module(self) -> None:
        """Validation in new module still raises ConfigError."""
        from trw_memory.namespaces.validation import validate_namespace

        with pytest.raises(ConfigError):
            validate_namespace("invalid")


class TestNamespaceManager:
    """NamespaceManager can list, register, and delete namespaces."""

    @pytest.fixture()
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        """Create a temporary backend."""
        return SQLiteBackend(tmp_path / "test.db")

    def test_register_valid_namespace(self, backend: SQLiteBackend) -> None:
        """Register returns the validated namespace."""
        from trw_memory.namespaces import NamespaceManager

        mgr = NamespaceManager(backend)
        assert mgr.register("project:test") == "project:test"

    def test_register_invalid_raises(self, backend: SQLiteBackend) -> None:
        """Register raises on invalid namespace."""
        from trw_memory.namespaces import NamespaceManager

        mgr = NamespaceManager(backend)
        with pytest.raises(ConfigError):
            mgr.register("bad-ns")

    def test_list_empty(self, backend: SQLiteBackend) -> None:
        """List on empty store returns empty list."""
        from trw_memory.namespaces import NamespaceManager

        mgr = NamespaceManager(backend)
        assert mgr.list_namespaces() == []

    def test_list_after_entries(self, backend: SQLiteBackend) -> None:
        """List returns namespaces that have entries."""
        from trw_memory.models.memory import MemoryEntry
        from trw_memory.namespaces import NamespaceManager

        now = datetime.now(timezone.utc)
        backend.store(
            MemoryEntry(
                id="M-001",
                content="test",
                namespace="project:alpha",
                created_at=now,
                updated_at=now,
            )
        )
        backend.store(
            MemoryEntry(
                id="M-002",
                content="test",
                namespace="project:beta",
                created_at=now,
                updated_at=now,
            )
        )

        mgr = NamespaceManager(backend)
        ns_list = mgr.list_namespaces()
        assert "project:alpha" in ns_list
        assert "project:beta" in ns_list

    def test_delete_removes_entries(self, backend: SQLiteBackend) -> None:
        """Delete removes all entries in the namespace."""
        from trw_memory.models.memory import MemoryEntry
        from trw_memory.namespaces import NamespaceManager

        now = datetime.now(timezone.utc)
        backend.store(
            MemoryEntry(
                id="M-001",
                content="test",
                namespace="project:doomed",
                created_at=now,
                updated_at=now,
            )
        )
        backend.store(
            MemoryEntry(
                id="M-002",
                content="test",
                namespace="project:doomed",
                created_at=now,
                updated_at=now,
            )
        )
        backend.store(
            MemoryEntry(
                id="M-003",
                content="keep",
                namespace="default",
                created_at=now,
                updated_at=now,
            )
        )

        mgr = NamespaceManager(backend)
        deleted_count = mgr.delete("project:doomed")
        assert deleted_count == 2
        assert mgr.count("project:doomed") == 0
        # Other namespace untouched
        assert mgr.count("default") == 1

    def test_count_namespace(self, backend: SQLiteBackend) -> None:
        """Count returns the correct number of entries."""
        from trw_memory.models.memory import MemoryEntry
        from trw_memory.namespaces import NamespaceManager

        now = datetime.now(timezone.utc)
        backend.store(
            MemoryEntry(
                id="M-001",
                content="a",
                namespace="default",
                created_at=now,
                updated_at=now,
            )
        )
        backend.store(
            MemoryEntry(
                id="M-002",
                content="b",
                namespace="default",
                created_at=now,
                updated_at=now,
            )
        )

        mgr = NamespaceManager(backend)
        assert mgr.count("default") == 2
        assert mgr.count("project:empty") == 0
