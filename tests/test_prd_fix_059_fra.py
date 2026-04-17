"""Tests for PRD-FIX-059 FR-01 (ID Collision) and FR-03 (ABC Completion).

FR-01: Increase ID entropy from 8 hex chars to 16 hex chars.
FR-03: Complete StorageBackend ABC with non-abstract default methods.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend


def _make_test_entry(
    *,
    entry_id: str = "M-test001",
    content: str = "test content",
    namespace: str = "default",
    tags: list[str] | None = None,
    importance: float = 0.5,
) -> MemoryEntry:
    """Create a MemoryEntry for testing with sensible defaults."""
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail="",
        tags=tags or [],
        importance=importance,
        status=MemoryStatus.ACTIVE,
        namespace=namespace,
        created_at=now,
        updated_at=now,
        source="agent",
    )


# ---------------------------------------------------------------------------
# FR-01: ID Entropy -- client._make_id()
# ---------------------------------------------------------------------------


class TestMakeIdFormat:
    """Verify _make_id() produces M- prefix + 16 hex characters."""

    def test_make_id_format(self) -> None:
        """FR-01: ID must match pattern M-[0-9a-f]{16}."""
        from trw_memory.client import _make_id

        mid = _make_id()
        assert re.fullmatch(r"M-[0-9a-f]{16}", mid), f"Expected M-<16 hex chars>, got {mid!r}"

    def test_make_id_uniqueness(self) -> None:
        """FR-01: 100 generated IDs must all be unique (collision resistance)."""
        from trw_memory.client import _make_id

        ids = {_make_id() for _ in range(100)}
        assert len(ids) == 100, f"Expected 100 unique IDs, got {len(ids)}"

    def test_make_id_length(self) -> None:
        """FR-01: Total ID length must be 2 (prefix) + 16 (hex) = 18."""
        from trw_memory.client import _make_id

        mid = _make_id()
        assert len(mid) == 18, f"Expected length 18, got {len(mid)}"


# ---------------------------------------------------------------------------
# FR-01: ID Entropy -- tools/store.py
# ---------------------------------------------------------------------------


class TestToolsStoreIdFormat:
    """Verify tools/store.py generates 16-hex-char IDs."""

    def test_tools_store_id_format(self, tmp_path: Path) -> None:
        """FR-01: memory_store_impl must produce M-<16 hex chars> IDs."""
        from trw_memory.storage.yaml_backend import YAMLBackend
        from trw_memory.tools.store import memory_store_impl

        backend = YAMLBackend(entries_dir=tmp_path / "entries")
        result = memory_store_impl(
            "test content",
            "default",
            backend=backend,
            tags=["test"],
        )
        assert result["status"] == "stored"
        memory_id = str(result["memory_id"])
        assert re.fullmatch(r"M-[0-9a-f]{16}", memory_id), f"Expected M-<16 hex chars>, got {memory_id!r}"


# ---------------------------------------------------------------------------
# FR-03: StorageBackend ABC -- default methods
# ---------------------------------------------------------------------------


class TestStorageBackendDefaults:
    """Verify non-abstract default methods on StorageBackend ABC."""

    def _make_concrete_backend(self) -> StorageBackend:
        """Create a minimal concrete subclass to test default implementations."""
        from trw_memory.storage.interface import StorageBackend

        class MinimalBackend(StorageBackend):
            """Concrete backend implementing only abstract methods."""

            def store(self, entry: MemoryEntry) -> None:
                pass

            def get(self, entry_id: str) -> MemoryEntry | None:
                return None

            def update(self, entry_id: str, **fields: object) -> MemoryEntry | None:
                return None

            def delete(self, entry_id: str) -> bool:
                return False

            def search(
                self,
                query: str,
                *,
                top_k: int = 25,
                tags: list[str] | None = None,
                status: MemoryStatus | None = None,
                min_importance: float = 0.0,
                namespace: str | None = None,
            ) -> list[MemoryEntry]:
                return []

            def count(self, namespace: str | None = None) -> int:
                return 0

            def list_entries(
                self,
                *,
                status: MemoryStatus | None = None,
                namespace: str | None = None,
                limit: int = 100,
            ) -> list[MemoryEntry]:
                return []

            def close(self) -> None:
                pass

        return MinimalBackend()

    def test_storage_backend_default_list_namespaces(self) -> None:
        """FR-03: Default list_namespaces() must return empty list."""
        backend = self._make_concrete_backend()
        result = backend.list_namespaces()
        assert result == []

    def test_storage_backend_default_delete_by_namespace(self) -> None:
        """FR-03: Default delete_by_namespace() must return 0."""
        backend = self._make_concrete_backend()
        result = backend.delete_by_namespace("anything")
        assert result == 0

    def test_storage_backend_default_upsert_vector(self) -> None:
        """FR-03: Default upsert_vector() must be a no-op (no error)."""
        backend = self._make_concrete_backend()
        # Should not raise
        backend.upsert_vector("M-test", [0.1, 0.2, 0.3])

    def test_storage_backend_default_search_vectors(self) -> None:
        """FR-03: Default search_vectors() must return empty list."""
        backend = self._make_concrete_backend()
        result = backend.search_vectors([0.1, 0.2, 0.3], top_k=5)
        assert result == []


# ---------------------------------------------------------------------------
# FR-03: YAMLBackend -- new method implementations
# ---------------------------------------------------------------------------


class TestYAMLBackendNamespaceOps:
    """Verify YAMLBackend list_namespaces and delete_by_namespace."""

    def test_yaml_backend_list_namespaces(self, tmp_path: Path) -> None:
        """FR-03: list_namespaces returns sorted unique namespaces."""
        from trw_memory.storage.yaml_backend import YAMLBackend

        backend = YAMLBackend(entries_dir=tmp_path / "entries")

        # Store entries in two different namespaces
        e1 = _make_test_entry(entry_id="M-ns1a", namespace="project:alpha", content="alpha entry 1")
        e2 = _make_test_entry(entry_id="M-ns1b", namespace="project:alpha", content="alpha entry 2")
        e3 = _make_test_entry(entry_id="M-ns2a", namespace="project:beta", content="beta entry")
        backend.store(e1)
        backend.store(e2)
        backend.store(e3)

        namespaces = backend.list_namespaces()
        assert namespaces == ["project:alpha", "project:beta"]

    def test_yaml_backend_list_namespaces_empty(self, tmp_path: Path) -> None:
        """FR-03: list_namespaces returns [] when no entries exist."""
        from trw_memory.storage.yaml_backend import YAMLBackend

        backend = YAMLBackend(entries_dir=tmp_path / "entries")
        assert backend.list_namespaces() == []

    def test_yaml_backend_delete_by_namespace(self, tmp_path: Path) -> None:
        """FR-03: delete_by_namespace removes matching entries, returns count."""
        from trw_memory.storage.yaml_backend import YAMLBackend

        backend = YAMLBackend(entries_dir=tmp_path / "entries")

        # Store 3 entries: 2 in "alpha", 1 in "beta"
        e1 = _make_test_entry(entry_id="M-del1", namespace="alpha", content="alpha 1")
        e2 = _make_test_entry(entry_id="M-del2", namespace="alpha", content="alpha 2")
        e3 = _make_test_entry(entry_id="M-del3", namespace="beta", content="beta 1")
        backend.store(e1)
        backend.store(e2)
        backend.store(e3)

        assert backend.count() == 3

        # Delete "alpha" namespace
        deleted = backend.delete_by_namespace("alpha")
        assert deleted == 2

        # Only "beta" entry remains
        assert backend.count() == 1
        remaining = backend.list_entries()
        assert len(remaining) == 1
        assert remaining[0].namespace == "beta"

    def test_yaml_backend_delete_by_namespace_none_match(self, tmp_path: Path) -> None:
        """FR-03: delete_by_namespace returns 0 when no entries match."""
        from trw_memory.storage.yaml_backend import YAMLBackend

        backend = YAMLBackend(entries_dir=tmp_path / "entries")

        e1 = _make_test_entry(entry_id="M-x1", namespace="alpha", content="hello")
        backend.store(e1)

        deleted = backend.delete_by_namespace("nonexistent")
        assert deleted == 0
        assert backend.count() == 1
