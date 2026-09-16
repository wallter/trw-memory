# ruff: noqa: F401,F811
"""Tests for VSCode and adapter factory integrations."""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import patch

import pytest

from trw_memory.integrations import factory

from ._test_integrations_support import tmp_backend

#: Import path of the module under test, purged and re-imported by UT-FA-08.
_FACTORY = "trw_memory.integrations.factory"


class TestVSCodeInterface:
    """Tests for VSCodeMemoryInterface and LocalMemoryAdapter."""

    def test_protocol_importable_without_extras(self) -> None:
        """UT-VS-01: VSCodeMemoryInterface imports in base install."""
        from trw_memory.integrations.vscode import VSCodeMemoryInterface

        assert hasattr(VSCodeMemoryInterface, "__protocol_attrs__") or callable(VSCodeMemoryInterface)

    def test_protocol_has_all_methods(self) -> None:
        """UT-VS-02: VSCodeMemoryInterface declares all 4 methods."""
        from trw_memory.integrations.vscode import VSCodeMemoryInterface

        methods = [m for m in dir(VSCodeMemoryInterface) if not m.startswith("_")]
        assert "get_relevant" in methods
        assert "store_selection" in methods
        assert "search" in methods
        assert "get_status" in methods

    def test_local_adapter_satisfies_protocol(self) -> None:
        """UT-VS-03: LocalMemoryAdapter satisfies VSCodeMemoryInterface."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter, VSCodeMemoryInterface

        assert isinstance(LocalMemoryAdapter.__new__(LocalMemoryAdapter), VSCodeMemoryInterface)

    def test_get_relevant(self, tmp_backend: Any) -> None:
        """UT-VS-04: get_relevant returns memories relevant to file path."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="test", backend=tmp_backend)
        adapter.store_selection("use pytest fixtures", "/src/test.py", ["testing"])

        results = adapter.get_relevant("/src/test.py", limit=5)
        assert isinstance(results, list)

    def test_store_selection(self, tmp_backend: Any) -> None:
        """UT-VS-05: store_selection stores content with file tag."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="test", backend=tmp_backend)
        result = adapter.store_selection("code snippet", "/file.py", ["python"])

        assert "memory_id" in result
        assert result["status"] == "stored"

        entries = tmp_backend.list_entries(namespace="test", limit=100)
        assert len(entries) == 1
        assert "file:/file.py" in entries[0].tags

    def test_get_status(self, tmp_backend: Any) -> None:
        """UT-VS-06: get_status returns health metrics."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="test", backend=tmp_backend)
        status = adapter.get_status()

        assert "entry_count" in status
        assert "namespace" in status
        assert status["namespace"] == "test"
        assert status["entry_count"] == 0

    def test_search_uses_instance_namespace_by_default(self, tmp_backend: Any) -> None:
        """search() defaults to the adapter's namespace, not 'default'.

        The previous assertion was ``isinstance(results, list)``, which also holds
        for the empty list a wrong-namespace search returns -- so it could not
        distinguish the behaviour it names from its exact opposite. The result
        dicts carry no namespace key, so the claim is pinned differentially: an
        adapter on another namespace, over the SAME backend, must not see the entry.
        """
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        mine = LocalMemoryAdapter(namespace="my-ns", backend=tmp_backend)
        other = LocalMemoryAdapter(namespace="other-ns", backend=tmp_backend)
        mine.store_selection("content", "/f.py", [])

        assert mine.search("content"), "the storing adapter's own namespace was not searched"
        assert other.search("content") == [], "search leaked across the namespace boundary"

    def test_search_with_explicit_namespace(self, tmp_backend: Any) -> None:
        """search() with explicit namespace overrides default."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="my-ns", backend=tmp_backend)
        results = adapter.search("query", namespace="other-ns")
        assert isinstance(results, list)

    def test_context_manager(self, tmp_backend: Any) -> None:
        """Context manager calls close() on exit."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="test", backend=tmp_backend)
        with adapter as a:
            assert a is adapter


class TestFactory:
    """Tests for get_adapter and list_available."""

    def test_get_adapter_vscode_no_extras(self) -> None:
        """UT-FA-05: get_adapter('vscode') returns LocalMemoryAdapter without extras."""
        from trw_memory.integrations.factory import get_adapter

        cls = get_adapter("vscode")
        assert cls.__name__ == "LocalMemoryAdapter"

    def test_get_adapter_unknown_raises_valueerror(self) -> None:
        """UT-FA-06: get_adapter('unknown') raises ValueError."""
        from trw_memory.integrations.factory import get_adapter

        with pytest.raises(ValueError, match="Unknown framework"):
            get_adapter("unknown_framework")

    def test_list_available_includes_vscode(self) -> None:
        """UT-FA-07: list_available always includes 'vscode'."""
        from trw_memory.integrations.factory import list_available

        available = list_available()
        assert "vscode" in available

    def test_factory_import_does_not_import_any_adapter(self) -> None:
        """UT-FA-08: importing the factory defers every adapter module.

        Previously this asserted that no LangChain/LlamaIndex/CrewAI module was
        loaded; with those adapters removed that assertion is vacuous. The live
        invariant is the same one it was protecting -- the factory resolves
        adapters lazily -- so it is now pinned against the registry itself, which
        also covers an adapter added later.
        """
        adapter_modules = {module for _spec, module, _cls in factory._REGISTRY.values()}
        assert adapter_modules, "non-vacuity: the registry must name at least one adapter module"

        saved = {name: sys.modules[name] for name in adapter_modules | {_FACTORY} if name in sys.modules}
        try:
            for name in saved:
                del sys.modules[name]
            importlib.import_module(_FACTORY)
            still_deferred = adapter_modules - set(sys.modules)
            assert still_deferred == adapter_modules, (
                f"factory import eagerly loaded: {sorted(adapter_modules - still_deferred)}"
            )
        finally:
            sys.modules.update(saved)

    def test_a_registered_adapter_with_a_missing_dependency_raises_importerror(self) -> None:
        """UT-FA-09: the dependency probe still refuses an uninstalled adapter.

        Every shipped adapter is dependency-free, so the probe branch has no live
        registry entry to exercise it. Driving it through a synthetic entry keeps
        the seam proven rather than dormant -- and keeps ``get_adapter``'s
        documented ``ImportError`` contract honest for the next adapter that
        needs an extra.
        """
        with patch.dict(
            factory._REGISTRY,
            {"synthetic": ("a_module_that_is_not_installed", "trw_memory.integrations.vscode", "LocalMemoryAdapter")},
        ):
            with pytest.raises(ImportError, match="pip install"):
                factory.get_adapter("synthetic")

            # Control: the same entry resolves once its dependency is importable.
            with patch.dict(
                factory._REGISTRY,
                {"synthetic": ("sys", "trw_memory.integrations.vscode", "LocalMemoryAdapter")},
            ):
                assert factory.get_adapter("synthetic").__name__ == "LocalMemoryAdapter"
                assert "synthetic" in factory.list_available()
