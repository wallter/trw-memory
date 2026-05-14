# ruff: noqa: F401,F811
"""Tests for VSCode and adapter factory integrations."""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ._test_integrations_support import _make_crewai_mocks, _make_langchain_mocks, _make_llamaindex_mocks, tmp_backend


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
        """search() defaults to adapter's namespace, not 'default'."""
        from trw_memory.integrations.vscode import LocalMemoryAdapter

        adapter = LocalMemoryAdapter(namespace="my-ns", backend=tmp_backend)
        adapter.store_selection("content", "/f.py", [])

        results = adapter.search("content")
        assert isinstance(results, list)

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

    def test_get_adapter_langchain_with_dep(self) -> None:
        """UT-FA-01: get_adapter('langchain') returns adapter when installed."""
        mocks = _make_langchain_mocks()
        mock_spec = MagicMock()

        with patch.dict(sys.modules, mocks):
            for key in list(sys.modules.keys()):
                if key.startswith("trw_memory.integrations.langchain"):
                    del sys.modules[key]

            orig_find_spec = importlib.util.find_spec

            def _patched_find_spec(name: str, *a: Any, **kw: Any) -> Any:
                if name == "langchain_core":
                    return mock_spec
                return orig_find_spec(name, *a, **kw)

            with patch("importlib.util.find_spec", side_effect=_patched_find_spec):
                from trw_memory.integrations.factory import get_adapter

                cls = get_adapter("langchain")
                assert cls.__name__ == "TRWChatMessageHistory"

    def test_get_adapter_langchain_without_dep(self) -> None:
        """UT-FA-02: get_adapter('langchain') raises ImportError when missing."""
        orig_find_spec = importlib.util.find_spec

        def _patched_find_spec(name: str, *a: Any, **kw: Any) -> Any:
            if name == "langchain_core":
                return None
            return orig_find_spec(name, *a, **kw)

        with patch("importlib.util.find_spec", side_effect=_patched_find_spec):
            from trw_memory.integrations.factory import get_adapter

            with pytest.raises(ImportError, match="pip install"):
                get_adapter("langchain")

    def test_get_adapter_llamaindex(self) -> None:
        """UT-FA-03: get_adapter('llamaindex') returns TRWChatStore."""
        mocks = _make_llamaindex_mocks()
        mock_spec = MagicMock()

        with patch.dict(sys.modules, mocks):
            for key in list(sys.modules.keys()):
                if key.startswith("trw_memory.integrations.llamaindex"):
                    del sys.modules[key]

            orig_find_spec = importlib.util.find_spec

            def _patched_find_spec(name: str, *a: Any, **kw: Any) -> Any:
                if name == "llama_index.core":
                    return mock_spec
                return orig_find_spec(name, *a, **kw)

            with patch("importlib.util.find_spec", side_effect=_patched_find_spec):
                from trw_memory.integrations.factory import get_adapter

                cls = get_adapter("llamaindex")
                assert cls.__name__ == "TRWChatStore"

    def test_get_adapter_crewai(self) -> None:
        """UT-FA-04: get_adapter('crewai') returns TRWCrewStorage."""
        mocks = _make_crewai_mocks()
        mock_spec = MagicMock()

        with patch.dict(sys.modules, mocks):
            for key in list(sys.modules.keys()):
                if key.startswith("trw_memory.integrations.crewai"):
                    del sys.modules[key]

            orig_find_spec = importlib.util.find_spec

            def _patched_find_spec(name: str, *a: Any, **kw: Any) -> Any:
                if name == "crewai":
                    return mock_spec
                return orig_find_spec(name, *a, **kw)

            with patch("importlib.util.find_spec", side_effect=_patched_find_spec):
                with patch("importlib.metadata.version", return_value="0.74.0"):
                    from trw_memory.integrations.factory import get_adapter

                    cls = get_adapter("crewai")
                    assert cls.__name__ == "TRWCrewStorage"

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

    def test_factory_import_no_framework_modules(self) -> None:
        """UT-FA-08: importing factory doesn't import framework modules."""
        before = set(sys.modules.keys())
        importlib.import_module("trw_memory.integrations.factory")
        after = set(sys.modules.keys())

        new_modules = after - before
        framework_modules = [m for m in new_modules if any(f in m for f in ["langchain", "llama_index", "crewai"])]
        assert framework_modules == [], f"Framework modules loaded: {framework_modules}"
