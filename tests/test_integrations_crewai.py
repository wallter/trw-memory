"""Tests for the CrewAI integration adapter."""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import patch

import pytest

from ._test_integrations_support import (
    _import_crewai_adapter,
    _make_crewai_mocks,
    _purge_modules,
    tmp_backend,
)


class TestCrewAIAdapter:
    """Tests for TRWCrewStorage."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self) -> None:
        self.mocks = _make_crewai_mocks()
        self.mod = _import_crewai_adapter(self.mocks)

    def test_instantiation(self, tmp_backend: Any) -> None:
        """UT-CA-01: TRWCrewStorage instantiates correctly."""
        storage = self.mod.TRWCrewStorage(namespace="test", backend=tmp_backend)
        assert storage.namespace == "test"

    def test_save_stores_entry(self, tmp_backend: Any) -> None:
        """UT-CA-02: save() stores content in backend."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("found a bug in module X")

        entries = tmp_backend.list_entries(namespace="default", limit=100)
        assert len(entries) == 1
        assert entries[0].content == "found a bug in module X"

    def test_save_with_agent_tag(self, tmp_backend: Any) -> None:
        """UT-CA-03: save() with agent adds agent tag."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("finding", agent="researcher")

        entries = tmp_backend.list_entries(namespace="default", limit=100)
        assert "agent:researcher" in entries[0].tags

    def test_search_returns_results(self, tmp_backend: Any) -> None:
        """UT-CA-04: search() returns matching entries."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("bug in authentication module")

        results = storage.search("authentication")
        assert len(results) >= 1
        assert "context" in results[0]

    def test_search_score_threshold(self, tmp_backend: Any) -> None:
        """search() with score_threshold filters results."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("low importance entry")

        results = storage.search("entry", score_threshold=0.6)
        assert len(results) == 0

        results = storage.search("entry", score_threshold=0.5)
        assert len(results) >= 1

    def test_reset_clears_all(self, tmp_backend: Any) -> None:
        """UT-CA-05: reset() clears all entries."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("entry 1")
        storage.save("entry 2")
        assert tmp_backend.count(namespace="default") == 2

        storage.reset()
        assert tmp_backend.count(namespace="default") == 0

    def test_reset_uses_bulk_namespace_delete(self, tmp_backend: Any) -> None:
        """reset() should clear the namespace through the backend bulk path."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        with patch.object(
            tmp_backend, "delete_by_namespace", wraps=tmp_backend.delete_by_namespace
        ) as delete_namespace:
            storage.save("entry 1")
            storage.save("entry 2")
            storage.reset()

        delete_namespace.assert_called_once_with("default")
        assert tmp_backend.count(namespace="default") == 0

    def test_save_with_metadata(self, tmp_backend: Any) -> None:
        """UT-CA-06: save() passes metadata to backend."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("finding", metadata={"priority": "high"})

        entries = tmp_backend.list_entries(namespace="default", limit=100)
        assert entries[0].metadata.get("priority") == "high"

    def test_search_respects_limit(self, tmp_backend: Any) -> None:
        """UT-CA-07: search() respects limit parameter."""
        storage = self.mod.TRWCrewStorage(namespace="default", search_limit=5, backend=tmp_backend)
        for i in range(10):
            storage.save(f"entry about topic {i}")

        results = storage.search("topic", limit=3)
        assert len(results) <= 3

    def test_search_applies_metadata_filter(self, tmp_backend: Any) -> None:
        """search() applies exact-match metadata filters."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        storage.save("auth finding", metadata={"team": "auth"})
        storage.save("billing finding", metadata={"team": "billing"})

        results = storage.search("finding", filter={"team": "auth"})
        assert [result["context"] for result in results] == ["auth finding"]

    def test_import_error_without_crewai(self) -> None:
        """UT-CA-08: ImportError raised when crewai not installed."""
        _purge_modules("trw_memory.integrations.crewai")

        saved = {}
        for key in list(sys.modules.keys()):
            if key == "crewai" or key.startswith("crewai."):
                saved[key] = sys.modules.pop(key)

        try:
            with pytest.raises(ImportError, match="pip install"):
                importlib.import_module("trw_memory.integrations.crewai")
        finally:
            sys.modules.update(saved)
            _purge_modules("trw_memory.integrations.crewai")

    def test_import_error_with_too_old_crewai_version(self) -> None:
        """CrewAI adapter rejects versions older than the documented floor."""
        _purge_modules("trw_memory.integrations.crewai")

        mocks = _make_crewai_mocks()
        with patch.dict(sys.modules, mocks):
            with patch("importlib.metadata.version", return_value="0.73.9"):
                with pytest.raises(ImportError, match=r"crewai>=0\.74\.0"):
                    importlib.import_module("trw_memory.integrations.crewai")

    def test_context_manager(self, tmp_backend: Any) -> None:
        """Context manager calls close() on exit."""
        storage = self.mod.TRWCrewStorage(namespace="default", backend=tmp_backend)
        with storage as s:
            assert s is storage
