# ruff: noqa: F401,F811
"""YAML field parity backward-compatibility tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.yaml_backend import YAMLBackend

from ._test_yaml_field_parity_support import backend, write_entry_yaml


@pytest.mark.unit
class TestBackwardCompatDefaults:
    def test_missing_fields_have_defaults(self, backend: YAMLBackend) -> None:
        entry = MemoryEntry(id="M-old-1", content="legacy entry", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        backend.store(entry)
        loaded = backend.get("M-old-1")
        assert loaded is not None
        assert loaded.vector_clock == {}
        assert loaded.remote_id is None
        assert loaded.published_to_platform is False
        assert loaded.pending_delete is False
        assert loaded.cross_validated is False
        assert loaded.outcome_history == []
        assert loaded.assertions == []

    def test_remote_id_none_round_trip(self, backend: YAMLBackend) -> None:
        entry = MemoryEntry(id="M-none-rid", content="no remote id", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc), remote_id=None)
        backend.store(entry)
        loaded = backend.get("M-none-rid")
        assert loaded is not None and loaded.remote_id is None


@pytest.mark.unit
class TestTrueLegacyYAML:
    def test_legacy_yaml_without_new_keys(self, backend: YAMLBackend) -> None:
        legacy_data: dict[str, object] = {
            "id": "M-legacy-1",
            "content": "legacy entry from v0.4.0",
            "detail": "",
            "tags": ["old"],
            "evidence": [],
            "importance": 0.7,
            "status": "active",
            "recurrence": 1,
            "namespace": "default",
            "created_at": "2026-01-15T10:00:00+00:00",
            "updated_at": "2026-01-15T10:00:00+00:00",
            "last_accessed_at": None,
            "access_count": 0,
            "q_value": 0.5,
            "q_observations": 0,
            "source": "agent",
            "source_identity": "",
            "merged_from": [],
            "consolidated_from": [],
            "consolidated_into": None,
            "metadata": {},
        }
        write_entry_yaml(backend, "M-legacy-1", legacy_data)
        loaded = backend.get("M-legacy-1")
        assert loaded is not None
        assert loaded.content == "legacy entry from v0.4.0"
        assert loaded.vector_clock == {}
        assert loaded.remote_id is None
        assert loaded.published_to_platform is False
        assert loaded.pending_delete is False
        assert loaded.cross_validated is False
        assert loaded.outcome_history == []
        assert loaded.assertions == []
