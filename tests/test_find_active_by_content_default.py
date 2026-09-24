"""The interface's default exact-content lookup, exercised through a backend that does not override it."""

from __future__ import annotations

from pathlib import Path

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.yaml_backend import YAMLBackend


def test_default_lookup_finds_only_an_active_exact_copy_in_the_namespace(tmp_path: Path) -> None:
    backend = YAMLBackend(tmp_path / "entries")
    backend.store(MemoryEntry(id="L-a", content="tip", detail="d", namespace="default"))
    backend.store(MemoryEntry(id="L-b", content="old", detail="d", namespace="default", status=MemoryStatus.OBSOLETE))

    assert backend.find_active_by_content("tip", "d") == "L-a"
    assert backend.find_active_by_content("tip", "other") is None
    assert backend.find_active_by_content("old", "d") is None
    assert backend.find_active_by_content("tip", "d", namespace="elsewhere") is None
