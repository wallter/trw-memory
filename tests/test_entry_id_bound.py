"""B71-85: every store refuses an entry id a resumable sweep's cursor could not carry."""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import SchemaValidationError
from trw_memory.models.memory import MAX_ENTRY_ID_CHARS, MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend


def test_an_id_at_the_bound_is_stored_and_one_past_it_is_refused(tmp_path: Path) -> None:
    store = SQLiteBackend(tmp_path / "memory.db")
    longest = "a" * MAX_ENTRY_ID_CHARS
    store.store(MemoryEntry(id=longest, content="at the bound", namespace="default"))
    assert store.get(longest, namespace="default") is not None

    with pytest.raises(SchemaValidationError) as refused:
        store.store(MemoryEntry(id=longest + "a", content="past the bound", namespace="default"))
    assert (refused.value.reason, refused.value.failed_fields) == ("entry_id_too_long", ["id"])


def test_the_yaml_backend_refuses_an_id_past_the_bound_before_touching_the_filesystem(tmp_path: Path) -> None:
    """Its file name is the id, so the filesystem's own name limit is lower still; the bound refuses first."""
    store = YAMLBackend(tmp_path / "entries")
    with pytest.raises(SchemaValidationError) as refused:
        store.store(MemoryEntry(id="a" * (MAX_ENTRY_ID_CHARS + 1), content="past the bound", namespace="default"))
    assert refused.value.reason == "entry_id_too_long"
    assert not any((tmp_path / "entries").rglob("*.yaml"))


def test_store_many_refuses_the_whole_batch_before_writing(tmp_path: Path) -> None:
    store = SQLiteBackend(tmp_path / "memory.db")
    batch = [
        MemoryEntry(id="fine", content="fine", namespace="default"),
        MemoryEntry(id="a" * (MAX_ENTRY_ID_CHARS + 1), content="past the bound", namespace="default"),
    ]
    with pytest.raises(SchemaValidationError):
        store.store_many(batch)
    assert store.get("fine", namespace="default") is None
