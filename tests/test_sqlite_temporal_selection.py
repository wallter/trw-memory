"""Opt-in temporal selection on public SQLite search/list preserves raw listing."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.temporal_selection import TemporalSelection
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend


@pytest.mark.parametrize("backend_type", [SQLiteBackend, YAMLBackend])
@pytest.mark.parametrize("method", ["search", "list_entries"])
@pytest.mark.parametrize("include", [False, True])
def test_public_storage_selects_eligible_before_cap(
    tmp_path: Path, method: str, include: bool, backend_type, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = backend_type(tmp_path / "store")
    early = datetime(2020, 1, 1, tzinfo=timezone.utc)
    later = datetime(2025, 1, 1, tzinfo=timezone.utc)
    try:
        for i in range(20):
            backend.store(
                MemoryEntry(
                    id=f"old{i:03}",
                    content="policy",
                    importance=0.9,
                    valid_from=early,
                    invalid_from=later,
                    invalidated_by="current",
                    updated_at=later,
                    namespace="project:test",
                    tags=["wanted"],
                )
            )
        backend.store(
            MemoryEntry(
                id="current",
                content="policy",
                importance=0.4,
                valid_from=early,
                updated_at=early,
                namespace="project:test",
                tags=["wanted"],
            )
        )
        backend.store(
            MemoryEntry(id="foreign", content="policy", importance=1, namespace="project:other", tags=["wanted"])
        )
        backend.store(MemoryEntry(id="untagged", content="policy", importance=1, namespace="project:test"))
        call = backend.search if method == "search" else backend.list_entries
        args = ("policy",) if method == "search" else ()
        kwargs = {"top_k": 3} if method == "search" else {"limit": 3}
        raw = call(*args, namespace="project:test", tags=["wanted"], **kwargs)
        assert len(raw) == 3 and all(row.id.startswith("old") for row in raw)
        if backend_type is YAMLBackend:

            def forbid_bulk_load():
                raise AssertionError("Temporal YAML must not call _load_all")

            monkeypatch.setattr(backend, "_load_all", forbid_bulk_load)
        selected = call(
            *args,
            namespace="project:test",
            tags=["wanted"],
            temporal_selection=TemporalSelection(include_superseded=include),
            **kwargs,
        )
        assert selected[0].id == "current"
        assert len(selected) == (3 if include else 1)
        assert all(row.id not in {"foreign", "untagged"} for row in selected)
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", [SQLiteBackend, YAMLBackend])
def test_temporal_selection_rejects_incompatible_raw_cursor(tmp_path: Path, backend_type) -> None:
    from trw_memory.storage.interface import EntryCursor

    backend = backend_type(tmp_path / "store")
    try:
        with pytest.raises(ValueError, match="raw-order cursor"):
            backend.list_entries(
                after=EntryCursor(updated_at="2022-01-01T00:00:00+00:00", entry_id="last"),
                temporal_selection=TemporalSelection(),
            )
    finally:
        backend.close()
