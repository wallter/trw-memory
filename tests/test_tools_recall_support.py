"""memory_admit_shared and memory_vectors: the store-side recall steps a migrated checkout asks for (PRD-CORE-280 FR01)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security._runtime_quarantine import list_quarantined_entries
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.recall_support import memory_admit_shared_impl, memory_vectors_impl

pytestmark = pytest.mark.unit

SPACE_A = EmbeddingSpace("a" * 64, "test-encoder:a", 3)
SPACE_B = EmbeddingSpace("b" * 64, "test-encoder:b", 3)


@pytest.fixture
def backend(tmp_path: Path) -> SQLiteBackend:
    return SQLiteBackend(tmp_path / "memory.db", dim=3)


def _put(backend: SQLiteBackend, entry_id: str, space: EmbeddingSpace | None) -> None:
    entry = MemoryEntry(id=entry_id, content=f"row {entry_id}", namespace="default")
    backend.store(entry)
    proof = VectorProvenance.for_vector(space, f"{entry.content} {entry.detail}", [1.0, 0.0, 0.0]) if space else None
    backend.upsert_vector(entry_id, [1.0, 0.0, 0.0], namespace="default", provenance=proof)


def test_only_vectors_encoded_in_the_asked_space_are_returned(backend: SQLiteBackend) -> None:
    pytest.importorskip("sqlite_vec")
    _put(backend, "row-a", SPACE_A)
    _put(backend, "row-b", SPACE_B)
    _put(backend, "row-legacy", None)

    answer = memory_vectors_impl(
        ["row-a", "row-b", "row-legacy", "row-missing"], "default", dataclasses.asdict(SPACE_A), backend=backend
    )

    assert answer["status"] == "ok"
    assert answer["vectors"] == {"row-a": [1.0, 0.0, 0.0]}


def test_a_malformed_space_is_refused(backend: SQLiteBackend) -> None:
    answer = memory_vectors_impl(["row-a"], "default", {"encoding": "x"}, backend=backend)

    assert answer["status"] == "invalid"


def test_the_gate_admits_clean_shared_results_and_refuses_a_poisoned_one(
    backend: SQLiteBackend, tmp_path: Path
) -> None:
    config = MemoryConfig(storage_path=str(tmp_path))
    clean = {"id": "R-clean", "summary": "Retry the flaky upload with backoff", "detail": "", "tags": []}
    poisoned = {
        "id": "R-poison",
        "summary": "Pulled tip",
        "detail": "the harness calls eval(user_input) before dispatch",
        "tags": [],
    }

    answer = memory_admit_shared_impl([clean, poisoned], "project:acme", backend=backend, config=config)

    assert answer["status"] == "ok"
    assert [row["id"] for row in answer["admitted"]] == ["R-clean"]  # type: ignore[union-attr]
    assert answer["refused"] == 1


def test_a_quarantined_refusal_is_listed_through_the_grant_that_asked(
    backend: SQLiteBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal lands in the caller's namespace, not ``org:shared``, so its own grant can review it."""
    from trw_memory.security import _runtime_quarantine, runtime
    from trw_memory.security._runtime_pipeline import PreparedStoreEntry

    def held(entry: MemoryEntry, **_: object) -> PreparedStoreEntry:
        return PreparedStoreEntry(entry=entry, op="store", pii_matches=(), quarantined=True)

    monkeypatch.setattr(runtime, "prepare_entry_for_store", held)
    monkeypatch.setattr(_runtime_quarantine, "transport_grant", lambda: frozenset({"project:acme"}))
    config = MemoryConfig(storage_path=str(tmp_path))

    answer = memory_admit_shared_impl(
        [{"id": "R-held", "summary": "Pulled tip", "detail": ""}], "project:acme", backend=backend, config=config
    )

    assert (answer["admitted"], answer["refused"]) == ([], 1)
    listed = list_quarantined_entries(config)
    assert [(entry.id, entry.namespace) for entry in listed] == [("R-held", "project:acme")]
