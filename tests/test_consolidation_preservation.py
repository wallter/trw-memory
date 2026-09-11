"""Consolidation preserves maintenance evidence through real SQLite archival."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.lifecycle.consolidation import consolidate_cycle
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import Assertion, AssertionType, Confidence, MemoryEntry, MemoryStatus, ProtectionTier
from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_consolidation_support import _make_embedder


def _sources() -> list[MemoryEntry]:
    common = Assertion(
        type=AssertionType.GREP_PRESENT,
        pattern="keep",
        target="src/a.py",
        last_result=True,
        last_verified_at=datetime.now(timezone.utc),
        last_evidence="old check passed",
        commit_hash="a" * 40,
    )
    extra = Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/b.py")
    return [
        MemoryEntry(
            id="M-first",
            namespace="project/preservation",
            content="Preserve critical constraints",
            helpful_count=2,
            unhelpful_count=3,
            q_observations=4,
            access_count=5,
            recall_count=6,
            protection_tier=ProtectionTier.CRITICAL,
            assertions=[common],
            confidence=Confidence.HIGH,
        ),
        MemoryEntry(
            id="M-second",
            namespace="project/preservation",
            content="Preserve permanent constraints too",
            helpful_count=7,
            unhelpful_count=8,
            q_observations=9,
            access_count=10,
            recall_count=11,
            protection_tier=ProtectionTier.PERMANENT,
            assertions=[common, extra],
            confidence=Confidence.HIGH,
        ),
    ]


def _cycle(backend: SQLiteBackend) -> dict[str, object]:
    return consolidate_cycle(
        backend,
        _make_embedder(vectors=[[1.0, 0.0], [1.0, 0.0]]),
        namespace="project/preservation",
        config=MemoryConfig(consolidation_enabled=True, consolidation_min_cluster=2),
    )


@pytest.fixture(autouse=True)
def _no_background_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    # Graph scheduling has independent teardown ownership; this test verifies
    # primary-store replacement/archival without spawning unrelated workers.
    monkeypatch.setattr("trw_memory.lifecycle.consolidation.schedule_graph_update", lambda *a, **kw: False)


def test_public_cycle_preserves_constraints_and_feedback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "memory.sqlite"
    originals = _sources()
    with SQLiteBackend(path) as backend:
        monkeypatch.setattr(backend, "supports_vectors", lambda: False)
        for entry in originals:
            backend.store(entry)
        result = _cycle(backend)
        assert result["consolidated_count"] == 1
        assert "errors" not in result

    # Reopening proves persistence, not merely the returned in-memory model.
    with SQLiteBackend(path) as backend:
        active = backend.list_entries(namespace="project/preservation", status=MemoryStatus.ACTIVE)
        assert len(active) == 1
        replacement = active[0]
        assert replacement.helpful_count == 9
        assert replacement.unhelpful_count == 11
        assert replacement.q_observations == 13
        assert replacement.access_count == 15
        assert replacement.recall_count == 17
        assert replacement.protection_tier == ProtectionTier.PERMANENT
        assert len(replacement.assertions) == 2
        assert {(a.type, a.pattern, a.target) for a in replacement.assertions} == {
            (a.type, a.pattern, a.target) for source in originals for a in source.assertions
        }
        assert replacement.confidence != Confidence.HIGH
        assert replacement.verification_status is None
        assert all(a.last_result is None and a.last_verified_at is None for a in replacement.assertions)
        assert all(a.last_evidence == "" and a.first_failed_at is None for a in replacement.assertions)
        assert any(a.commit_hash == "a" * 40 for a in replacement.assertions)
        for source in originals:
            archived = backend.get(source.id, namespace=source.namespace)
            assert archived is not None
            assert archived.status == MemoryStatus.ARCHIVED
            assert archived.consolidated_into == replacement.id
            assert archived.invalidated_by == replacement.id
            assert archived.unhelpful_count == source.unhelpful_count
            assert archived.assertions == source.assertions


def test_archival_failure_restores_original_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with SQLiteBackend(tmp_path / "memory.sqlite") as backend:
        monkeypatch.setattr(backend, "supports_vectors", lambda: False)
        originals = _sources()
        for entry in originals:
            backend.store(entry)
        # Compensation re-stores originals, legitimately advancing sync_seq;
        # every knowledge, protection, provenance and lifecycle field must survive.
        before = {
            e.id: e.model_dump(mode="json", exclude={"sync_seq"})
            for e in backend.list_entries(namespace="project/preservation")
        }
        original_update = backend.update
        calls = 0

        def fail_second_archive(entry_id: str, *, namespace: str, **fields: object) -> MemoryEntry | None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise StorageError("injected archival failure")
            return original_update(entry_id, namespace=namespace, **fields)

        monkeypatch.setattr(backend, "update", fail_second_archive)
        result = _cycle(backend)
        assert result["consolidated_count"] == 0
        assert result["errors"]
        after = {
            e.id: e.model_dump(mode="json", exclude={"sync_seq"})
            for e in backend.list_entries(namespace="project/preservation")
        }
        assert after == before


def test_replacement_write_failure_does_not_archive_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with SQLiteBackend(tmp_path / "memory.sqlite") as backend:
        monkeypatch.setattr(backend, "supports_vectors", lambda: False)
        for entry in _sources():
            backend.store(entry)
        before = {e.id: e.model_dump(mode="json") for e in backend.list_entries(namespace="project/preservation")}

        def fail_store(entry: MemoryEntry) -> str:
            raise StorageError("injected replacement write failure")

        monkeypatch.setattr(backend, "store", fail_store)
        result = _cycle(backend)
        assert result["consolidated_count"] == 0
        assert result["errors"]
        after = {e.id: e.model_dump(mode="json") for e in backend.list_entries(namespace="project/preservation")}
        assert after == before
