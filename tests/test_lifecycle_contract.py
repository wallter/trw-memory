"""PRD-CORE-294 FR03: correct or retire a learning by id through one implementation.

Drives trw-memory's ``memory_update`` surface (``memory_update_impl``) against a
real SQLite store; trw-mcp's ``trw_learn(learning_id=...)`` side of the same
contract lives in ``trw-mcp/tests/test_learn_update_by_id.py``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import patch

import pytest

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.lifecycle.correction import LearningPatch, parse_patch
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.tools.update import memory_update_impl

NAMESPACE = "project:default"


@pytest.fixture()
def store() -> Iterator[tuple[StorageBackend, MemoryConfig]]:
    with tempfile.TemporaryDirectory() as td:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=td)
        with create_backend_from_config(cfg, NAMESPACE) as backend:
            backend.store(
                MemoryEntry(
                    id="L-1",
                    content="retry the flaky network call with backoff",
                    detail="original detail",
                    tags=["net", "retry"],
                    importance=0.4,
                    namespace=NAMESPACE,
                )
            )
            yield backend, cfg


def _update(store: tuple[StorageBackend, MemoryConfig], entry_id: str, **patch_fields: object) -> dict[str, str]:
    """Drive the tool as the MCP wrapper does: parse the raw dict, then apply the typed patch."""
    backend, cfg = store
    parsed = parse_patch(patch_fields)
    if not isinstance(parsed, LearningPatch):
        return parsed
    return memory_update_impl(entry_id, parsed, NAMESPACE, backend=backend, config=cfg)


def _get(store: tuple[StorageBackend, MemoryConfig], entry_id: str) -> MemoryEntry:
    entry = store[0].get(entry_id, namespace=NAMESPACE)
    assert entry is not None
    return entry


def test_named_fields_change_and_the_rest_do_not(store: tuple[StorageBackend, MemoryConfig]) -> None:
    result = _update(store, "L-1", impact=0.9, detail="corrected detail")

    assert result == {"learning_id": "L-1", "status": "updated", "changes": "detail updated, impact→0.9"}
    entry = _get(store, "L-1")
    assert (entry.id, entry.importance, entry.detail) == ("L-1", 0.9, "corrected detail")
    assert entry.content == "retry the flaky network call with backoff"
    assert entry.tags == ["net", "retry"]


def test_tags_replace_but_tags_add_appends_without_duplicates(store: tuple[StorageBackend, MemoryConfig]) -> None:
    _update(store, "L-1", tags_add=["retry", "backoff", "backoff"])
    assert _get(store, "L-1").tags == ["net", "retry", "backoff"]

    _update(store, "L-1", tags=["only"])
    assert _get(store, "L-1").tags == ["only"]


def test_unknown_id_fails_loudly(store: tuple[StorageBackend, MemoryConfig]) -> None:
    result = _update(store, "L-missing", impact=0.5)

    assert result["status"] == "not_found"
    assert result["error_type"] == "learning_not_found"


@pytest.mark.parametrize(
    ("patch_fields", "field"),
    [({"impact": 1.5}, "impact"), ({"status": "archived"}, "status"), ({"q_value": 0.3}, "q_value")],
)
def test_invalid_patch_is_refused_and_nothing_is_written(
    store: tuple[StorageBackend, MemoryConfig], patch_fields: dict[str, object], field: str
) -> None:
    result = _update(store, "L-1", **patch_fields)

    assert result["status"] == "invalid"
    assert result["error"].startswith(f"Invalid {field}")
    assert _get(store, "L-1").importance == 0.4


def test_unsubstantiated_verified_promotion_is_refused(store: tuple[StorageBackend, MemoryConfig]) -> None:
    # PRD-CORE-312-FR02: evidence_level="verified" isolates this test to the
    # ARTIFACT-substantiation axis; the evidence-level axis has its own test below.
    result = _update(store, "L-1", confidence="verified", evidence_level="verified")

    assert result["status"] == "invalid"
    assert result["reason"] == "unsubstantiated_verified"
    assert _get(store, "L-1").confidence == "unverified"


def _seed_legacy_violation(backend: StorageBackend) -> None:
    """Simulate a pre-FR01 row: ``confidence='verified'`` with no evidence_level ever
    recorded, written by a release before the invariant existed. No CURRENT write path
    can create this state (that is the point of the invariant), so the only honest way
    to seed it in a test is the same way a real legacy row got there: underneath the
    invariant, via a raw column write.
    """
    sqlite_backend = cast("Any", backend)
    with sqlite_backend._lock:
        sqlite_backend._conn.execute(
            "UPDATE memories SET confidence = 'verified' WHERE namespace = ? AND id = ?",
            (NAMESPACE, "L-1"),
        )
        sqlite_backend._conn.commit()


_SUBSTANTIATING_ASSERTION = {"type": "glob_exists", "target": "pyproject.toml"}


def test_unverified_evidence_level_promotion_is_refused(store: tuple[StorageBackend, MemoryConfig]) -> None:
    """PRD-CORE-312-FR02: a promotion to 'verified' also needs Observed/Verified evidence."""
    result = _update(
        store,
        "L-1",
        confidence="verified",
        evidence_level="inferred",
        assertions=[_SUBSTANTIATING_ASSERTION],
    )

    assert result["status"] == "invalid"
    assert result["reason"] == "verified_requires_observed_or_verified_evidence"
    assert _get(store, "L-1").confidence == "unverified"


def test_evidence_only_downgrade_on_an_already_verified_row_is_refused(
    store: tuple[StorageBackend, MemoryConfig],
) -> None:
    """PRD-CORE-312-FR02 round-1 review: an update that never names ``confidence``
    must still be judged when the ROW is already 'verified' -- an evidence-only
    edit that drops evidence_level to 'inferred' on a verified row is the same
    poisoning shape as promoting straight to verified+inferred. Redesigned (round
    3) as a data invariant in ``_crud_ops.update()``: a NEW violation (this row was
    NOT already violating) is refused regardless of which named field introduced it.
    """
    backend, _cfg = store
    result = _update(
        store,
        "L-1",
        confidence="verified",
        evidence_level="verified",
        assertions=[_SUBSTANTIATING_ASSERTION],
    )
    assert result["status"] == "updated"
    assert _get(store, "L-1").confidence == "verified"

    result = _update(store, "L-1", evidence_level="inferred")

    assert result["status"] == "invalid"
    assert result["reason"] == "verified_requires_observed_or_verified_evidence"
    assert _get(store, "L-1").evidence_level == "verified", "the refused downgrade must not have landed"


def test_legacy_verified_row_can_still_be_retired(store: tuple[StorageBackend, MemoryConfig]) -> None:
    """PRD-CORE-312-FR02 round-2 review: a patch touching neither ``confidence``
    nor ``evidence_level`` must not be refused merely for carrying a PRE-EXISTING
    (legacy) violation forward unchanged -- else a legacy 'verified' row (no
    evidence_level ever recorded -> UNKNOWN) could never be retired
    (``status=obsolete``) or edited again, which would also silently defeat FR04's
    own auto-retraction (``apply_correction`` with only ``status`` set).
    """
    backend, _cfg = store
    _seed_legacy_violation(backend)
    assert _get(store, "L-1").confidence == "unverified", "read-time demotion serves the legacy violation as unverified"

    result = _update(store, "L-1", status="obsolete")

    assert result["status"] == "updated"
    assert _get(store, "L-1").status == "obsolete"


def test_supersedes_closes_the_prior_validity_window(store: tuple[StorageBackend, MemoryConfig]) -> None:
    backend, _cfg = store
    backend.store(MemoryEntry(id="L-new", content="use the retry helper instead", namespace=NAMESPACE))

    result = _update(store, "L-new", supersedes="L-1")

    assert result["changes"] == "supersedes→L-1"
    prior = _get(store, "L-1")
    assert prior.invalid_from is not None
    assert prior.invalidated_by == "L-new"


def test_retired_learning_leaves_default_recall_but_stays_addressable(
    store: tuple[StorageBackend, MemoryConfig],
) -> None:
    backend, cfg = store

    def recalled_ids() -> list[str]:
        with patch("trw_memory.tools.recall.get_local_embedder", return_value=None):
            result = memory_recall_impl(
                "retry flaky network backoff", NAMESPACE, backend=backend, config=cfg, include_org_memories=False
            )
        return [str(row["id"]) for row in cast("list[dict[str, object]]", result["memories"])]

    assert recalled_ids() == ["L-1"]

    assert _update(store, "L-1", status="obsolete")["changes"] == "status→obsolete"

    assert recalled_ids() == []
    obsolete = backend.list_entries(namespace=NAMESPACE, status=MemoryStatus.OBSOLETE)
    assert [entry.id for entry in obsolete] == ["L-1"]


def test_the_mcp_surface_is_registered_over_the_same_function() -> None:
    from trw_memory.lifecycle import correction
    from trw_memory.server import REGISTERED_TOOL_NAMES
    from trw_memory.tools import update

    assert update.apply_correction is correction.apply_correction
    assert "memory_update" in REGISTERED_TOOL_NAMES


def test_a_correction_is_audited_with_what_changed(
    store: tuple[StorageBackend, MemoryConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.tools import update

    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(update, "append_audit_event", lambda _cfg, op, **kw: events.append((op, kw["data"])))

    _update(store, "L-1", impact=0.7)
    _update(store, "L-1")  # nothing named -> no_changes -> no audit event

    assert events == [("update", {"entry_id": "L-1", "changes": "impact→0.7"})]


def test_tags_add_applies_to_the_stored_row_not_a_stale_read(store: tuple[StorageBackend, MemoryConfig]) -> None:
    from trw_memory.lifecycle.correction import Store, apply_correction

    stale = _get(store, "L-1")
    assert _update(store, "L-1", tags_add=["first"])["status"] == "updated"

    result = apply_correction(Store(*store), stale, LearningPatch(tags_add=["second"]))

    assert result["status"] == "updated"
    assert _get(store, "L-1").tags == ["net", "retry", "first", "second"]


def test_concurrent_tags_add_from_two_connections_loses_nothing(store: tuple[StorageBackend, MemoryConfig]) -> None:
    import threading

    from trw_memory.lifecycle.correction import Store, apply_correction

    _backend, cfg = store
    start = threading.Barrier(2)

    def worker(prefix: str) -> None:
        with create_backend_from_config(cfg, NAMESPACE) as own:
            start.wait()
            for n in range(15):
                # Each caller reads, then corrects: the window a lost update needs.
                entry = own.get("L-1", namespace=NAMESPACE)
                assert entry is not None
                apply_correction(Store(own, cfg), entry, LearningPatch(tags_add=[f"{prefix}{n}"]))

    threads = [threading.Thread(target=worker, args=(p,)) for p in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    tags = set(_get(store, "L-1").tags)
    assert {f"{p}{n}" for p in "ab" for n in range(15)} <= tags


def test_a_failed_correction_leaves_a_prior_in_another_store_open(store: tuple[StorageBackend, MemoryConfig]) -> None:
    """The prior closes only after the new learning's write commits; a failure before that leaves it open."""
    from trw_memory.lifecycle.correction import Store, apply_correction

    backend, cfg = store
    backend.store(MemoryEntry(id="L-new", content="use the retry helper instead", namespace=NAMESPACE))
    new_entry = _get(store, "L-new")
    with tempfile.TemporaryDirectory() as other_dir:
        other_cfg = MemoryConfig(storage_backend="sqlite", storage_path=other_dir)
        with create_backend_from_config(other_cfg, "user:other") as other:
            other.store(MemoryEntry(id="L-prior", content="the old way", namespace="user:other"))
            prior = other.get("L-prior", namespace="user:other")

            with (
                patch.object(backend, "update", side_effect=RuntimeError("disk full")),
                pytest.raises(RuntimeError, match="disk full"),
            ):
                apply_correction(
                    Store(backend, cfg),
                    new_entry,
                    LearningPatch(detail="corrected", supersedes="L-prior"),
                    prior=(Store(other, other_cfg), prior),
                )

            reread = other.get("L-prior", namespace="user:other")
            assert reread is not None
            assert reread.invalid_from is None


def test_a_correction_whose_replacement_was_forgotten_leaves_the_prior_open(
    store: tuple[StorageBackend, MemoryConfig],
) -> None:
    """C12 rc7: an empty reread fell back to the caller's stale copy, update()'s None went unchecked, and the
    prior was closed anyway -- ``updated`` for a learning that no longer existed."""
    from trw_memory.lifecycle.correction import Store, apply_correction

    backend, cfg = store
    backend.store(MemoryEntry(id="L-prior", content="the old way", namespace=NAMESPACE))
    backend.store(MemoryEntry(id="L-new", content="use the retry helper instead", namespace=NAMESPACE))
    stale, prior = _get(store, "L-new"), _get(store, "L-prior")
    backend.delete("L-new", namespace=NAMESPACE)  # a forget between the caller's read and the write

    result = apply_correction(
        Store(backend, cfg),
        stale,
        LearningPatch(detail="corrected", supersedes="L-prior"),
        prior=(Store(backend, cfg), prior),
    )

    assert result["status"] == "not_found"
    assert backend.get("L-new", namespace=NAMESPACE) is None
    assert _get(store, "L-prior").invalid_from is None, "the prior was closed by a replacement that is gone"


class _Encoder:
    """A 3-d encoder in one fixed space: every text lands on the same axis, which is all this test reads."""

    model_name = "test-encoder"

    def available(self) -> bool:
        return True

    def embedding_space(self) -> object:
        from trw_memory.embeddings.provenance import EmbeddingSpace

        return EmbeddingSpace("c" * 64, "test-encoder:c", 3)

    def embed(self, text: str) -> list[float]:
        return [0.0, 1.0, 0.0]


@pytest.fixture()
def vec_store() -> Iterator[tuple[StorageBackend, MemoryConfig]]:
    pytest.importorskip("sqlite_vec")
    with tempfile.TemporaryDirectory() as td:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=td, embedding_dim=3)
        with create_backend_from_config(cfg, NAMESPACE) as backend:
            backend.store(MemoryEntry(id="L-1", content="old summary", detail="old detail", namespace=NAMESPACE))
            backend.upsert_vector("L-1", [1.0, 0.0, 0.0], namespace=NAMESPACE)
            yield backend, cfg


def test_a_text_correction_re_encodes_the_live_vector(
    vec_store: tuple[StorageBackend, MemoryConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD-CORE-302 C5: the vector follows the committed text, with provenance for it."""
    monkeypatch.setattr("trw_memory.tools._embedder.get_local_embedder", lambda **_kw: _Encoder())
    backend, _cfg = vec_store

    assert _update(vec_store, "L-1", detail="new detail")["status"] == "updated"

    record = backend.get_vector_records(["L-1"], namespace=NAMESPACE)["L-1"]
    assert list(record.embedding) == [0.0, 1.0, 0.0]
    assert record.provenance is not None


def test_a_text_correction_without_an_embedder_drops_the_stale_vector(
    vec_store: tuple[StorageBackend, MemoryConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("trw_memory.tools._embedder.get_local_embedder", lambda **_kw: None)
    backend, _cfg = vec_store

    assert _update(vec_store, "L-1", summary="new summary")["status"] == "updated"

    assert not backend.vector_exists("L-1", namespace=NAMESPACE)


def test_a_non_text_correction_keeps_the_vector_and_loads_no_model(
    vec_store: tuple[StorageBackend, MemoryConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_load(**_kw: object) -> None:
        raise AssertionError("an impact change must not resolve an embedder")

    monkeypatch.setattr("trw_memory.tools._embedder.get_local_embedder", _no_load)
    backend, _cfg = vec_store

    assert _update(vec_store, "L-1", impact=0.9)["status"] == "updated"

    assert list(backend.get_vector_records(["L-1"], namespace=NAMESPACE)["L-1"].embedding) == [1.0, 0.0, 0.0]


def test_a_stale_if_revision_is_a_conflict_that_writes_nothing(store: tuple[StorageBackend, MemoryConfig]) -> None:
    """PRD-CORE-308: a patch computed from a row that changed since is refused, not applied over it."""
    from trw_memory.lifecycle.correction import revision_of

    stale = revision_of(_get(store, "L-1"))
    assert _update(store, "L-1", tags_add=["landed-first"])["status"] == "updated"
    before = _get(store, "L-1").model_dump()

    result = _update(store, "L-1", detail="computed from the stale row", if_revision=stale)

    assert result["status"] == "conflict"
    assert _get(store, "L-1").model_dump() == before
    current = revision_of(_get(store, "L-1"))
    assert _update(store, "L-1", detail="computed from the current row", if_revision=current)["status"] == "updated"
    assert _get(store, "L-1").detail == "computed from the current row"


def test_a_revision_ignores_recall_counters_and_survives_the_json_wire(
    store: tuple[StorageBackend, MemoryConfig],
) -> None:
    """The daemon hands rows over as JSON; the client's revision must equal the one the server compares."""
    import json

    from trw_memory.lifecycle.correction import revision_of

    entry = _get(store, "L-1")
    wired = MemoryEntry.model_validate_json(json.dumps(entry.model_dump(mode="json")))
    assert revision_of(wired) == revision_of(entry)
    assert revision_of(entry.model_copy(update={"access_count": 9, "recall_count": 4})) == revision_of(entry)
    assert revision_of(entry.model_copy(update={"detail": "edited"})) != revision_of(entry)


def test_if_revision_on_a_backend_without_transactions_is_refused(tmp_path: object) -> None:
    """YAML's transaction() is the no-op default: the compare and the write could not be atomic."""
    from trw_memory.lifecycle.correction import Store, apply_correction, revision_of

    cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))
    with create_backend_from_config(cfg, NAMESPACE) as backend:
        backend.store(MemoryEntry(id="L-y", content="yaml row", detail="kept", namespace=NAMESPACE))
        entry = backend.get("L-y", namespace=NAMESPACE)
        assert entry is not None

        result = apply_correction(Store(backend, cfg), entry, LearningPatch(detail="x", if_revision=revision_of(entry)))

        assert result["status"] == "invalid"
        assert "transactional" in result["error"]
        reread = backend.get("L-y", namespace=NAMESPACE)
        assert reread is not None and reread.detail == "kept"
