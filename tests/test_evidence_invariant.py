"""PRD-CORE-312 FR02, redesigned as a data invariant (round-3 review).

The first cut put the ``confidence='verified'`` requires Observed/Verified
evidence rule at ONE write chokepoint (``poisoning.reject_unsubstantiated_verified``),
reached only from the store and update paths. Three review rounds each found a new
bypass on that boundary; round 3's own finding was that consolidation's and sync's
direct ``backend.store()``/``backend.update()`` calls never reached it at all. This
suite proves the invariant now lives in the ONE place every entry-producing path
already goes through: the SQLite CRUD layer itself.

Soundness scope: proves no persisted or served entry claims verified without
observed/verified evidence; does NOT prove the evidence content itself is true.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from trw_memory.exceptions import SchemaValidationError
from trw_memory.models.memory import MemoryEntry
from trw_memory.security._evidence_invariant import refuse_new_violation, served_view, violates_evidence_invariant
from trw_memory.storage.sqlite_backend import SQLiteBackend

NAMESPACE = "project:default"


def _entry(entry_id: str, **kwargs: object) -> MemoryEntry:
    return MemoryEntry(id=entry_id, namespace=NAMESPACE, **{"content": "a claim", **kwargs})  # type: ignore[arg-type]


class TestViolatesEvidenceInvariant:
    def test_verified_with_unknown_evidence_violates(self) -> None:
        assert violates_evidence_invariant(_entry("M-1", confidence="verified")) is True

    def test_verified_with_inferred_evidence_violates(self) -> None:
        assert violates_evidence_invariant(_entry("M-1", confidence="verified", evidence_level="inferred")) is True

    @pytest.mark.parametrize("level", ["observed", "verified"])
    def test_verified_with_observed_or_verified_evidence_does_not_violate(self, level: str) -> None:
        assert violates_evidence_invariant(_entry("M-1", confidence="verified", evidence_level=level)) is False

    def test_unverified_confidence_never_violates(self) -> None:
        assert violates_evidence_invariant(_entry("M-1", confidence="unverified")) is False


class TestRefuseNewViolation:
    def test_a_brand_new_violating_entry_is_refused(self) -> None:
        with pytest.raises(SchemaValidationError) as excinfo:
            refuse_new_violation(None, _entry("M-1", confidence="verified"))
        assert excinfo.value.reason == "verified_requires_observed_or_verified_evidence"

    def test_a_brand_new_valid_entry_is_accepted(self) -> None:
        refuse_new_violation(None, _entry("M-1", confidence="verified", evidence_level="observed"))  # no raise

    def test_an_edit_introducing_a_new_violation_is_refused(self) -> None:
        """The round-1 finding: an evidence-only downgrade on an already-valid verified row."""
        existing = _entry("M-1", confidence="verified", evidence_level="verified")
        edited = existing.model_copy(update={"evidence_level": "inferred"})
        with pytest.raises(SchemaValidationError):
            refuse_new_violation(existing, edited)

    def test_an_edit_carrying_forward_a_pre_existing_violation_is_allowed(self) -> None:
        """The round-2 finding: a legacy violating row must still be editable."""
        existing = _entry("M-1", confidence="verified")  # legacy: evidence_level defaults to unknown
        edited = existing.model_copy(update={"status": "obsolete"})
        refuse_new_violation(existing, edited)  # no raise

    @pytest.mark.parametrize(
        "change",
        [{"content": "an unrelated new claim"}, {"detail": "new supporting prose"}],
        ids=["content", "detail"],
    )
    def test_a_new_claim_under_a_legacy_id_is_refused(self, change: dict[str, str]) -> None:
        """Delta review v2 round 2: the exemption covers the SAME claim only."""
        existing = _entry("M-1", confidence="verified")
        with pytest.raises(SchemaValidationError):
            refuse_new_violation(existing, existing.model_copy(update={**change, "evidence_level": "inferred"}))


class TestServedView:
    def test_a_violating_entry_is_served_demoted_with_a_marker(self) -> None:
        entry = _entry("M-1", confidence="verified")
        served = served_view(entry)
        assert served.confidence == "unverified"
        assert served.metadata["served_confidence_demoted"] == "true"

    def test_served_view_never_mutates_the_original(self) -> None:
        entry = _entry("M-1", confidence="verified")
        served_view(entry)
        assert entry.confidence == "verified", "served_view must return a copy, never rewrite in place"

    def test_a_valid_entry_is_served_unchanged(self) -> None:
        entry = _entry("M-1", confidence="verified", evidence_level="observed")
        assert served_view(entry) is entry


class TestStorageLayerCensus:
    """Q2: every entry-producing path funnels through ``SQLiteBackend.store``/``update``,
    so exercising those two directly proves consolidation, sync and any other current or
    future writer inherits the invariant without its own bespoke check."""

    @pytest.fixture()
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(tmp_path / "m.db")

    def test_backend_store_refuses_a_new_violating_entry(self, backend: SQLiteBackend) -> None:
        """Proves the round-3 finding is closed: consolidation's and sync's own
        direct ``backend.store(entry)`` calls (not just ``memory_store_impl``) hit
        this same refusal, with no per-caller opt-in required."""
        entry = _entry("M-consolidation", confidence="verified", evidence=["a citation"])
        with pytest.raises(SchemaValidationError) as excinfo:
            backend.store(entry)
        assert excinfo.value.reason == "verified_requires_observed_or_verified_evidence"
        assert backend.get("M-consolidation", namespace=NAMESPACE) is None, "the refused write must not have landed"

    def test_backend_store_accepts_a_new_valid_entry(self, backend: SQLiteBackend) -> None:
        entry = _entry("M-1", confidence="verified", evidence_level="observed")
        backend.store(entry)
        stored = backend.get("M-1", namespace=NAMESPACE)
        assert stored is not None
        assert stored.confidence == "verified"

    def test_backend_store_replacing_an_existing_violation_does_not_refuse_an_unrelated_change(
        self, backend: SQLiteBackend
    ) -> None:
        """A sync pull replacing a row via INSERT OR REPLACE, where BOTH the
        incoming and the existing copy already carry the same legacy violation,
        must not be refused merely for the violation it did not introduce."""
        sqlite_backend = cast("Any", backend)
        with sqlite_backend._lock:
            sqlite_backend._conn.execute(
                "INSERT INTO memories (id, namespace, content, confidence, evidence_level, created_at, updated_at) "
                "VALUES ('M-legacy', ?, 'a claim', 'verified', 'unknown', '2026-01-01T00:00:00+00:00', "
                "'2026-01-01T00:00:00+00:00')",
                (NAMESPACE,),
            )
            sqlite_backend._conn.commit()
        incoming = _entry("M-legacy", confidence="verified", tags=["synced-from-peer"])
        backend.store(incoming)  # no raise: the same claim, only its tags changed
        assert backend.get("M-legacy", namespace=NAMESPACE) is not None
        with pytest.raises(SchemaValidationError):
            backend.store(_entry("M-legacy", confidence="verified", detail="synced from a peer"))

    def test_backend_update_refuses_a_new_violation(self, backend: SQLiteBackend) -> None:
        backend.store(_entry("M-1", confidence="unverified"))
        with pytest.raises(SchemaValidationError) as excinfo:
            backend.update("M-1", namespace=NAMESPACE, confidence="verified")
        assert excinfo.value.reason == "verified_requires_observed_or_verified_evidence"

    def test_backend_update_allows_a_legacy_violation_to_be_retired(self, backend: SQLiteBackend) -> None:
        sqlite_backend = cast("Any", backend)
        backend.store(_entry("M-1", confidence="unverified"))
        with sqlite_backend._lock:
            sqlite_backend._conn.execute(
                "UPDATE memories SET confidence = 'verified' WHERE namespace = ? AND id = ?", (NAMESPACE, "M-1")
            )
            sqlite_backend._conn.commit()
        backend.update("M-1", namespace=NAMESPACE, status="obsolete")  # no raise
        assert backend.get("M-1", namespace=NAMESPACE).status == "obsolete"  # type: ignore[union-attr]

    def test_backend_get_serves_a_legacy_violation_demoted(self, backend: SQLiteBackend) -> None:
        """READ time: storage is never rewritten, but the served copy is honest."""
        sqlite_backend = cast("Any", backend)
        backend.store(_entry("M-1", confidence="unverified"))
        with sqlite_backend._lock:
            sqlite_backend._conn.execute(
                "UPDATE memories SET confidence = 'verified' WHERE namespace = ? AND id = ?", (NAMESPACE, "M-1")
            )
            sqlite_backend._conn.commit()
        served = backend.get("M-1", namespace=NAMESPACE)
        assert served is not None
        assert served.confidence == "unverified"
        assert served.metadata.get("served_confidence_demoted") == "true"


def _legacy_violation(backend: SQLiteBackend, entry_id: str = "M-1") -> None:
    """Plant a pre-invariant row: verified confidence, unknown evidence, straight into SQL."""
    sqlite_backend = cast("Any", backend)
    backend.store(_entry(entry_id, content=f"legacy claim {entry_id}", confidence="unverified"))
    with sqlite_backend._lock:
        sqlite_backend._conn.execute(
            "UPDATE memories SET confidence = 'verified' WHERE namespace = ? AND id = ?", (NAMESPACE, entry_id)
        )
        sqlite_backend._conn.commit()


_SERVED_READS: dict[str, Any] = {
    "search": lambda b: b.search("legacy", namespace=NAMESPACE),
    "search_keyword_tokens": lambda b: b.search("legacy", keyword_tokens=["legacy"], namespace=NAMESPACE),
    "list_entries": lambda b: b.list_entries(namespace=NAMESPACE),
    "list_entries_by_id": lambda b: b.list_entries_by_id(namespace=NAMESPACE),
    "update_return": lambda b: [b.update("M-1", namespace=NAMESPACE, importance=0.9)],
}


class TestEveryServedReadDemotes:
    """Round-1 finding 1: demotion lived only in ``get()``; search/list returned the raw row."""

    @pytest.fixture()
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(tmp_path / "m.db")

    @pytest.mark.parametrize("read", list(_SERVED_READS), ids=list(_SERVED_READS))
    def test_a_legacy_violation_is_served_demoted(self, backend: SQLiteBackend, read: str) -> None:
        _legacy_violation(backend)
        served = [e for e in _SERVED_READS[read](backend) if e.id == "M-1"]
        assert len(served) == 1, served
        assert served[0].confidence == "unverified"
        assert served[0].metadata.get("served_confidence_demoted") == "true"

    @pytest.mark.parametrize("read", list(_SERVED_READS), ids=list(_SERVED_READS))
    def test_a_substantiated_verified_row_is_served_unchanged(self, backend: SQLiteBackend, read: str) -> None:
        backend.store(_entry("M-1", content="legacy claim M-1", confidence="verified", evidence_level="observed"))
        served = [e for e in _SERVED_READS[read](backend) if e.id == "M-1"]
        assert [(e.confidence, "served_confidence_demoted" in e.metadata) for e in served] == [("verified", False)]


class TestNoWriteSeamSkipsTheInvariant:
    @pytest.fixture()
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(tmp_path / "m.db")

    def test_caller_supplied_sync_fields_do_not_skip_the_update_check(self, backend: SQLiteBackend) -> None:
        """Round-1 finding 2: the check sat inside the skip-sync-bookkeeping branch."""
        backend.store(_entry("M-1", confidence="unverified"))
        with pytest.raises(SchemaValidationError):
            backend.update("M-1", namespace=NAMESPACE, confidence="verified", sync_seq=1)
        raw = cast("Any", backend)._conn.execute("SELECT confidence FROM memories WHERE id = 'M-1'").fetchone()
        assert raw[0] == "unverified"

    def test_store_many_refuses_a_new_violation_and_writes_nothing(self, backend: SQLiteBackend) -> None:
        batch = [_entry("M-ok", confidence="unverified"), _entry("M-bad", confidence="verified")]
        with pytest.raises(SchemaValidationError):
            backend.store_many(batch)
        assert backend.count(NAMESPACE) == 0

    def test_store_many_accepts_substantiated_and_carried_forward_rows(self, backend: SQLiteBackend) -> None:
        _legacy_violation(backend, "M-legacy")
        batch = [
            _entry("M-new", confidence="verified", evidence_level="verified"),
            _entry("M-legacy", content="legacy claim M-legacy", confidence="verified", tags=["re-synced"]),
        ]
        assert backend.store_many(batch) == 2

    def test_store_many_reads_the_replaced_rows_inside_the_write_transaction(
        self, backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CORE-312-STORE-MANY-TXN: the carried-forward check read the row BEFORE ``BEGIN IMMEDIATE``, so a
        concurrent writer could change it between the check and the replace. The read must see the
        state the write then replaces: inside the transaction, holding the backend lock."""
        from trw_memory.storage import _crud_ops

        _legacy_violation(backend, "M-legacy")
        real_get = _crud_ops.get
        seen: list[tuple[bool, bool]] = []

        def spy(target: SQLiteBackend, *args: Any, **kwargs: Any) -> Any:
            conn = cast("Any", target)._conn
            seen.append((conn.in_transaction, cast("Any", target)._lock._is_owned()))
            return real_get(target, *args, **kwargs)

        monkeypatch.setattr(_crud_ops, "get", spy)
        batch = [_entry("M-legacy", content="legacy claim M-legacy", confidence="verified", tags=["re-synced"])]
        assert backend.store_many(batch) == 1
        assert seen == [(True, True)]

    def test_store_many_refusal_inside_the_transaction_rolls_back_and_frees_the_connection(
        self, backend: SQLiteBackend
    ) -> None:
        batch = [_entry("M-ok", confidence="unverified"), _entry("M-bad", confidence="verified")]
        before = [(e.sync_seq, e.sync_hash, e.last_synced_at) for e in batch]
        with pytest.raises(SchemaValidationError):
            backend.store_many(batch)
        assert [(e.sync_seq, e.sync_hash, e.last_synced_at) for e in batch] == before  # caller's entries untouched
        assert cast("Any", backend)._conn.in_transaction is False
        assert backend.store_many([_entry("M-after", confidence="unverified")]) == 1


@pytest.mark.parametrize("read", ["dirty_page", "find"])
def test_sync_served_rows_are_demoted(tmp_path: Path, read: str) -> None:
    """Delta review v2 round 2: the daemon's sync responses deserialized raw rows."""
    from trw_memory.sync.delta import DeltaTracker, find_synced_entry

    backend = SQLiteBackend(tmp_path / "m.db")
    _legacy_violation(backend)
    if read == "dirty_page":
        served = DeltaTracker.get_dirty_entries(backend, namespace=NAMESPACE)
    else:
        served = [find_synced_entry(backend, NAMESPACE, "no-remote-id", ["M-1"])]  # type: ignore[list-item]
    assert [(e.id, e.confidence) for e in served] == [("M-1", "unverified")]
    raw = cast("Any", backend)._conn.execute("SELECT confidence FROM memories WHERE id = 'M-1'").fetchone()
    assert raw[0] == "verified"  # served, never rewritten


class TestYamlBackendParity:
    """Round-1 finding 3: ``MEMORY_STORAGE_BACKEND=yaml`` bypassed both halves."""

    @pytest.fixture()
    def backend(self, tmp_path: Path) -> Any:
        from trw_memory.storage.yaml_backend import YAMLBackend

        return YAMLBackend(tmp_path / "entries")

    def test_store_refuses_a_new_violation(self, backend: Any) -> None:
        with pytest.raises(SchemaValidationError):
            backend.store(_entry("M-1", confidence="verified"))
        assert backend.get("M-1", namespace=NAMESPACE) is None

    def test_update_refuses_a_new_violation_before_writing(self, backend: Any) -> None:
        backend.store(_entry("M-1", confidence="unverified"))
        with pytest.raises(SchemaValidationError):
            backend.update("M-1", confidence="verified")
        assert backend.get("M-1", namespace=NAMESPACE).confidence == "unverified"

    def test_a_legacy_violation_is_served_demoted_and_still_editable(self, backend: Any) -> None:
        import yaml

        backend.store(_entry("M-1", confidence="unverified"))
        path = backend._path("M-1")
        data = yaml.safe_load(path.read_text())
        data["confidence"] = "verified"
        path.write_text(yaml.safe_dump(data))
        for served in (backend.get("M-1", namespace=NAMESPACE), *backend.list_entries(namespace=NAMESPACE)):
            assert served.confidence == "unverified"
            assert served.metadata.get("served_confidence_demoted") == "true"
        assert backend.update("M-1", status="obsolete").status == "obsolete"  # carried forward, not refused
        assert yaml.safe_load(path.read_text())["confidence"] == "verified"  # storage never rewritten by a read


def test_reject_unsubstantiated_verified_no_longer_owns_the_evidence_level_axis() -> None:
    """The evidence_level check is deleted from the old chokepoint (Q4: delete the
    chokepoint-specific patches this redesign replaces) -- an entry that passes the
    artifact-count rule but violates evidence_level is NOT rejected by this function
    any more; ``_evidence_invariant`` is now the sole owner of that axis."""
    from trw_memory.models.memory import Assertion, AssertionType
    from trw_memory.security.poisoning import reject_unsubstantiated_verified

    entry = _entry(
        "M-1",
        confidence="verified",
        evidence_level="inferred",
        assertions=[Assertion(type=AssertionType.GLOB_EXISTS, target="pyproject.toml")],
    )
    reject_unsubstantiated_verified(entry, min_items=1)  # no raise: artifact present, evidence_level not its concern
