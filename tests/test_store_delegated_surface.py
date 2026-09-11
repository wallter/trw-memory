"""PRD-CORE-251 FR03: the parameters that let trw-mcp write through this surface.

``memory_store_impl`` was the only sanctioned write path but could not express
what the trw-mcp learning path writes: the PRD-CORE-110 classification fields,
the PRD-CORE-111 anchors, the SEC-001 provenance anchor directory, and the two
error/enrichment ownership choices a caller with its own singleton connection
and its own corruption-recovery retry has to make. Those were a *signature* gap,
not a model gap -- every field below has been first-class on ``MemoryEntry``
since PRD-CORE-110/111.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import AuthorizationError, StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import Anchor, Confidence, MemoryType, ProtectionTier
from trw_memory.security.audit import AuditLog
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.store import memory_store_impl


@pytest.fixture
def backend(tmp_path: Path) -> SQLiteBackend:
    store = SQLiteBackend(tmp_path / "store.db")
    yield store
    store.close()


@pytest.fixture
def cfg(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_path=str(tmp_path / "mem"))


class TestTypedLearningFields:
    def test_classification_fields_land_on_the_stored_entry(self, backend: SQLiteBackend, cfg: MemoryConfig) -> None:
        result = memory_store_impl(
            "a typed learning",
            "default",
            backend=backend,
            config=cfg,
            entry_id="M-typed-1",
            enrich_after_store=False,
            evidence=["tests/test_store_delegated_surface.py"],
            client_profile="claude-code",
            model_id="opus-5",
            q_value=0.725,
            type=MemoryType.INCIDENT,
            nudge_line="watch the write path",
            confidence=Confidence.VERIFIED,
            task_type="coding",
            domain=["memory"],
            phase_origin="implement",
            phase_affinity=["validate"],
            team_origin="platform",
            protection_tier=ProtectionTier.PROTECTED,
            anchors=[Anchor(file="src/x.py", symbol_name="store_learning")],
            anchor_validity=0.9,
        )

        assert result["status"] == "stored"
        entry = backend.get("M-typed-1", namespace="default")
        assert entry is not None
        assert entry.client_profile == "claude-code"
        assert entry.model_id == "opus-5"
        assert entry.q_value == pytest.approx(0.725)
        assert entry.type == MemoryType.INCIDENT.value
        assert entry.nudge_line == "watch the write path"
        assert entry.confidence == Confidence.VERIFIED.value
        assert entry.task_type == "coding"
        assert entry.domain == ["memory"]
        assert entry.phase_origin == "implement"
        assert entry.phase_affinity == ["validate"]
        assert entry.team_origin == "platform"
        assert entry.protection_tier == ProtectionTier.PROTECTED.value
        assert [a.symbol_name for a in entry.anchors] == ["store_learning"]
        assert entry.anchor_validity == pytest.approx(0.9)

    def test_unsupplied_fields_are_preserved_on_an_update(self, backend: SQLiteBackend, cfg: MemoryConfig) -> None:
        """``None`` means "the caller said nothing", never "reset it".

        The revise branch applies whatever it is handed. Passing the model
        defaults for every unsupplied field would silently wipe an entry's
        classification on any re-store -- including the ``trw-memory-server``
        tool path, which supplies none of them.
        """
        first = memory_store_impl(
            "first version",
            "default",
            backend=backend,
            config=cfg,
            entry_id="M-typed-2",
            enrich_after_store=False,
            # A ``verified`` entry with no evidence is refused by the poisoning
            # gate (min_evidence_items_for_verified), which would make the
            # "update" below a first store and this test vacuous.
            evidence=["tests/test_store_delegated_surface.py"],
            type=MemoryType.INCIDENT,
            confidence=Confidence.VERIFIED,
            protection_tier=ProtectionTier.PROTECTED,
            task_type="coding",
        )
        assert first["status"] == "stored", first

        result = memory_store_impl(
            "second version",
            "default",
            backend=backend,
            config=cfg,
            entry_id="M-typed-2",
            enrich_after_store=False,
        )

        assert result["status"] == "updated"
        entry = backend.get("M-typed-2", namespace="default")
        assert entry is not None
        assert entry.content == "second version"
        assert entry.type == MemoryType.INCIDENT.value
        assert entry.confidence == Confidence.VERIFIED.value
        assert entry.protection_tier == ProtectionTier.PROTECTED.value
        assert entry.task_type == "coding"

    def test_supplied_fields_do_overwrite_on_an_update(self, backend: SQLiteBackend, cfg: MemoryConfig) -> None:
        """Preservation is not stickiness: an explicit value still wins."""
        memory_store_impl(
            "first",
            "default",
            backend=backend,
            config=cfg,
            entry_id="M-typed-3",
            enrich_after_store=False,
            type=MemoryType.INCIDENT,
        )
        memory_store_impl(
            "second",
            "default",
            backend=backend,
            config=cfg,
            entry_id="M-typed-3",
            enrich_after_store=False,
            type=MemoryType.CONVENTION,
        )

        entry = backend.get("M-typed-3", namespace="default")
        assert entry is not None
        assert entry.type == MemoryType.CONVENTION.value


class TestEnrichmentOwnership:
    def test_enrich_after_store_false_skips_embedder_and_graph(
        self, backend: SQLiteBackend, cfg: MemoryConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caller that owns the connection owns the enrichment.

        Without this the trw-mcp store path would load an embedding model it has
        configured OFF and dispatch a graph update against a per-namespace
        database file its own singleton never reads.
        """
        calls: list[str] = []
        monkeypatch.setattr(
            "trw_memory.tools.store.get_local_embedder",
            lambda **_kwargs: calls.append("embedder") or None,  # type: ignore[func-returns-value]
        )
        monkeypatch.setattr(
            "trw_memory.tools.store.schedule_graph_update",
            lambda *_a, **_k: calls.append("graph"),
        )
        monkeypatch.setattr(
            "trw_memory.tools.store.remember_entry_in_tiers",
            lambda *_a, **_k: calls.append("tiers"),
        )

        result = memory_store_impl(
            "no enrichment here",
            "default",
            backend=backend,
            config=cfg,
            entry_id="M-noenrich",
            enrich_after_store=False,
        )

        assert result["status"] == "stored"
        assert calls == []
        assert backend.get("M-noenrich", namespace="default") is not None

    def test_enrichment_runs_by_default(
        self, backend: SQLiteBackend, cfg: MemoryConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-vacuity: the skip above is the flag, not a dead code path."""
        calls: list[str] = []
        monkeypatch.setattr(
            "trw_memory.tools.store.schedule_graph_update",
            lambda *_a, **_k: calls.append("graph"),
        )

        memory_store_impl(
            "enrichment here",
            "default",
            backend=backend,
            config=cfg,
            entry_id="M-enrich",
        )

        assert calls == ["graph"]


class TestErrorOwnership:
    def test_raise_storage_errors_propagates_for_the_recovery_owner(
        self, backend: SQLiteBackend, cfg: MemoryConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """trw-mcp's corruption-recovery retry can only key on the exception.

        Swallowing a ``StorageError`` into ``{"status": "error"}`` erases the
        type, and with it the ability to tell "the database is corrupt, reset the
        singleton and retry" from "the disk is full".
        """

        def _boom(*_a: object, **_k: object) -> None:
            raise StorageError("database disk image is malformed")

        monkeypatch.setattr(backend, "store", _boom)

        with pytest.raises(StorageError):
            memory_store_impl(
                "will fail",
                "default",
                backend=backend,
                config=cfg,
                entry_id="M-raise",
                enrich_after_store=False,
                raise_storage_errors=True,
            )

    def test_storage_errors_still_default_to_a_result_dict(
        self, backend: SQLiteBackend, cfg: MemoryConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: object, **_k: object) -> None:
            raise StorageError("disk full")

        monkeypatch.setattr(backend, "store", _boom)

        result = memory_store_impl(
            "will fail",
            "default",
            backend=backend,
            config=cfg,
            entry_id="M-dict",
            enrich_after_store=False,
        )

        assert result["status"] == "error"


class TestUnauthorizedStoreIsAudited:
    def test_refused_namespace_write_leaves_a_store_rejected_event(self, tmp_path: Path) -> None:
        """PRD-CORE-251 FR03: the refusal is recorded, not just raised.

        ``require_namespace_permission`` raises without writing anything, so
        before this the only refusal in the whole store path that produced no
        audit trail was the access-control one.
        """
        cfg = MemoryConfig(
            storage_path=str(tmp_path / "mem"),
            rbac_enabled=True,
            namespace_roles={"default": "reader"},
        )
        store = SQLiteBackend(tmp_path / "store.db")
        try:
            with pytest.raises(AuthorizationError):
                memory_store_impl(
                    "refused",
                    "default",
                    backend=store,
                    config=cfg,
                    entry_id="M-denied",
                    enrich_after_store=False,
                )
        finally:
            store.close()

        records = AuditLog(Path(cfg.audit_log_path)).read_all()
        rejected = [r for r in records if r.op == "store_rejected"]
        assert rejected, "an unauthorized store wrote no audit event"
        assert rejected[-1].data["reason"] == "unauthorized"
        assert rejected[-1].id == "M-denied"
