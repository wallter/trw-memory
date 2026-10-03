"""Pin the SEMANTIC stage order of the store-intake pipeline.

The intake pipeline (_runtime_pipeline.py) encodes its check order as data — two
ordered stage lists — because the order is load-bearing (PII must precede
provenance hashing; the rate-limit..provenance slice must sit inside one audited
try). These tests fail if a future edit reorders a stage. The trust-quarantine short-circuit and the anomaly
stage were removed with the intake trust scorer and the anomaly quarantine (UF-MEM-03, 2026-10-01).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security import _runtime_pipeline as pipeline
from trw_memory.security import runtime


def test_declared_stage_order_is_pinned() -> None:
    """The ordered stage lists encode the exact semantic sequence."""
    assert [stage.__name__ for stage in pipeline._PRE_AUDIT_STAGES] == [
        "_stage_queue_drain",
        "_stage_classify",
        "_stage_flag_code",
        "_stage_verify_defaults",
    ]
    assert [stage.__name__ for stage in pipeline._AUDITED_STAGES] == [
        "_stage_rate_limit",
        "_stage_validate_payload",
        "_stage_pii_policy",
        "_stage_provenance_hash",
    ]
    # The two invariants a naive reorder most often breaks, asserted by index.
    audited = [stage.__name__ for stage in pipeline._AUDITED_STAGES]
    assert audited.index("_stage_pii_policy") < audited.index("_stage_provenance_hash")
    assert audited.index("_stage_rate_limit") < audited.index("_stage_validate_payload")


def _record_stage_calls(monkeypatch: pytest.MonkeyPatch, order: list[str]) -> None:
    """Wrap each delegate a stage calls so it appends its label at call time."""

    def wrap(module: object, name: str, label: str) -> None:
        original = getattr(module, name)

        def wrapped(*args: object, **kwargs: object) -> object:
            order.append(label)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapped)

    # queue-drain + rate-limit are reached through the runtime facade (_rt()).
    wrap(runtime, "ensure_security_maintenance", "queue_drain")
    wrap(runtime, "enforce_write_rate_limit", "rate_limit")
    # the remaining delegates are module globals resolved inside each stage.
    wrap(pipeline, "_actor_for_entry", "classify")
    wrap(pipeline, "_flag_code_snippet", "flag")
    wrap(pipeline, "_apply_sec001_intake", "verify_defaults")
    wrap(pipeline, "validate_entry_payload", "validate")
    wrap(pipeline, "_apply_runtime_pii_policy", "pii")
    wrap(pipeline, "_apply_provenance_hash", "provenance")


def test_runtime_stage_execution_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A live store runs the stages in the pinned order; PII before provenance."""
    order: list[str] = []
    _record_stage_calls(monkeypatch, order)
    cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))

    with create_backend_from_config(cfg, "project:default") as backend:
        prepared = pipeline.prepare_entry_for_store(
            MemoryEntry(id="M-1", content="hello", namespace="project:default"),
            backend=backend,
            config=cfg,
            session_id="s1",
        )

    assert prepared.op == "store"
    # Every stage ran; relative ordering is the load-bearing assertion.
    for label in (
        "queue_drain",
        "classify",
        "flag",
        "verify_defaults",
        "rate_limit",
        "validate",
        "pii",
        "provenance",
    ):
        assert label in order, f"stage {label!r} did not run"
    assert order.index("classify") < order.index("flag") < order.index("verify_defaults")
    assert order.index("verify_defaults") < order.index("rate_limit")
    assert order.index("rate_limit") < order.index("validate") < order.index("pii")
    # PRD-DIST-2046 c793: provenance hash MUST follow PII redaction.
    assert order.index("pii") < order.index("provenance")


def test_a_caller_set_quarantined_flag_never_skips_the_audited_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the short-circuit gone, a ``quarantined`` metadata key (stripped anyway) changes nothing."""
    order: list[str] = []
    _record_stage_calls(monkeypatch, order)
    cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
    with create_backend_from_config(cfg, "project:default") as backend:
        prepared = pipeline.prepare_entry_for_store(
            MemoryEntry(id="M-held", content="note", namespace="project:default", metadata={"quarantined": "true"}),
            backend=backend,
            config=cfg,
            session_id="s1",
        )
    assert prepared.quarantined is False
    assert "quarantined" not in prepared.entry.metadata
    assert {"rate_limit", "validate", "pii", "provenance"} <= set(order)


def test_store_rejected_audit_wraps_the_audited_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stage raising inside the audited slice emits store_rejected then re-raises."""
    from trw_memory.exceptions import RateLimitError

    audited: list[str] = []

    def boom_rate_limit(*args: object, **kwargs: object) -> None:
        raise RateLimitError("nope", retry_after=42.0)

    def record_audit(config: object, op: str, **kwargs: object) -> None:
        audited.append(op)

    monkeypatch.setattr(runtime, "enforce_write_rate_limit", boom_rate_limit)
    monkeypatch.setattr(runtime, "append_audit_event", record_audit)

    cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
    with create_backend_from_config(cfg, "project:default") as backend:
        with pytest.raises(RateLimitError):
            pipeline.prepare_entry_for_store(
                MemoryEntry(id="M-1", content="x", namespace="project:default"),
                backend=backend,
                config=cfg,
                session_id="s1",
            )

    assert audited == ["store_rejected"]
