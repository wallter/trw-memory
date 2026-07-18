"""Pin the SEMANTIC stage order of the store-intake pipeline.

The intake pipeline (_runtime_pipeline.py) encodes its check order as data — two
ordered stage lists — because the order is load-bearing (PII must precede
provenance hashing; the trust-quarantine short-circuit must skip
rate-limit/PII/anomaly; the rate-limit..anomaly slice must sit inside one audited
try). These tests fail if a future edit reorders a stage or lets the
short-circuit fall through into the audited stages.
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
    assert [stage.__name__ for stage in pipeline._PRE_QUARANTINE_STAGES] == [
        "_stage_queue_drain",
        "_stage_classify",
        "_stage_flag_code",
        "_stage_trust_intake",
    ]
    assert [stage.__name__ for stage in pipeline._AUDITED_STAGES] == [
        "_stage_rate_limit",
        "_stage_validate_payload",
        "_stage_pii_policy",
        "_stage_provenance_hash",
        "_stage_anomaly_score",
    ]
    # The two invariants a naive reorder most often breaks, asserted by index.
    audited = [stage.__name__ for stage in pipeline._AUDITED_STAGES]
    assert audited.index("_stage_pii_policy") < audited.index("_stage_provenance_hash")
    assert audited.index("_stage_rate_limit") < audited.index("_stage_anomaly_score")


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
    wrap(pipeline, "_apply_sec001_intake", "trust_intake")
    wrap(pipeline, "validate_entry_payload", "validate")
    wrap(pipeline, "_apply_runtime_pii_policy", "pii")
    wrap(pipeline, "_apply_provenance_hash", "provenance")
    wrap(pipeline, "_score_entry_anomaly", "anomaly")


def test_runtime_stage_execution_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A live store runs the stages in the pinned order; PII before provenance."""
    order: list[str] = []
    _record_stage_calls(monkeypatch, order)
    cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), poisoning_detection_enabled=False)

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
        "trust_intake",
        "rate_limit",
        "validate",
        "pii",
        "provenance",
        "anomaly",
    ):
        assert label in order, f"stage {label!r} did not run"
    assert order.index("classify") < order.index("flag") < order.index("trust_intake")
    assert order.index("trust_intake") < order.index("rate_limit")
    assert order.index("rate_limit") < order.index("validate") < order.index("pii")
    # PRD-DIST-2046 c793: provenance hash MUST follow PII redaction.
    assert order.index("pii") < order.index("provenance") < order.index("anomaly")


def test_trust_quarantine_short_circuits_before_audited_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trust-score quarantine signs provenance but SKIPS rate-limit/PII/anomaly."""
    order: list[str] = []
    _record_stage_calls(monkeypatch, order)

    # Force the trust-intake stage to return a quarantined entry.
    original_intake = pipeline._apply_sec001_intake

    def quarantining_intake(entry: MemoryEntry, **kwargs: object) -> MemoryEntry:
        order.append("trust_intake")
        return entry.model_copy(update={"metadata": {**entry.metadata, "quarantined": "true"}})

    monkeypatch.setattr(pipeline, "_apply_sec001_intake", quarantining_intake)

    cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), poisoning_detection_enabled=False)
    with create_backend_from_config(cfg, "project:default") as backend:
        prepared = pipeline.prepare_entry_for_store(
            MemoryEntry(id="M-held", content="suspicious", namespace="project:default"),
            backend=backend,
            config=cfg,
            session_id="s1",
        )

    assert prepared.quarantined is True
    assert prepared.anomaly_dimension == "trust_score"
    # Provenance still signs on the short-circuit path...
    assert "provenance" in order
    # ...but the audited stages are skipped for a held entry.
    assert "rate_limit" not in order
    assert "validate" not in order
    assert "pii" not in order
    assert "anomaly" not in order
    # unused so mypy/ruff keep the reference meaningful
    assert callable(original_intake)


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

    cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), poisoning_detection_enabled=False)
    with create_backend_from_config(cfg, "project:default") as backend:
        with pytest.raises(RateLimitError):
            pipeline.prepare_entry_for_store(
                MemoryEntry(id="M-1", content="x", namespace="project:default"),
                backend=backend,
                config=cfg,
                session_id="s1",
            )

    assert audited == ["store_rejected"]
