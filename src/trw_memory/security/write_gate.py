"""Single write gate for storage surfaces that do not go through ``MemoryClient``.

``MemoryClient.store`` (``_client_store``), ``bulk_store`` (``_client_bulk_store``)
and the MCP ``memory_store`` tool (``tools/store``) each call
:func:`~trw_memory.security.runtime.prepare_entry_for_store` inline before their
``backend.store``. The integration adapters and the ``trw-memory import`` CLI did
not: they called ``backend.store(entry)`` directly, so an entry arriving through
LangChain / CrewAI / LlamaIndex / the VSCode adapter / a JSON import file skipped
the injection-pattern gate, the PII scan, the write rate limit, anomaly scoring
and provenance signing entirely, and was replayed verbatim on every later recall.

:func:`guarded_store` is the shared seam those surfaces now use. It is the ONLY
supported way to persist a caller-supplied entry from outside ``security/``;
``tests/test_store_write_gate_totality.py`` derives every ``.store(...)`` call
site in the production tree and fails on any new one that is neither guarded nor
in its documented exclusion set.

Error contract: security rejections propagate as-is (``PoisoningError``,
``PIIBlockError``, ``RateLimitError``, ``SchemaValidationError``) so each caller
picks raise-vs-skip for its own audience. A *quarantine* decision is not an
error — the entry is diverted to the quarantine store for review and the result
reports ``stored=False``, so a caller can never mistake "held" for "written".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import MemoryQuarantinedError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.runtime import (
    PreparedStoreEntry,
    append_audit_event,
    prepare_entry_for_store,
    store_quarantined_entry,
)

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

__all__ = ["GuardedStoreResult", "guarded_store", "guarded_store_or_raise"]

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class GuardedStoreResult:
    """Outcome of a gated write.

    ``stored`` and ``quarantined`` are separate flags rather than one status
    string so a caller cannot read "not stored" as "nothing happened": a
    quarantined entry is durable in the review store even though it never
    reached the active backend.
    """

    entry: MemoryEntry
    stored: bool
    quarantined: bool
    anomaly_dimension: str = ""
    anomaly_z_score: float = 0.0


def guarded_store(
    backend: StorageBackend,
    entry: MemoryEntry,
    *,
    config: MemoryConfig | None = None,
    session_id: str | None = None,
) -> GuardedStoreResult:
    """Run the SEC-001 store intake, then persist *entry* via *backend*.

    Args:
        backend: Destination backend for an accepted entry.
        config: Config whose paths anchor the audit log, quarantine store and
            provenance key. Defaults to :class:`MemoryConfig`; pass the config
            the backend was built from so security artifacts land beside it.
        session_id: Optional session identity for provenance + audit records.
            It is also the write rate limiter's bucket key, so pass only a real
            caller principal. The chat adapters deliberately pass ``None`` (as
            ``memory_store`` does for an anonymous MCP caller) rather than their
            conversation key: a conversation is not a security principal, and
            bucketing on it would cap a normal chat at
            ``max_memory_writes_per_minute`` turns.

    Returns:
        A :class:`GuardedStoreResult` describing what was persisted where.

    Raises:
        PoisoningError, PIIBlockError, RateLimitError, SchemaValidationError:
            propagated unchanged from the intake pipeline.
    """
    cfg = config or MemoryConfig()
    decision = prepare_entry_for_store(entry, backend=backend, config=cfg, session_id=session_id)
    actor = decision.entry.source_identity or decision.entry.source

    if decision.quarantined:
        store_quarantined_entry(cfg, decision.entry)
        append_audit_event(
            cfg,
            "quarantine",
            entry_id=decision.entry.id,
            actor=actor,
            namespace=decision.entry.namespace,
            data={
                "stored": False,
                "quarantined": True,
                "anomaly_dimension": decision.anomaly_dimension,
                "z_score": decision.anomaly_z_score,
            },
        )
        # Held entries never reach the active store, so without this the caller's
        # only signal is a boolean it may not read. Log at warning level.
        logger.warning(
            "guarded_store_quarantined",
            entry_id=decision.entry.id,
            namespace=decision.entry.namespace,
            anomaly_dimension=decision.anomaly_dimension,
            z_score=decision.anomaly_z_score,
        )
        return GuardedStoreResult(
            entry=decision.entry,
            stored=False,
            quarantined=True,
            anomaly_dimension=decision.anomaly_dimension,
            anomaly_z_score=decision.anomaly_z_score,
        )

    _store_and_audit(backend, decision, cfg, actor=actor, session_id=session_id)
    return GuardedStoreResult(entry=decision.entry, stored=True, quarantined=False)


def guarded_store_or_raise(
    backend: StorageBackend,
    entry: MemoryEntry,
    *,
    config: MemoryConfig | None = None,
    session_id: str | None = None,
) -> GuardedStoreResult:
    """:func:`guarded_store` for callers that cannot report "held" in a return value.

    ``guarded_store`` reports a quarantine by returning ``stored=False`` rather
    than raising, which is right for a caller that can surface the distinction
    (the VSCode adapter puts it in ``status``). A caller whose only return channel
    is ``None`` — the LangChain, CrewAI and LlamaIndex adapters — cannot, and all
    three discarded the result: a quarantined turn vanished from the transcript
    while ``add_messages`` returned normally, which is exactly the "censored
    transcript indistinguishable from a complete one" failure their own docstrings
    called out as the reason to raise.

    This lives here rather than being re-implemented per adapter because three
    hand-rolled copies of one four-line check is how the original divergence
    happened (``docs/documentation/wiring-defect-patterns.md`` P10).

    Raises:
        MemoryQuarantinedError: when the entry was held for review.
        PoisoningError, PIIBlockError, RateLimitError, SchemaValidationError:
            propagated unchanged from the intake pipeline.
    """
    result = guarded_store(backend, entry, config=config, session_id=session_id)
    if result.quarantined:
        raise MemoryQuarantinedError(
            f"memory entry {result.entry.id!r} was quarantined for review, not stored",
            entry_id=result.entry.id,
            anomaly_dimension=result.anomaly_dimension,
        )
    return result


def _store_and_audit(
    backend: StorageBackend,
    decision: PreparedStoreEntry,
    cfg: MemoryConfig,
    *,
    actor: str,
    session_id: str | None,
) -> None:
    backend.store(decision.entry)
    append_audit_event(
        cfg,
        decision.op,
        entry_id=decision.entry.id,
        actor=actor,
        namespace=decision.entry.namespace,
        data={
            "status": "updated" if decision.op == "update" else "stored",
            "session_id": session_id,
            "pii_types": sorted({match.pii_type for match in decision.pii_matches}),
            "quarantined": False,
        },
    )
