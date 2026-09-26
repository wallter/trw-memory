"""MCP tool: memory_store — persist a new memory entry.

Validates namespace, creates a MemoryEntry with a unique M-prefixed ID,
stores it via the backend, and returns the memory_id and status.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import structlog
from pydantic import BaseModel, ConfigDict

from trw_memory._client_store import _existing_entry_for_namespace, _revision
from trw_memory.daemon._offload import run_offloaded
from trw_memory.embeddings import get_local_embedder, keyword_only_on_refusal
from trw_memory.embeddings.provenance import generation_provenance_kwargs
from trw_memory.exceptions import (
    AuthorizationError,
    ConfigError,
    MemoryNotFoundError,
    PIIBlockError,
    PoisoningError,
    RateLimitError,
    SchemaValidationError,
    StorageError,
)
from trw_memory.graph import schedule_graph_update
from trw_memory.lifecycle.tiers._runtime import (
    embedding_has_consumer,
    remember_entry_in_tiers,
    supports_tier_runtime,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.models.entry_factory import local_node_id_for, new_entry, revise_entry
from trw_memory.models.memory import Anchor, Assertion, Confidence, MemoryStatus, MemoryType, ProtectionTier
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.security.poisoning import validate_store_inputs
from trw_memory.security.rbac import Permission, require_namespace_permission
from trw_memory.security.runtime import append_audit_event, prepare_entry_for_store, store_quarantined_entry
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools._types import McpServer

logger = structlog.get_logger(__name__)


def memory_store_impl(
    content: str,
    namespace: str,
    *,
    backend: StorageBackend,
    tags: list[str] | None = None,
    importance: float = 0.5,
    detail: str = "",
    metadata: dict[str, str] | None = None,
    config: MemoryConfig | None = None,
    source: Literal["human", "agent", "tool", "consolidated", "team_sync", "company_sync"] = "tool",
    source_identity: str = "",
    session_id: str | None = None,
    entry_id: str | None = None,
    evidence: list[str] | None = None,
    expires: str = "",
    assertions: list[Assertion] | None = None,
    client_profile: str | None = None,
    model_id: str | None = None,
    type: MemoryType | str | None = None,
    nudge_line: str | None = None,
    confidence: Confidence | str | None = None,
    task_type: str | None = None,
    domain: list[str] | None = None,
    phase_origin: str | None = None,
    phase_affinity: list[str] | None = None,
    team_origin: str | None = None,
    protection_tier: ProtectionTier | str | None = None,
    anchors: list[Anchor] | None = None,
    anchor_validity: float | None = None,
    trw_dir: Path | None = None,
) -> dict[str, object]:
    """Core implementation of memory_store (callable without MCP).

    Args:
        content: Core knowledge statement to store. Must be non-empty.
        namespace: Namespace scope (e.g., "project:default", "global").
        backend: Storage backend instance.
        tags: Optional list of tags to associate with the entry.
        importance: Importance score in [0.0, 1.0]. Defaults to 0.5.
        detail: Extended explanation or context. Defaults to "".
        metadata: Optional string key-value metadata. Defaults to {}.
        evidence: Optional source references supporting the entry.
        expires: Optional expiration date or condition.
        assertions: Optional machine-verifiable grounding assertions.
        client_profile: Writer's client profile, when known.
        model_id: Writer's model identifier, when known.
        type: Entry classification (PRD-CORE-110).
        nudge_line: Short nudge text rendered from this entry.
        confidence: Validation confidence (PRD-CORE-110).
        task_type: Task-type identifier the entry was learned under.
        domain: Domain tags.
        phase_origin: Phase the entry was learned in.
        phase_affinity: Phases the entry is most useful in.
        team_origin: Team identifier.
        protection_tier: Lifecycle protection level.
        anchors: Code-symbol anchors (PRD-CORE-111).
        anchor_validity: Computed anchor-validity score.
        trw_dir: Ceremony directory the SEC-001 intake anchors provenance to.

    Returns:
        {"memory_id": str, "status": "stored", "namespace": str}; a refusal is a
        status, never a raise: "invalid" (schema), "blocked" (PII, poisoning),
        "rate_limited" (retry after ``retry_after`` seconds), "conflict" (the row changed
        while the store was prepared; retry), "error" (storage).
        Only an authorization refusal raises.

    The status vocabulary this returns is mapped -- explicitly and under test --
    onto the trw-mcp learning vocabulary by
    ``trw_mcp.state.memory_adapter._STORE_STATUS_TO_LEARNING_STATUS``. A new
    status added here without a mapping entry there is a data-loss bug, not a
    cosmetic one: trw-mcp suppresses its YAML sidecar on exactly those statuses.
    """
    # Validate namespace
    try:
        validate_namespace(namespace)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}
    cfg = config or MemoryConfig()
    try:
        require_namespace_permission(cfg, namespace, Permission.WRITE, "store")
    except AuthorizationError:
        # The permission helper raises without leaving a trace, so a refused
        # write was invisible to the audit log that exists to record exactly
        # that (PRD-CORE-251 FR03). The raise is UNCONDITIONAL -- an
        # authorization refusal is the one failure no caller may downgrade into
        # a result dict, and every existing caller already sees it raise.
        append_audit_event(
            cfg,
            "store_rejected",
            entry_id=entry_id or "",
            actor=source_identity or source,
            namespace=namespace,
            data={"reason": "unauthorized", "permission": Permission.WRITE.value, "session_id": session_id},
        )
        raise
    try:
        validate_store_inputs(
            content=content, detail=detail, tags=tags, metadata=metadata, importance=importance, assertions=assertions
        )
    except SchemaValidationError as exc:
        append_audit_event(
            cfg,
            "store_rejected",
            entry_id=entry_id or "",
            actor=source_identity or source,
            namespace=namespace,
            data={"reason": "schema_invalid", "failed_fields": exc.failed_fields, "session_id": session_id},
        )
        return {"error": str(exc), "status": "invalid", "namespace": namespace}

    entry_id = entry_id or ("M-" + uuid4().hex[:16])
    now = datetime.now(timezone.utc)
    try:
        existing = _existing_entry_for_namespace(backend, entry_id, namespace)
    except MemoryNotFoundError as exc:
        return {"error": str(exc), "status": "not_found", "namespace": namespace}
    entry_metadata = dict(existing.metadata) if existing is not None else {}
    entry_metadata.update(metadata or {})
    entry_expires = expires or (existing.expires if existing is not None else "")
    # PRD-CORE-245 FR08: through the shared factory, not a bare constructor.
    # This surface is what ``trw-memory-server`` writes through, and it used to
    # leave ``vector_clock`` at its ``{}`` default -- which makes an org-shared
    # pull discard a newer local edit rather than merge it.
    local_node_id = local_node_id_for(cfg.storage_path)
    # Optional entry facets. ``None`` means "the caller said nothing about this
    # field", which must NOT overwrite what an existing row already carries --
    # that is the difference between an update and a silent reset.
    facets: dict[str, object] = {
        name: value
        for name, value in (
            ("client_profile", client_profile),
            ("model_id", model_id),
            ("type", type),
            ("nudge_line", nudge_line),
            ("confidence", confidence),
            ("task_type", task_type),
            ("domain", domain),
            ("phase_origin", phase_origin),
            ("phase_affinity", phase_affinity),
            ("team_origin", team_origin),
            ("protection_tier", protection_tier),
            ("anchors", anchors),
            ("anchor_validity", anchor_validity),
        )
        if value is not None
    }
    if existing is None:
        entry = new_entry(
            entry_id=entry_id,
            content=content.strip(),
            namespace=namespace,
            local_node_id=local_node_id,
            now=now,
            fields={
                **facets,
                "detail": detail,
                "tags": tags or [],
                "evidence": list(evidence or []),
                "importance": importance,
                "metadata": entry_metadata,
                "expires": entry_expires,
                "assertions": list(assertions or []),
                "status": MemoryStatus.ACTIVE,
                "source": source,
                "source_identity": source_identity,
            },
        )
    else:
        entry = revise_entry(
            existing,
            local_node_id=local_node_id,
            now=now,
            fields={
                **facets,
                "content": content.strip(),
                "detail": detail,
                "tags": tags or [],
                "evidence": list(evidence) if evidence is not None else existing.evidence,
                "importance": importance,
                "metadata": entry_metadata,
                "expires": entry_expires,
                "assertions": list(assertions) if assertions is not None else existing.assertions,
                "source": source,
                "source_identity": source_identity or existing.source_identity,
            },
        )

    try:
        decision = prepare_entry_for_store(entry, backend=backend, config=cfg, session_id=session_id, trw_dir=trw_dir)
        if decision.quarantined:
            store_quarantined_entry(cfg, decision.entry)
            append_audit_event(
                cfg,
                "quarantine",
                entry_id=decision.entry.id,
                actor=decision.entry.source_identity or decision.entry.source,
                namespace=namespace,
                data={
                    "stored": False,
                    "quarantined": True,
                    "anomaly_dimension": decision.anomaly_dimension,
                    "z_score": decision.anomaly_z_score,
                },
            )
            return {
                "memory_id": decision.entry.id,
                "status": "quarantined",
                "namespace": namespace,
                "stored": False,
                "quarantined": True,
                "anomaly_dimension": decision.anomaly_dimension,
                "z_score": decision.anomaly_z_score,
            }

        entry = decision.entry
        if namespace.startswith("team:"):
            NamespaceManager(backend).ensure_team_namespace(namespace, created_at=now)
        # Mirror MemoryClient.store(): tool writes populate vectors too, otherwise
        # tool-created memories rank differently from SDK-created ones. Compute the
        # embedding *before* opening the write transaction — it is pure CPU work
        # with no DB state, so a failure here must leave nothing written.
        embedding: list[float] | None = None
        # Resolve the embedder only when a vector sink can consume the result;
        # otherwise the embed call is wasted on a no-op upsert_vector.
        embedder = (
            keyword_only_on_refusal(
                lambda: get_local_embedder(model_name=cfg.embedding_model, dim=cfg.embedding_dim),
                surface="memory_store",
            )[0]
            if embedding_has_consumer(cfg, backend)
            else None
        )
        if embedder is not None:
            try:
                embedding = embedder.embed(f"{entry.content} {entry.detail}")
            except Exception as exc:
                raise StorageError(f"failed to compute embedding for {entry_id!r}; entry was not written") from exc
        proof = (
            generation_provenance_kwargs(embedder, f"{entry.content} {entry.detail}", embedding)
            if embedding is not None
            else {}
        )
        # S1-parity fix: commit the row + its vector in ONE transaction so a crash
        # between the two writes can no longer leave a row with no vector, and a
        # vector failure rolls the row back automatically. This matches
        # MemoryClient.store() (_client_store.py) instead of the older
        # compensating-delete path, giving both store seams one atomicity model.
        # The row was read before the embedding, off the serialized lane, so a forget, update or
        # other store may have landed since: write only over the row this revision was built from.
        try:
            with backend.transaction():
                if _revision(_existing_entry_for_namespace(backend, entry_id, namespace)) != _revision(existing):
                    msg = f"{entry_id!r} changed while this store was prepared; nothing was written, retry"
                    return {"error": msg, "status": "conflict", "namespace": namespace}
                backend.store(entry)
                if embedding is not None:
                    backend.upsert_vector(entry.id, embedding, namespace=entry.namespace, **proof)
        except Exception as exc:
            raise StorageError(f"failed to persist entry+vector for {entry_id!r}; transaction rolled back") from exc
        try:
            # Graph enrichment is a secondary index over the stored entry, so we
            # dispatch it after the canonical row/vector write succeeds.
            schedule_graph_update(entry, backend, embedding=embedding, config=cfg)
        except RuntimeError:
            logger.warning("memory_store_graph_schedule_failed", entry_id=entry_id, exc_info=True)
        if supports_tier_runtime(backend):
            remember_entry_in_tiers(cfg, namespace, entry, embedding, proof.get("provenance"))
        append_audit_event(
            cfg,
            decision.op,
            entry_id=entry.id,
            actor=entry.source_identity or entry.source,
            namespace=namespace,
            data={
                "status": "updated" if decision.op == "update" else "stored",
                "session_id": session_id,
                "pii_types": sorted({match.pii_type for match in decision.pii_matches}),
                "quarantined": False,
            },
        )
    except SchemaValidationError as exc:
        return {"error": str(exc), "status": "invalid", "namespace": namespace}
    except RateLimitError as exc:
        # Transient: the window clears, so a caller retries rather than giving up on the content.
        return {"error": str(exc), "status": "rate_limited", "namespace": namespace, "retry_after": exc.retry_after}
    except (PIIBlockError, PoisoningError) as exc:
        return {"error": str(exc), "status": "blocked", "namespace": namespace}
    except (StorageError, RuntimeError, ValueError) as exc:
        logger.exception("memory_store_failed", entry_id=entry_id, error=str(exc))
        return {"error": f"storage error: {exc}", "status": "error"}

    logger.info(
        "memory_stored",
        entry_id=entry_id,
        namespace=namespace,
        tags=tags or [],
    )

    return {
        "memory_id": entry_id,
        "status": "updated" if decision.op == "update" else "stored",
        "namespace": namespace,
    }


class LearningFields(BaseModel):
    """The typed learning fields ``memory_store`` forwards to :func:`memory_store_impl` (PRD-CORE-294 FR07a).

    One optional object instead of ten parameters keeps the tool definition small;
    an unknown key is refused rather than silently dropped.
    """

    model_config = ConfigDict(extra="forbid")

    type: MemoryType | None = None
    confidence: Confidence | None = None
    task_type: str | None = None
    domain: list[str] | None = None
    phase_origin: str | None = None
    phase_affinity: list[str] | None = None
    team_origin: str | None = None
    protection_tier: ProtectionTier | None = None
    anchors: list[Anchor] | None = None
    nudge_line: str | None = None
    # PRD-CORE-298 FR01: provenance the daemon store forwards for trw-mcp writes.
    source: Literal["human", "agent", "tool", "consolidated", "team_sync", "company_sync"] | None = None
    client_profile: str | None = None
    model_id: str | None = None
    anchor_validity: float | None = None


def register_store_tool(mcp: McpServer) -> None:
    """Register memory_store with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from trw_memory.integrations._backend import create_backend_from_config

    @mcp.tool()
    async def memory_store(
        content: str,
        namespace: str = "project:default",
        tags: list[str] | None = None,
        importance: float = 0.5,
        detail: str = "",
        metadata: dict[str, str] | None = None,
        source_identity: str = "",
        session_id: str | None = None,
        entry_id: str | None = None,
        evidence: list[str] | None = None,
        expires: str = "",
        assertions: list[Assertion] | None = None,
        learning: LearningFields | None = None,
    ) -> dict[str, object]:
        """Store a new memory entry in the memory system.

        Args:
            content: Core knowledge statement to remember. Must be non-empty.
            namespace: Namespace scope (e.g., 'project:default', 'global').
            tags: Optional list of tags for categorisation.
            importance: Importance score 0.0-1.0 (default 0.5).
            detail: Extended explanation or context.
            metadata: Optional key-value string metadata.
            evidence: Optional source references supporting the entry.
            expires: Optional expiration date or condition.
            assertions: Optional machine-verifiable grounding assertions.
            learning: Optional typed fields (type, confidence, task_type, domain,
                phase_origin, phase_affinity, team_origin, protection_tier, anchors,
                nudge_line, source, client_profile, model_id, anchor_validity);
                unknown keys are rejected.

        Returns:
            {"memory_id": str, "status": "stored", "namespace": str}
        """

        def _run() -> dict[str, object]:
            # PRD-CORE-279 FR04: backend open, write and close all happen in ONE
            # worker thread, so the SQLite connection never crosses threads.
            cfg = MemoryConfig()
            typed = learning or LearningFields()
            with create_backend_from_config(cfg, namespace) as backend:
                return memory_store_impl(
                    content,
                    namespace,
                    backend=backend,
                    tags=tags,
                    importance=importance,
                    detail=detail,
                    metadata=metadata,
                    config=cfg,
                    source_identity=source_identity,
                    session_id=session_id,
                    entry_id=entry_id,
                    evidence=evidence,
                    expires=expires,
                    assertions=assertions,
                    source=typed.source or "tool",
                    client_profile=typed.client_profile,
                    model_id=typed.model_id,
                    anchor_validity=typed.anchor_validity,
                    type=typed.type,
                    confidence=typed.confidence,
                    task_type=typed.task_type,
                    domain=typed.domain,
                    phase_origin=typed.phase_origin,
                    phase_affinity=typed.phase_affinity,
                    team_origin=typed.team_origin,
                    protection_tier=typed.protection_tier,
                    anchors=typed.anchors,
                    nudge_line=typed.nudge_line,
                )

        return await run_offloaded(_run)
