"""Correct or retire a stored learning by id — the one implementation (PRD-CORE-294 FR03).

Both surfaces call ``apply_correction``: trw-mcp's ``trw_learn(learning_id=...)``
update mode (after it picks the owning project/user store) and this package's
``memory_update`` tool. Validation, patch semantics, the verified-promotion
gate, the provenance-hash refresh, supersession and the tier-mirror refresh
therefore mean the same thing everywhere.

The read the patch is applied to happens inside the store's write transaction,
so two concurrent ``tags_add`` calls both land instead of one overwriting the
other with a set built from a stale read.

Patch semantics: a field the caller did not name is untouched (``None`` is
"not named"; clearing is explicit with ``""`` or ``[]``). ``tags`` replaces the
tag set; ``tags_add`` appends, keeping order and dropping duplicates. Retiring
a learning is ``status="obsolete"``: default recall reads active rows only.

Failures are returned, never raised: ``{"status": "invalid" | "not_found" | "conflict", "error": ...}``,
and ``not_found`` also carries ``error_type: learning_not_found``.

``if_revision`` makes the patch conditional (PRD-CORE-308): it applies only while
``revision_of(row)`` still equals it, else ``conflict`` and nothing is written. A
caller that computes absolute values from a row it read sends that row's revision.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, NamedTuple

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from trw_memory.embeddings.provenance import generation_provenance_kwargs
from trw_memory.exceptions import PoisoningError, SchemaValidationError
from trw_memory.lifecycle.tiers._runtime import (
    remember_entries_data_in_tiers,
    remove_entry_from_tiers,
    supports_tier_runtime,
)
from trw_memory.models._assertion_cap import OVERLONG, overlong
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import (
    Anchor,
    Assertion,
    Confidence,
    EvidenceLevel,
    MemoryStatus,
    MemoryType,
    ProtectionTier,
)
from trw_memory.security.poisoning import reject_injection, reject_unsubstantiated_verified
from trw_memory.storage._shared import revision_of
from trw_memory.storage._utf8_validator import refuse_overlong_text_fields
from trw_memory.storage.interface import is_transactional

if TYPE_CHECKING:
    from trw_memory.embeddings.interface import EmbeddingProvider
    from trw_memory.embeddings.provenance import VectorProvenance
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

__all__ = ["CONFLICT_ATTEMPTS", "LearningPatch", "Store", "apply_correction", "not_found", "parse_patch", "revision_of"]

#: Times a revision-conditional writer re-reads and retries before it reports ``conflict`` (PRD-CORE-308).
CONFLICT_ATTEMPTS = 3


class Store(NamedTuple):
    """One store a learning can live in: its backend and the config naming its tier mirror."""

    backend: StorageBackend
    config: MemoryConfig


#: Statuses a correction may set. ``archived`` belongs to tier lifecycle, not callers.
CorrectableStatus = Literal["active", "resolved", "obsolete", "obsolete_poisoned"]
Phase = Literal["", "RESEARCH", "PLAN", "IMPLEMENT", "VALIDATE", "REVIEW", "DELIVER"]


class LearningPatch(BaseModel):
    """The fields a correction may name. Unknown keys are rejected, not ignored."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    status: CorrectableStatus | None = None
    summary: str | None = None
    detail: str | None = None
    impact: float | None = Field(default=None, ge=0.0, le=1.0)
    type: MemoryType | None = None
    confidence: Confidence | None = None
    evidence_level: EvidenceLevel | None = None  # PRD-CORE-312-FR01
    protection_tier: ProtectionTier | None = None
    phase_origin: Phase | None = None
    phase_affinity: list[str] | None = None
    nudge_line: str | None = Field(default=None, max_length=80)
    expires: str | None = None
    task_type: str | None = None
    domain: list[str] | None = None
    team_origin: str | None = None
    tags: list[str] | None = None
    tags_add: list[str] | None = None
    assertions: list[Assertion] | None = None
    supersedes: str | None = None
    # Written by maintenance rather than by hand: anchor re-verification, dedup
    # merges and promotion marks. ``metadata_add`` merges into metadata like ``tags_add``.
    anchor_validity: float | None = Field(default=None, ge=0.0, le=1.0)
    anchors: list[Anchor] | None = None  # replaces the whole list; the anchor postings follow it
    evidence: list[str] | None = None
    recurrence: int | None = Field(default=None, ge=0)
    merged_from: list[str] | None = None
    metadata_add: dict[str, str] | None = None
    if_revision: str | None = None  # apply only over this ``revision_of`` the row (PRD-CORE-308)

    @field_validator("assertions", mode="before")
    @classmethod
    def _lax_assertions(cls, value: object) -> object:
        # Assertion is strict (enum instances only); callers send JSON strings.
        if isinstance(value, list):
            value = [Assertion.model_validate(item, strict=False) if isinstance(item, dict) else item for item in value]
            if any(isinstance(item, Assertion) and overlong(item) for item in value):
                raise ValueError(OVERLONG)
        return value


def parse_patch(fields: dict[str, object]) -> LearningPatch | dict[str, str]:
    """Validate raw fields into a patch, or return the ``invalid`` result naming the first bad field."""
    try:
        return LearningPatch.model_validate({k: v for k, v in fields.items() if v is not None})
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "patch"
        return {"status": "invalid", "error": f"Invalid {field} {first.get('input')!r}: {first['msg']}"}


def not_found(learning_id: str) -> dict[str, str]:
    return {
        "status": "not_found",
        "error_type": "learning_not_found",
        "error": f"Learning {learning_id} not found",
    }


# (patch attribute, stored field). Order is the order changes are reported in.
_PLAIN_FIELDS: tuple[tuple[str, str], ...] = (
    ("status", "status"),
    ("detail", "detail"),
    ("summary", "content"),
    ("impact", "importance"),
    ("type", "type"),
    ("nudge_line", "nudge_line"),
    ("expires", "expires"),
    ("confidence", "confidence"),
    ("evidence_level", "evidence_level"),
    ("task_type", "task_type"),
    ("domain", "domain"),
    ("phase_origin", "phase_origin"),
    ("phase_affinity", "phase_affinity"),
    ("team_origin", "team_origin"),
    ("protection_tier", "protection_tier"),
    ("anchor_validity", "anchor_validity"),
    ("anchors", "anchors"),
    ("evidence", "evidence"),
    ("recurrence", "recurrence"),
    ("merged_from", "merged_from"),
)


#: Fields whose change is reported with its new value; the rest report "<field> updated".
_VALUED = frozenset(
    {"status", "impact", "type", "confidence", "task_type", "phase_origin", "team_origin", "protection_tier"}
)


def _label(attr: str, value: object) -> str:
    if attr not in _VALUED:
        return f"{attr} updated"
    return f"{attr}→{value}" if value != "" else f"{attr} cleared"


def _collect(entry: MemoryEntry, patch: LearningPatch) -> tuple[dict[str, object], list[str]]:
    fields: dict[str, object] = {}
    changes: list[str] = []
    for attr, stored in _PLAIN_FIELDS:
        value = getattr(patch, attr)
        if value is None:
            continue
        fields[stored] = value
        changes.append(_label(attr, value))
    if patch.tags is not None or patch.tags_add is not None:
        tags = list(patch.tags if patch.tags is not None else entry.tags)
        tags += [tag for tag in dict.fromkeys(patch.tags_add or []) if tag not in tags]
        fields["tags"] = tags
        changes.append("tags updated")
    if patch.assertions is not None:
        fields["assertions"] = patch.assertions
        changes.append("assertions updated")
    metadata = {**entry.metadata, **(patch.metadata_add or {})}
    if patch.metadata_add:
        from trw_memory.labels import LabelPolicy  # PRD-SEC-023 FR07: a correction can only raise the stored stamp

        metadata = LabelPolicy.current().joined(metadata, entry.metadata)
    if patch.metadata_add:
        changes.append("metadata updated")
    if (patch.summary is not None or patch.detail is not None) and (
        entry.metadata.get("provenance_content_hash") or entry.metadata.get("content_hash")
    ):
        content = patch.summary if patch.summary is not None else entry.content
        detail = patch.detail if patch.detail is not None else entry.detail
        metadata["provenance_content_hash"] = hashlib.sha256(f"{content}{detail}".encode()).hexdigest()
    if metadata != entry.metadata:
        fields["metadata"] = metadata
    return fields, changes


# The fields ``poisoning.scannable_text`` reads, under their stored names: a patch that changes none of them is never
# scanned, so retiring or re-rating a row stored before a pattern existed still works.
_SCANNED_FIELDS = frozenset({"content", "detail", "nudge_line", "tags", "evidence", "assertions"})


def _refuse_injection(entry: MemoryEntry, fields: dict[str, object]) -> dict[str, str] | None:
    # UF-MEM-03: the store path's injection gate on the post-correction row. The recall-time redactor that used to
    # stand behind an unchecked correction is gone, and a correction refreshes the content hash, so nothing else would.
    if not _SCANNED_FIELDS & fields.keys():
        return None
    try:
        reject_injection(entry.model_copy(update={k: v for k, v in fields.items() if k in _SCANNED_FIELDS}))
    except PoisoningError as exc:
        logger.warning("injection_correction_rejected", learning_id=entry.id)
        return {"status": "invalid", "error": str(exc), "reason": exc.reason}
    return None


def _refuse_unsubstantiated(entry: MemoryEntry, fields: dict[str, object], min_items: int) -> dict[str, str] | None:
    # PRD-CORE-244 FR02 on the update path: only a promotion needs a basis, and the
    # entry judged is the post-update one, since this call may carry the assertions.
    # PRD-CORE-312: the evidence_level axis is enforced separately, as a data
    # invariant in the storage layer (``_evidence_invariant``), not here.
    if fields.get("confidence") != Confidence.VERIFIED.value:
        return None
    projected = entry.model_copy(
        update={"confidence": Confidence.VERIFIED, "assertions": fields.get("assertions", entry.assertions)}
    )
    try:
        reject_unsubstantiated_verified(projected, min_items=min_items)
    except SchemaValidationError as exc:
        logger.warning("unsubstantiated_verified_update_rejected", learning_id=entry.id, reason=exc.reason)
        return {"status": "invalid", "error": str(exc), "reason": exc.reason}
    return None


def _encode_patched(
    entry: MemoryEntry, patch: LearningPatch, embedder: EmbeddingProvider | None
) -> tuple[str, list[float] | None, dict[str, VectorProvenance]]:
    """Encode the text *patch* gives *entry*, before the write transaction (encoding holds no row lock)."""
    content = patch.summary if patch.summary is not None else entry.content
    detail = patch.detail if patch.detail is not None else entry.detail
    text = f"{content} {detail}"
    vector = embedder.embed(text) if embedder is not None else None
    return text, vector, generation_provenance_kwargs(embedder, text, vector) if vector is not None else {}


def apply_correction(
    store: Store,
    entry: MemoryEntry,
    patch: LearningPatch,
    *,
    prior: tuple[Store, MemoryEntry | None] | None = None,
    embedder: EmbeddingProvider | None = None,
) -> dict[str, str]:
    """Apply ``patch`` to ``entry`` in ``store``; return ``updated`` / ``no_changes`` / ``invalid`` / ``conflict``.

    ``entry`` identifies the row; the patch is applied to the row as re-read inside
    the write transaction, not to this possibly stale copy. ``prior`` is the owning
    store and entry of ``patch.supersedes``, resolved by the caller (which knows
    every store the id could live in). A missing or already-closed prior is a
    no-op; the primary edit still applies.

    A summary or detail change replaces the row's live vector in the same
    transaction (PRD-CORE-302 C5): re-encoded with *embedder* when the text it
    encoded is still the committed text, else dropped -- never left beside text it
    was not computed from. A row with no live vector (pruned, or never embedded)
    stays without one; ``memory_reembed`` owns that backfill.
    """
    backend = store.backend
    if patch.if_revision is not None and not is_transactional(backend):
        # The compare and the write must share one lock; a no-op transaction (YAML) cannot give one.
        msg = f"if_revision needs a transactional backend, not {type(backend).__name__}"
        return {"status": "invalid", "error": msg}
    try:
        # Same local cap as a store (PRD-CORE-331 FR07 / B71-94): refused, not truncated, before
        # any encoding or backend work, so a patched-in overlong value can't slip past the write path.
        refuse_overlong_text_fields(content=patch.summary, detail=patch.detail)
    except SchemaValidationError as exc:
        return {"status": "invalid", "error": str(exc), "failed_fields": ",".join(exc.failed_fields)}
    text_changed = patch.summary is not None or patch.detail is not None
    encoded_text, vector, proof = _encode_patched(entry, patch, embedder) if text_changed else ("", None, {})
    same_store, closed = prior is not None and prior[0].backend is backend, None
    with backend.transaction():
        # A row deleted since the caller read it is gone, not a stale copy to patch (C12 rc7).
        if (current := backend.get(entry.id, namespace=entry.namespace)) is None:
            return not_found(entry.id)
        if patch.if_revision is not None and patch.if_revision != revision_of(current):
            msg = f"{entry.id} changed since revision {patch.if_revision[:12]}; nothing was written, re-read and retry"
            return {"learning_id": entry.id, "status": "conflict", "error": msg}
        fields, changes = _collect(current, patch)
        refusal = _refuse_injection(current, fields) or _refuse_unsubstantiated(
            current, fields, int(store.config.min_evidence_items_for_verified)
        )
        if refusal is not None:
            return refusal
        if fields:
            try:
                if backend.update(entry.id, namespace=entry.namespace, **fields) is None:
                    return not_found(entry.id)
            except SchemaValidationError as exc:
                # PRD-CORE-312: the evidence-invariant refusal now fires INSIDE
                # backend.update() itself (a data invariant, not a per-caller
                # chokepoint check) -- convert it to the same rejection shape
                # ``_refuse_unsubstantiated``'s own check already returns.
                logger.warning("unsubstantiated_verified_update_rejected", learning_id=entry.id, reason=exc.reason)
                return {"status": "invalid", "error": str(exc), "reason": exc.reason}
        if text_changed and backend.vector_exists(entry.id, namespace=entry.namespace):
            backend.delete_vector(entry.id, namespace=entry.namespace)
            committed = f"{fields.get('content', current.content)} {fields.get('detail', current.detail)}"
            if vector is not None and committed == encoded_text:
                backend.upsert_vector(entry.id, vector, namespace=entry.namespace, **proof)
        if same_store:  # one commit with its replacement, so no forget of that lands between them
            closed = _close_prior(entry.id, patch, prior)
    # A prior in another store is outside this transaction: close it only once the new
    # row has committed. A failure between the two leaves the prior open, and repeating
    # the correction closes it.
    if not same_store:
        closed = _close_prior(entry.id, patch, prior)
    if closed is not None:
        changes.append(f"supersedes→{patch.supersedes}")
    if not changes:
        return {"learning_id": entry.id, "status": "no_changes"}
    _refresh_tiers(store, entry.namespace, entry.id)
    if closed is not None:
        _refresh_tiers(closed[0], closed[1].namespace, closed[1].id)
    logger.info("learning_corrected", learning_id=entry.id, changes=changes)
    return {"learning_id": entry.id, "status": "updated", "changes": ", ".join(changes)}


def _close_prior(
    entry_id: str, patch: LearningPatch, prior: tuple[Store, MemoryEntry | None] | None
) -> tuple[Store, MemoryEntry] | None:
    if patch.supersedes is None or patch.supersedes == entry_id or prior is None:
        return None
    prior_store, prior_entry = prior
    # Re-read: the caller's copy predates this write, and a closer committed since keeps its window.
    if prior_entry is not None:
        prior_entry = prior_store.backend.get(prior_entry.id, namespace=prior_entry.namespace)
    if prior_entry is None or prior_entry.invalid_from is not None:
        logger.info("supersession_prior_skipped", supersedes=patch.supersedes, found=prior_entry is not None)
        return None
    prior_store.backend.update(
        prior_entry.id,
        namespace=prior_entry.namespace,
        invalid_from=datetime.now(timezone.utc),
        invalidated_by=entry_id,
    )
    return prior_store, prior_entry


def _refresh_tiers(store: Store, namespace: str, entry_id: str) -> None:
    # Recall merges the hot/warm tier mirror, which holds a copy of each row as it
    # was last stored or recalled; a corrected row must not come back stale, and a
    # retired one must not come back at all.
    if not supports_tier_runtime(store.backend):
        return
    fresh = store.backend.get(entry_id, namespace=namespace)
    if fresh is not None and fresh.status == MemoryStatus.ACTIVE:
        remember_entries_data_in_tiers(store.config, [fresh.model_dump(mode="json")])
    else:
        remove_entry_from_tiers(store.config, namespace, entry_id)
