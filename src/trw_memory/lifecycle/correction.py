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

Failures are returned, never raised: ``{"status": "invalid" | "not_found", "error": ...}``,
and ``not_found`` also carries ``error_type: learning_not_found``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, NamedTuple

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from trw_memory.exceptions import SchemaValidationError
from trw_memory.lifecycle.tiers._runtime import (
    remember_entries_data_in_tiers,
    remove_entry_from_tiers,
    supports_tier_runtime,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import Assertion, Confidence, MemoryStatus, MemoryType, ProtectionTier
from trw_memory.security.poisoning import reject_unsubstantiated_verified

if TYPE_CHECKING:
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

__all__ = ["LearningPatch", "Store", "apply_correction", "not_found", "parse_patch"]


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
    evidence: list[str] | None = None
    recurrence: int | None = Field(default=None, ge=0)
    merged_from: list[str] | None = None
    metadata_add: dict[str, str] | None = None

    @field_validator("assertions", mode="before")
    @classmethod
    def _lax_assertions(cls, value: object) -> object:
        # Assertion is strict (enum instances only); callers send JSON strings.
        if isinstance(value, list):
            return [Assertion.model_validate(item, strict=False) if isinstance(item, dict) else item for item in value]
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
    ("task_type", "task_type"),
    ("domain", "domain"),
    ("phase_origin", "phase_origin"),
    ("phase_affinity", "phase_affinity"),
    ("team_origin", "team_origin"),
    ("protection_tier", "protection_tier"),
    ("anchor_validity", "anchor_validity"),
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


def _refuse_unsubstantiated(entry: MemoryEntry, fields: dict[str, object], min_items: int) -> dict[str, str] | None:
    # PRD-CORE-244 FR02 on the update path: only a promotion needs a basis, and the
    # entry judged is the post-update one, since this call may carry the assertions.
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


def apply_correction(
    store: Store,
    entry: MemoryEntry,
    patch: LearningPatch,
    *,
    prior: tuple[Store, MemoryEntry | None] | None = None,
) -> dict[str, str]:
    """Apply ``patch`` to ``entry`` in ``store``; return ``updated`` / ``no_changes`` / ``invalid``.

    ``entry`` identifies the row; the patch is applied to the row as re-read inside
    the write transaction, not to this possibly stale copy. ``prior`` is the owning
    store and entry of ``patch.supersedes``, resolved by the caller (which knows
    every store the id could live in). A missing or already-closed prior is a
    no-op; the primary edit still applies.
    """
    backend = store.backend
    with backend.transaction():
        current = backend.get(entry.id, namespace=entry.namespace) or entry
        fields, changes = _collect(current, patch)
        refusal = _refuse_unsubstantiated(current, fields, int(store.config.min_evidence_items_for_verified))
        if refusal is not None:
            return refusal
        if fields:
            backend.update(entry.id, namespace=entry.namespace, **fields)
    # The prior may live in another store, outside this transaction: close it only
    # once the new row has committed. A failure between the two leaves the prior
    # open, and repeating the correction closes it.
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
