"""YAML row mapping — dict <-> :class:`MemoryEntry`.

Belongs to the ``yaml_backend.py`` facade; re-exported there for back-compat.
The SQLite backend has kept its row mapper in a sibling (``_row_mapper.py``)
since PRD-DIST-245; this is the YAML twin, extracted by PRD-CORE-245 when the
schema-5 field changes pushed ``yaml_backend.py`` over the 350 effective-LOC
gate. Keeping the two mappers symmetric matters: every column change has to be
applied to both, and a reviewer comparing them should be comparing two files of
the same shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, NamedTuple, cast

import structlog

from trw_memory.models.memory import Anchor, Assertion, MemoryEntry, MemoryStatus
from trw_memory.storage._parsing import (
    parse_dt_safe,
    parse_json_dict_int,
    parse_json_dict_str,
    parse_json_list,
    parse_optional_float,
)
from trw_memory.storage._row_mapper import parse_verification_status

logger = structlog.get_logger(__name__)

__all__ = ["ParsedRow", "dict_to_entry", "entry_to_dict"]


class ParsedRow(NamedTuple):
    """A deserialised entry plus what could not be deserialised with it.

    The mapper drops verification evidence it cannot parse -- an assertion whose
    shape has drifted, an anchor missing a field -- and used to hand back a
    :class:`MemoryEntry` that was byte-identical to one written with no evidence
    at all. That is not a cosmetic distinction: ``YAMLBackend.update`` reads a
    row, mutates the parsed object and writes it back, so a silent drop on the
    read leg DELETES the unparsed evidence from disk on the write leg.

    ``dropped_assertions`` / ``dropped_anchors`` count what was discarded so a
    caller can log it, refuse to rewrite the row, or both.
    """

    entry: MemoryEntry
    dropped_assertions: int
    dropped_anchors: int

    @property
    def partial(self) -> bool:
        """Whether anything on the row failed to parse."""
        return bool(self.dropped_assertions or self.dropped_anchors)


def _parse_assertions(raw: object) -> tuple[list[Assertion], int]:
    """Deserialise assertions from YAML data, returning them and the drop count.

    A malformed item is skipped rather than failing the whole entry -- but the
    count rides out with the result, because "this row has no assertions" and
    "this row's assertions did not parse" are different facts.
    """
    if not raw or not isinstance(raw, list):
        return [], 0
    result: list[Assertion] = []
    dropped = 0
    for item in raw:
        if not isinstance(item, dict):
            dropped += 1
            continue
        try:
            result.append(Assertion.model_validate(item, strict=False))
        except (ValueError, KeyError):
            logger.debug("yaml_assertion_parse_skipped", item=item)
            dropped += 1
    return result, dropped


def _parse_anchors(raw: object) -> tuple[list[Anchor], int]:
    """Deserialise anchors from YAML data, returning them and the drop count.

    Per ITEM, unlike the list comprehension this replaces: that one ran inside a
    single ``try``, so ONE malformed anchor discarded every valid anchor beside
    it and the row read back as unanchored.
    """
    if not raw:
        return [], 0
    if not isinstance(raw, list):
        logger.debug("yaml_anchor_parse_skipped", anchors=raw)
        return [], 1
    result: list[Anchor] = []
    dropped = 0
    for item in raw:
        try:
            result.append(Anchor.model_validate(item))
        except (ValueError, KeyError):
            logger.debug("yaml_anchor_parse_skipped", anchors=item)
            dropped += 1
    return result, dropped


def entry_to_dict(entry: MemoryEntry) -> dict[str, object]:
    """Serialise a :class:`MemoryEntry` to a plain dict suitable for YAML."""
    return entry.to_dict()


def dict_to_entry(data: dict[str, object]) -> ParsedRow:
    """Deserialise a YAML dict back into a :class:`MemoryEntry` plus drop counts.

    All fields are cast explicitly to satisfy Pydantic strict mode. Returns a
    :class:`ParsedRow` rather than a bare entry so a partial parse is a fact the
    caller receives, not one it has to infer from an empty list.
    """

    def _str(key: str, default: str = "") -> str:
        val = data.get(key, default)
        return str(val) if val is not None else default

    def _int(key: str, default: int = 0) -> int:
        val = data.get(key, default)
        try:
            return int(str(val))
        except (TypeError, ValueError):
            return default

    def _float(key: str, default: float = 0.5) -> float:
        val = data.get(key, default)
        try:
            return float(str(val))
        except (TypeError, ValueError):
            return default

    def _str_list(key: str) -> list[str]:
        return parse_json_list(data.get(key, []))

    def _str_dict(key: str) -> dict[str, str]:
        return parse_json_dict_str(data.get(key, {}))

    # Timestamps are parsed fail-open: a malformed value (e.g. the WAL-reset
    # byte-shift '026-04-13T00:00:00+00:002') degrades the single field to a
    # usable default instead of raising and collapsing the whole listing —
    # matching how this mapper already fail-opens status / anchors / floats.
    now = datetime.now(timezone.utc)
    created_at_raw = data.get("created_at")
    updated_at_raw = data.get("updated_at")
    created_at = parse_dt_safe(created_at_raw, default=now) or now if created_at_raw else now
    updated_at = parse_dt_safe(updated_at_raw, default=now) or now if updated_at_raw else now

    last_accessed_raw = data.get("last_accessed_at")
    last_accessed_at: datetime | None = parse_dt_safe(last_accessed_raw, default=None) if last_accessed_raw else None

    status_raw = _str("status", "active")
    try:
        status = MemoryStatus(status_raw)
    except ValueError:
        status = MemoryStatus.ACTIVE

    consolidated_into_raw = data.get("consolidated_into")
    consolidated_into: str | None = str(consolidated_into_raw) if consolidated_into_raw else None

    anchors, dropped_anchors = _parse_anchors(data.get("anchors"))
    assertions, dropped_assertions = _parse_assertions(data.get("assertions", []))

    entry = MemoryEntry(
        id=_str("id"),
        content=_str("content"),
        detail=_str("detail"),
        tags=_str_list("tags"),
        evidence=_str_list("evidence"),
        importance=_float("importance", 0.5),
        status=status,
        recurrence=_int("recurrence", 1),
        namespace=_str("namespace", "default"),
        created_at=created_at,
        updated_at=updated_at,
        last_accessed_at=last_accessed_at,
        access_count=_int("access_count", 0),
        session_count=_int("session_count", 0),
        q_value=_float("q_value", 0.5),
        q_observations=_int("q_observations", 0),
        source=cast("Literal['human', 'agent', 'tool', 'consolidated']", _str("source", "agent")),
        source_identity=_str("source_identity"),
        client_profile=_str("client_profile"),
        model_id=_str("model_id"),
        merged_from=_str_list("merged_from"),
        consolidated_from=_str_list("consolidated_from"),
        consolidated_into=consolidated_into,
        metadata=_str_dict("metadata"),
        vector_clock=parse_json_dict_int(data.get("vector_clock", {})),
        remote_id=str(remote_id_raw) if (remote_id_raw := data.get("remote_id")) else None,
        published_to_platform=bool(data.get("published_to_platform", False)),
        pending_delete=bool(data.get("pending_delete", False)),
        cross_validated=bool(data.get("cross_validated", False)),
        outcome_history=_str_list("outcome_history"),
        assertions=assertions,
        anchors=anchors,
        valid_from=parse_dt_safe(data.get("valid_from"), default=created_at) or created_at,
        invalid_from=parse_dt_safe(data.get("invalid_from"), default=None) if data.get("invalid_from") else None,
        invalidated_by=str(data["invalidated_by"]) if data.get("invalidated_by") else None,
        # PRD-CORE-244-FR01: absent => never assessed, not a perfect score.
        anchor_validity=parse_optional_float(data.get("anchor_validity")),
        verification_status=parse_verification_status(data.get("verification_status")),
        verification_checked_at=str(data.get("verification_checked_at") or ""),
        type=_str("type", "pattern"),
        nudge_line=_str("nudge_line", ""),
        expires=_str("expires", ""),
        confidence=_str("confidence", "unverified"),
        task_type=_str("task_type", ""),
        domain=_str_list("domain"),
        phase_origin=_str("phase_origin", ""),
        phase_affinity=_str_list("phase_affinity"),
        team_origin=_str("team_origin", ""),
        protection_tier=_str("protection_tier", "normal"),
        sync_hash=_str("sync_hash", ""),
        sync_seq=_int("sync_seq", 0),
        last_synced_at=parse_dt_safe(data.get("last_synced_at"), default=None) if data.get("last_synced_at") else None,
    )
    return ParsedRow(entry, dropped_assertions, dropped_anchors)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
