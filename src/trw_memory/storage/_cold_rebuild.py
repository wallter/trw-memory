"""Rebuild the SQLite memories table from the cold YAML tier.

PRD-CORE-140 — provides an explicit, gated, cold-tier rehydration path for
``SQLiteBackend.recover_db`` and a user-facing CLI ``trw-memory restore
--from-cold``. Walks ``base_dir/memory/cold/**/*.yaml``, validates each path
against the cold root (path-traversal guard), hydrates per the verified
2026-04-12 mapping, and ``INSERT OR IGNORE``s into the caller-supplied
connection inside a single transaction.

Key invariants (do not relax without updating the PRD):

- DB ``type`` is hardcoded to ``"pattern"``. Do NOT map YAML ``source_type``
  onto DB ``type`` — different concepts (provenance vs MemoryType enum). This
  was the actual bug during 2026-04-12 manual recovery.
- NOT NULL columns (``id``, ``content``, ``created_at``, ``updated_at``) must
  be populated — ``INSERT OR IGNORE`` silently drops rows that violate
  NOT NULL, which would appear as a phantom skip without a WARNING.
- Per-file failures skip; the whole rebuild never aborts because of one bad
  YAML. Every skip emits a structured WARNING with filename + field.
- No strict validation against the live ``MemoryEntry`` Pydantic model — let
  any row with id + content + timestamps pass; enum strictness is enforced
  later on read via ``row_to_entry``.
- ``status`` is preserved verbatim (``obsolete``/``resolved`` are NOT
  blanket-reset — cold tier is the user's preserved record).
"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path
from sqlite3 import Connection

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.storage.persistence import read_yaml

__all__ = ["rebuild_from_cold"]

logger = structlog.get_logger(__name__)

# Matches bare ``YYYY-MM-DD`` (no 'T' separator). Used to detect date-only
# timestamps that need normalisation to full ISO-8601 before insertion.
_DATE_ONLY_RE: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# YAML list-typed fields that round-trip as JSON arrays in the DB.
_LIST_FIELDS: tuple[str, ...] = (
    "tags",
    "evidence",
    "merged_from",
    "consolidated_from",
    "outcome_history",
    "assertions",
    "anchors",
    "domain",
    "phase_affinity",
)

# YAML dict-typed fields that round-trip as JSON objects in the DB.
_DICT_FIELDS: tuple[str, ...] = ("metadata", "vector_clock")

# Column order for the INSERT statement. Matches the subset of
# ``ENTRY_COLUMNS`` that ``_hydrate_yaml`` actually populates. Kept local
# rather than importing from ``_shared`` because we deliberately hardcode
# ``type`` and default several columns here rather than relying on SQL
# ``DEFAULT`` clauses (safer under schema drift).
_INSERT_COLUMNS: tuple[str, ...] = (
    "id",
    "content",
    "detail",
    "tags",
    "evidence",
    "importance",
    "status",
    "recurrence",
    "namespace",
    "created_at",
    "updated_at",
    "source",
    "merged_from",
    "consolidated_from",
    "consolidated_into",
    "metadata",
    "vector_clock",
    "outcome_history",
    "assertions",
    "anchors",
    "type",
    "domain",
    "phase_affinity",
)


def _assert_within_cold_dir(cold_base: Path, candidate: Path) -> None:
    """Reject ``candidate`` if it escapes ``cold_base`` after resolution.

    Mirrors the semantics of
    :meth:`trw_memory.lifecycle.tiers._cold.ColdTierStore._assert_within_cold_dir`.
    Resolves both paths so that symlinks cannot slip out of the cold root.

    Raises:
        ValueError: when ``candidate`` is not a descendant of ``cold_base``.
    """
    resolved_base = cold_base.resolve()
    resolved_candidate = candidate.resolve()
    if not resolved_candidate.is_relative_to(resolved_base):
        raise ValueError(
            f"Path traversal guard: {candidate} is not under cold dir {resolved_base}"
        )


def _normalize_ts(ts_str: str | None) -> str | None:
    """Convert date-only ``YYYY-MM-DD`` to full ISO-8601.

    Leaves other strings (already full ISO, numeric offsets, etc.) unchanged.
    Returns ``None`` for ``None`` input.
    """
    if ts_str is None:
        return None
    if _DATE_ONLY_RE.fullmatch(ts_str):
        return f"{ts_str}T00:00:00+00:00"
    return ts_str


def _coerce_ts(value: object) -> str | None:
    """Coerce a YAML-loaded timestamp into an ISO-8601 string or ``None``.

    ruamel.yaml parses bare ``YYYY-MM-DD`` literals as :class:`datetime.date`
    and full timestamps as :class:`datetime.datetime`. Older entries may also
    store the string directly.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    # date (and its subclass datetime handled above) — stringify and normalise
    if isinstance(value, date):
        return f"{value.isoformat()}T00:00:00+00:00"
    if isinstance(value, str):
        return _normalize_ts(value)
    # Unexpected type — coerce best-effort; the caller will skip if invalid.
    return str(value)


def _dumps_list(value: object) -> str:
    """Serialise a list-typed YAML field as a JSON array string.

    Treats ``None`` as ``[]``. Non-list inputs propagate the
    :class:`TypeError` from :func:`json.dumps` so callers can record the
    offending field.
    """
    payload = value if value is not None else []
    if not isinstance(payload, list):
        raise TypeError(f"expected list, got {type(payload).__name__}")
    return json.dumps(payload)


def _dumps_dict(value: object) -> str:
    """Serialise a dict-typed YAML field as a JSON object string."""
    payload = value if value is not None else {}
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict, got {type(payload).__name__}")
    return json.dumps(payload)


def _hydrate_yaml(y: dict[str, object]) -> tuple[object, ...] | None:
    """Map a cold-tier YAML dict to an insert tuple.

    Returns ``None`` when a required NOT NULL field is missing. The caller
    treats ``None`` as a skip-with-WARNING (``field=<first-missing>``).

    Raises:
        ValueError / TypeError: for type errors on list/dict fields; the
            caller catches and records the offending field.
    """
    entry_id = y.get("id")
    if not entry_id or not isinstance(entry_id, str):
        raise _HydrationError("id")

    # YAML ``summary`` is the DB ``content`` column.
    content_raw = y.get("summary")
    if content_raw is None:
        # Accept ``content`` as fallback for any future YAML writers that
        # stamp it directly; cold archive writers today only stamp ``summary``.
        content_raw = y.get("content")
    if content_raw is None or not str(content_raw).strip():
        raise _HydrationError("summary")
    content = str(content_raw)

    created_at = _coerce_ts(y.get("created") or y.get("created_at"))
    if not created_at:
        raise _HydrationError("created")

    updated_at = _coerce_ts(y.get("updated") or y.get("updated_at"))
    if not updated_at:
        # Permissive fallback: reuse created_at rather than skipping the row.
        # The 2026-04-12 recovery cohort had a small number of YAMLs with
        # missing ``updated`` — preserving them is preferable to dropping.
        updated_at = created_at

    detail = str(y.get("detail") or "")

    impact_raw = y.get("impact", 0.5)
    try:
        importance = float(impact_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise _HydrationError("impact") from exc

    recurrence_raw = y.get("recurrence", 1)
    try:
        recurrence = int(recurrence_raw)  # type: ignore[call-overload]
    except (TypeError, ValueError) as exc:
        raise _HydrationError("recurrence") from exc

    status_raw = y.get("status", "active")
    status = str(status_raw) if status_raw else "active"

    namespace = str(y.get("namespace") or "default")

    # Provenance on the DB side lives on the ``source`` column — NOT ``type``.
    # This is the invariant that was violated during 2026-04-12 manual
    # recovery (``type = 'agent'`` raised ``ValueError: not a valid MemoryType``).
    source = str(y.get("source_type") or y.get("source") or "agent")

    consolidated_into_raw = y.get("consolidated_into")
    consolidated_into = (
        str(consolidated_into_raw) if consolidated_into_raw not in (None, "") else None
    )

    # List-typed fields — each may raise TypeError on unexpected shape.
    serialised_lists: dict[str, str] = {}
    for field in _LIST_FIELDS:
        try:
            serialised_lists[field] = _dumps_list(y.get(field))
        except (TypeError, ValueError) as exc:
            raise _HydrationError(field) from exc

    serialised_dicts: dict[str, str] = {}
    for field in _DICT_FIELDS:
        try:
            serialised_dicts[field] = _dumps_dict(y.get(field))
        except (TypeError, ValueError) as exc:
            raise _HydrationError(field) from exc

    # Column order MUST match _INSERT_COLUMNS exactly.
    return (
        entry_id,
        content,
        detail,
        serialised_lists["tags"],
        serialised_lists["evidence"],
        importance,
        status,
        recurrence,
        namespace,
        created_at,
        updated_at,
        source,
        serialised_lists["merged_from"],
        serialised_lists["consolidated_from"],
        consolidated_into,
        serialised_dicts["metadata"],
        serialised_dicts["vector_clock"],
        serialised_lists["outcome_history"],
        serialised_lists["assertions"],
        serialised_lists["anchors"],
        "pattern",  # hardcoded DB `type` — see module docstring
        serialised_lists["domain"],
        serialised_lists["phase_affinity"],
    )


class _HydrationError(ValueError):
    """Raised when a single YAML cannot be hydrated; carries the offending field name."""

    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.field = field


def _iter_cold_files(cold_base: Path) -> list[Path]:
    """Return all ``*.yaml`` files under the cold root, or empty if none."""
    if not cold_base.exists() or not cold_base.is_dir():
        return []
    return list(cold_base.rglob("*.yaml"))


def rebuild_from_cold(base_dir: Path, new_conn: Connection) -> int:
    """Rebuild ``memories`` rows from the cold YAML archive.

    Args:
        base_dir: Root directory whose ``memory/cold/`` subtree holds the
            archived YAMLs. Typical caller value is ``db_path.parent.parent``
            for the standard ``<base>/memory/memory.db`` layout.
        new_conn: Open SQLite connection on the destination DB. Must already
            have the ``memories`` table (call ``ensure_schema`` first).

    Returns:
        Count of rows successfully inserted via ``INSERT OR IGNORE``. A
        duplicate id does NOT raise — it is counted as a skipped row.

    Notes:
        - Does NOT open its own DB — caller is responsible for the
          connection and for committing any surrounding transaction.
        - Emits a structured summary log ``cold_rebuild_complete`` with
          ``rebuilt``, ``skipped``, ``cold_files``, ``base_dir``, and
          ``duration_ms``.
    """
    started_ns = time.monotonic_ns()
    cold_base = base_dir / "memory" / "cold"
    cold_files = _iter_cold_files(cold_base)

    if not cold_files:
        logger.info(
            "cold_rebuild_complete",
            rebuilt=0,
            skipped=0,
            cold_files=0,
            base_dir=str(base_dir),
            duration_ms=0,
            reason="no_cold_files",
        )
        return 0

    placeholders = ", ".join(["?"] * len(_INSERT_COLUMNS))
    insert_sql = (
        f"INSERT OR IGNORE INTO memories ({', '.join(_INSERT_COLUMNS)}) "  # noqa: S608
        f"VALUES ({placeholders})"
    )

    rebuilt = 0
    skipped = 0
    cursor = new_conn.cursor()
    try:
        # Single explicit transaction per NFR01.
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute("BEGIN")

        for yaml_path in cold_files:
            try:
                _assert_within_cold_dir(cold_base, yaml_path)
            except ValueError:
                skipped += 1
                logger.warning(
                    "cold_rebuild_skipped",
                    file=str(yaml_path),
                    field=None,
                    reason="path_traversal_guard",
                )
                continue

            try:
                data = read_yaml(yaml_path)
            except (StorageError, OSError) as exc:
                skipped += 1
                logger.warning(
                    "cold_rebuild_skipped",
                    file=str(yaml_path),
                    field=None,
                    reason="read_failed",
                    detail=str(exc),
                )
                continue

            try:
                row = _hydrate_yaml(data)
            except _HydrationError as exc:
                skipped += 1
                logger.warning(
                    "cold_rebuild_skipped",
                    file=str(yaml_path),
                    field=exc.field,
                    reason="hydration_failed",
                )
                continue
            except (TypeError, ValueError) as exc:
                # Defensive: any unexpected hydration error is a skip.
                skipped += 1
                logger.warning(
                    "cold_rebuild_skipped",
                    file=str(yaml_path),
                    field=None,
                    reason="hydration_failed",
                    detail=str(exc),
                )
                continue

            if row is None:
                skipped += 1
                logger.warning(
                    "cold_rebuild_skipped",
                    file=str(yaml_path),
                    field=None,
                    reason="hydration_returned_none",
                )
                continue

            try:
                cursor.execute(insert_sql, row)
            except sqlite3.Error as exc:
                skipped += 1
                logger.warning(
                    "cold_rebuild_skipped",
                    file=str(yaml_path),
                    field=None,
                    reason="insert_failed",
                    detail=str(exc),
                )
                continue

            # INSERT OR IGNORE: rowcount is 1 on insert, 0 on ignored duplicate.
            if cursor.rowcount > 0:
                rebuilt += 1
            else:
                skipped += 1
                logger.debug(
                    "cold_rebuild_insert_ignored",
                    file=str(yaml_path),
                    reason="duplicate_id",
                )

        new_conn.commit()
    except Exception:
        # NFR01: rollback on unhandled exception.
        with contextlib.suppress(sqlite3.Error):
            new_conn.rollback()
        raise
    finally:
        cursor.close()

    duration_ms = (time.monotonic_ns() - started_ns) // 1_000_000
    logger.info(
        "cold_rebuild_complete",
        rebuilt=rebuilt,
        skipped=skipped,
        cold_files=len(cold_files),
        base_dir=str(base_dir),
        duration_ms=duration_ms,
    )
    return rebuilt
