"""Migration from TRW LearningEntry format to MemoryEntry.

Converts the YAML files produced by ``trw-mcp``'s learning system into
:class:`~trw_memory.models.memory.MemoryEntry` instances.

Field mapping
-------------
+-----------------------+------------------+------------------------------------------+
| LearningEntry field   | MemoryEntry field | Notes                                   |
+=======================+==================+==========================================+
| ``summary``           | ``content``      | Core knowledge statement                 |
| ``impact``            | ``importance``   | float 0-1                                |
| ``created``           | ``created_at``   | date → datetime midnight UTC             |
| ``updated``           | ``updated_at``   | date → datetime midnight UTC             |
| ``last_accessed_at``  | ``last_accessed_at`` | date|None → datetime|None           |
| ``status``            | ``status``       | active/resolved/obsolete pass through    |
| all other fields      | same name        | copied where the field exists            |
+-----------------------+------------------+------------------------------------------+

Fields absent from the source are replaced with :class:`MemoryEntry` defaults.
Unknown source fields are silently ignored.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import structlog
from ruamel.yaml.error import YAMLError

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.persistence import _new_yaml

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _date_to_dt(value: object) -> datetime:
    """Convert a ``date`` or ISO date string to a timezone-aware ``datetime``.

    Returns midnight UTC for date values.  Datetime values are returned as-is
    if already timezone-aware, or localised to UTC if naive.

    Raises:
        ValueError: If *value* cannot be coerced to a datetime.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    # Try ISO string
    text = str(value)
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    # Try date-only string YYYY-MM-DD
    try:
        d = date.fromisoformat(text[:10])
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"Cannot convert {value!r} to datetime") from exc


def _opt_date_to_dt(value: object) -> datetime | None:
    """Convert an optional date-like value to ``datetime | None``."""
    if value is None:
        return None
    try:
        return _date_to_dt(value)
    except ValueError:
        return None


def _safe_float(value: object, default: float) -> float:
    """Coerce *value* to float, returning *default* on failure."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return default


def _safe_int(value: object, default: int) -> int:
    """Coerce *value* to int, returning *default* on failure."""
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (ValueError, TypeError):
        return default


def _safe_str(value: object, default: str = "") -> str:
    """Coerce *value* to str, returning *default* for None."""
    if value is None:
        return default
    return str(value)


def _safe_str_list(value: object) -> list[str]:
    """Coerce *value* to ``list[str]``, returning ``[]`` on failure."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _safe_str_dict(value: object) -> dict[str, str]:
    """Coerce *value* to ``dict[str, str]``, returning ``{}`` on failure."""
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}


def _resolve_status(value: object) -> MemoryStatus:
    """Map a LearningEntry status string to :class:`MemoryStatus`.

    Handles the ``obsolete`` → ``OBSOLETE`` mapping and defaults unknown
    values to ``ACTIVE``.
    """
    mapping: dict[str, MemoryStatus] = {
        "active": MemoryStatus.ACTIVE,
        "resolved": MemoryStatus.RESOLVED,
        "obsolete": MemoryStatus.OBSOLETE,
        "archived": MemoryStatus.ARCHIVED,
    }
    if isinstance(value, str):
        return mapping.get(value.lower(), MemoryStatus.ACTIVE)
    return MemoryStatus.ACTIVE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_learning_entry(data: dict[str, object]) -> MemoryEntry:
    """Convert a raw LearningEntry YAML ``dict`` to a :class:`MemoryEntry`.

    The function is intentionally lenient — missing or invalid fields fall
    back to safe defaults so a single bad file does not abort a bulk
    migration.

    Args:
        data: Parsed YAML dict from a ``*.yaml`` file in the learnings entries
            directory.  Must contain at least ``summary`` and ``created``.

    Returns:
        A fully validated :class:`MemoryEntry` instance.
    """
    # --- required: id ---------------------------------------------------------
    entry_id = _safe_str(data.get("id"), "")
    if not entry_id:
        entry_id = str(uuid.uuid4())

    # --- content (was summary) ------------------------------------------------
    content = _safe_str(data.get("summary") or data.get("content"), "")

    # --- timestamps -----------------------------------------------------------
    now = datetime.now(timezone.utc)

    raw_created = data.get("created") or data.get("created_at")
    created_at = _date_to_dt(raw_created) if raw_created is not None else now

    raw_updated = data.get("updated") or data.get("updated_at")
    updated_at = _date_to_dt(raw_updated) if raw_updated is not None else created_at

    last_accessed_at = _opt_date_to_dt(data.get("last_accessed_at"))

    # --- importance (was impact) ----------------------------------------------
    raw_importance = data.get("impact") if "impact" in data else data.get("importance")
    importance = _safe_float(raw_importance, 0.5)
    importance = max(0.0, min(1.0, importance))

    # --- status ---------------------------------------------------------------
    status = _resolve_status(data.get("status"))

    # --- simple string fields -------------------------------------------------
    detail = _safe_str(data.get("detail"), "")
    namespace = _safe_str(data.get("namespace"), "default")
    source = _safe_str(data.get("source"), "agent")
    source_identity = _safe_str(data.get("source_identity"), "")
    client_profile = _safe_str(data.get("client_profile"), "")
    model_id = _safe_str(data.get("model_id"), "")
    consolidated_into: str | None = _safe_str(data.get("consolidated_into")) or None

    # --- list fields ----------------------------------------------------------
    tags = _safe_str_list(data.get("tags"))
    evidence = _safe_str_list(data.get("evidence"))
    merged_from = _safe_str_list(data.get("merged_from"))
    consolidated_from = _safe_str_list(data.get("consolidated_from"))

    # --- numeric fields -------------------------------------------------------
    recurrence = _safe_int(data.get("recurrence"), 1)
    access_count = _safe_int(data.get("access_count"), 0)
    session_count = _safe_int(data.get("session_count"), 0)
    q_value = _safe_float(data.get("q_value"), 0.5)
    q_value = max(0.0, min(1.0, q_value))
    q_observations = _safe_int(data.get("q_observations"), 0)

    # --- metadata dict --------------------------------------------------------
    metadata = _safe_str_dict(data.get("metadata"))

    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        tags=tags,
        evidence=evidence,
        importance=importance,
        status=status,
        recurrence=recurrence,
        namespace=namespace,
        created_at=created_at,
        updated_at=updated_at,
        last_accessed_at=last_accessed_at,
        access_count=access_count,
        session_count=session_count,
        q_value=q_value,
        q_observations=q_observations,
        source=source,
        source_identity=source_identity,
        client_profile=client_profile,
        model_id=model_id,
        merged_from=merged_from,
        consolidated_from=consolidated_from,
        consolidated_into=consolidated_into,
        metadata=metadata,
    )


def migrate_entries_dir(entries_dir: Path) -> list[MemoryEntry]:
    """Read all YAML files from *entries_dir* and convert to :class:`MemoryEntry`.

    Files that cannot be parsed or converted are logged as warnings and
    skipped; they do not abort the migration.

    Args:
        entries_dir: Directory containing ``*.yaml`` LearningEntry files
            (typically ``.trw/learnings/entries/``).

    Returns:
        List of :class:`MemoryEntry` objects in lexicographic filename order.
    """
    if not entries_dir.exists():
        logger.warning(
            "migration_source_dir_not_found",
            path=str(entries_dir),
        )
        return []

    yml = _new_yaml()
    entries: list[MemoryEntry] = []
    yaml_files = sorted(entries_dir.glob("*.yaml"))

    logger.info(
        "migration_start",
        source=str(entries_dir),
        file_count=len(yaml_files),
    )

    for yaml_path in yaml_files:
        try:
            raw: object = yml.load(yaml_path)
            if not isinstance(raw, dict):
                logger.warning(
                    "migration_skip_non_dict",
                    file=yaml_path.name,
                )
                continue
            data: dict[str, object] = {str(k): v for k, v in raw.items()}
            entry = from_learning_entry(data)
            entries.append(entry)
        except (OSError, ValueError, KeyError, TypeError, YAMLError):
            logger.warning(
                "migration_entry_failed",
                file=yaml_path.name,
                exc_info=True,
            )

    logger.info(
        "migration_complete",
        migrated=len(entries),
        skipped=len(yaml_files) - len(entries),
    )
    return entries
