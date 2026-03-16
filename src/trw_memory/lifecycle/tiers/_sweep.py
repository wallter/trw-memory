"""Sweep logic for tier lifecycle transitions.

Implements the three-phase sweep: Hot->Warm, Warm->Cold, Cold->Purge.
Called by TierManager.sweep() with the manager's internal state.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.lifecycle._utils import days_since_access as _days_since_access
from trw_memory.lifecycle.tiers._scoring import TierSweepResult, compute_importance_score
from trw_memory.models.config import MemoryConfig
from trw_memory.storage.persistence import read_yaml

if TYPE_CHECKING:
    from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger()


def _sweep_hot_to_warm(
    hot: OrderedDict[str, MemoryEntry],
    config: MemoryConfig,
    today: date,
    warm_add_fn: Callable[[str, dict[str, object], list[float] | None], None],
) -> tuple[int, int]:
    """Evict stale hot entries and promote to warm tier.

    Returns (demoted_count, error_count).
    """
    demoted = 0
    errors = 0

    stale_hot_ids = [
        entry_id
        for entry_id, entry in list(hot.items())
        if _days_since_access(entry.model_dump(), today) > config.hot_ttl_days
    ]

    for entry_id in stale_hot_ids:
        try:
            evicted = hot.pop(entry_id)
            warm_add_fn(entry_id, evicted.model_dump(), None)
            demoted += 1
            logger.debug("sweep_hot_to_warm", entry_id=entry_id)
        except (OSError, StorageError, ValueError):  # noqa: PERF203 — per-entry error handling
            logger.warning("sweep_hot_to_warm_failed", entry_id=entry_id, exc_info=True)
            errors += 1

    return demoted, errors


def _sweep_warm_to_cold(
    entries_dir: Path,
    config: MemoryConfig,
    today: date,
    cold_archive_fn: Callable[[str, Path], None],
) -> tuple[int, int]:
    """Demote idle low-importance warm entries to cold tier.

    Returns (demoted_count, error_count).
    """
    demoted = 0
    errors = 0

    if not entries_dir.exists():
        return demoted, errors

    for yaml_file in sorted(entries_dir.glob("*.yaml")):
        if yaml_file.name == "index.yaml":
            continue
        try:
            data = read_yaml(yaml_file)
            entry_id = str(data.get("id", ""))
            if not entry_id or str(data.get("status", "active")) != "active":
                continue

            days = _days_since_access(data, today)
            importance = compute_importance_score(data, [], config=config)
            if days > config.cold_threshold_days and importance < 0.22:
                cold_archive_fn(entry_id, yaml_file)
                demoted += 1
                logger.debug(
                    "sweep_warm_to_cold",
                    entry_id=entry_id,
                    days=days,
                    importance_score=importance,
                )
        except (OSError, StorageError, ValueError):
            logger.warning(
                "sweep_warm_to_cold_failed",
                path=str(yaml_file),
                exc_info=True,
            )
            errors += 1

    return demoted, errors


def _sweep_cold_to_purge(
    cold_dir: Path,
    config: MemoryConfig,
    today: date,
    purge_audit_path: Path,
) -> tuple[int, int]:
    """Purge expired cold entries with audit trail.

    Returns (purged_count, error_count).
    """
    purged = 0
    errors = 0

    if not cold_dir.exists():
        return purged, errors

    for yaml_file in sorted(cold_dir.rglob("*.yaml")):
        try:
            data = read_yaml(yaml_file)
            entry_id = str(data.get("id", ""))
            days = _days_since_access(data, today)
            importance = compute_importance_score(data, [], config=config)

            if days > config.retention_days and importance < 0.1:
                audit_record: dict[str, object] = {
                    "entry_id": entry_id,
                    "purged_at": datetime.now(timezone.utc).isoformat(),
                    "days_idle": days,
                    "importance_score": importance,
                    "importance": float(str(data.get("importance", data.get("impact", 0.5)))),
                    "content": str(data.get("content", data.get("summary", ""))),
                }
                purge_audit_path.parent.mkdir(parents=True, exist_ok=True)
                with purge_audit_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(audit_record) + "\n")
                yaml_file.unlink(missing_ok=True)
                purged += 1
                logger.debug(
                    "sweep_cold_purge",
                    entry_id=entry_id,
                    days=days,
                    importance_score=importance,
                )
        except (OSError, StorageError, ValueError):  # noqa: PERF203 — per-entry error handling
            logger.warning(
                "sweep_cold_purge_failed",
                path=str(yaml_file),
                exc_info=True,
            )
            errors += 1

    return purged, errors


def execute_sweep(
    *,
    hot: OrderedDict[str, MemoryEntry],
    config: MemoryConfig,
    entries_dir: Path,
    base_dir: Path,
    warm_add_fn: Callable[[str, dict[str, object], list[float] | None], None],
    cold_archive_fn: Callable[[str, Path], None],
    cold_dir: Path,
) -> TierSweepResult:
    """Execute lifecycle sweep across all tiers.

    Performs three transition checks in order:
    1. Hot -> Warm: entries whose last_accessed_at exceeds hot_ttl_days.
    2. Warm -> Cold: entries idle > cold_threshold_days with importance < 0.22.
    3. Cold -> Purge: entries idle > retention_days with importance < 0.1.

    Args:
        hot: The hot tier OrderedDict (mutated in-place for evictions).
        config: MemoryConfig for threshold settings.
        entries_dir: Directory containing warm-tier YAML entries.
        base_dir: Base directory for purge audit log.
        warm_add_fn: Callable(entry_id, entry_data, embedding) for warm add.
        cold_archive_fn: Callable(entry_id, entry_path) for cold archive.
        cold_dir: Base cold archive directory path.

    Returns:
        TierSweepResult with counts of promoted, demoted, purged, and errors.
    """
    today = datetime.now(tz=timezone.utc).date()
    purge_audit_path = base_dir / "memory" / "purge_audit.jsonl"

    demoted_hot, errors_hot = _sweep_hot_to_warm(hot, config, today, warm_add_fn)
    demoted_warm, errors_warm = _sweep_warm_to_cold(entries_dir, config, today, cold_archive_fn)
    purged, errors_cold = _sweep_cold_to_purge(cold_dir, config, today, purge_audit_path)

    total_errors = errors_hot + errors_warm + errors_cold
    total_demoted = demoted_hot + demoted_warm

    logger.info(
        "tier_sweep_complete",
        promoted=0,
        demoted=total_demoted,
        purged=purged,
        errors=total_errors,
    )
    return TierSweepResult(
        promoted=0,
        demoted=total_demoted,
        purged=purged,
        errors=total_errors,
    )
