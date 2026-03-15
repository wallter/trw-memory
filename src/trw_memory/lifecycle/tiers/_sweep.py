"""Sweep logic for tier lifecycle transitions.

Implements the three-phase sweep: Hot->Warm, Warm->Cold, Cold->Purge.
Called by TierManager.sweep() with the manager's internal state.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timezone
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


def execute_sweep(  # noqa: C901
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
    cfg = config
    today = datetime.now(tz=timezone.utc).date()
    promoted = 0
    demoted = 0
    purged = 0
    errors = 0

    purge_audit_path = base_dir / "memory" / "purge_audit.jsonl"

    # 1. Hot -> Warm: evict stale hot entries
    stale_hot_ids: list[str] = []
    for entry_id, entry in list(hot.items()):
        days = _days_since_access(entry.model_dump(), today)
        if days > cfg.hot_ttl_days:
            stale_hot_ids.append(entry_id)

    for entry_id in stale_hot_ids:
        try:
            evicted = hot.pop(entry_id)
            warm_add_fn(entry_id, evicted.model_dump(), None)
            demoted += 1
            logger.debug("sweep_hot_to_warm", entry_id=entry_id)
        except (OSError, StorageError, ValueError):  # per-item error handling: one failed eviction must not abort the sweep  # noqa: PERF203
            logger.warning("sweep_hot_to_warm_failed", entry_id=entry_id, exc_info=True)
            errors += 1

    # 2. Warm -> Cold: scan entries directory for idle low-importance entries
    if entries_dir.exists():
        for yaml_file in sorted(entries_dir.glob("*.yaml")):
            if yaml_file.name == "index.yaml":
                continue
            try:
                data = read_yaml(yaml_file)
                entry_id = str(data.get("id", ""))
                if not entry_id:
                    continue
                # Skip non-active entries
                if str(data.get("status", "active")) != "active":
                    continue
                days = _days_since_access(data, today)
                importance = compute_importance_score(data, [], config=cfg)
                if days > cfg.cold_threshold_days and importance < 0.22:
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

    # 3. Cold -> Purge: scan cold archive for expired entries
    if cold_dir.exists():
        for yaml_file in sorted(cold_dir.rglob("*.yaml")):
            try:
                data = read_yaml(yaml_file)
                entry_id = str(data.get("id", ""))
                days = _days_since_access(data, today)
                importance = compute_importance_score(data, [], config=cfg)
                if days > cfg.retention_days and importance < 0.1:
                    # Append to purge audit log before deleting
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
            except (OSError, StorageError, ValueError):  # per-item error handling: skip unreadable files, continue sweep  # noqa: PERF203
                logger.warning(
                    "sweep_cold_purge_failed",
                    path=str(yaml_file),
                    exc_info=True,
                )
                errors += 1

    logger.info(
        "tier_sweep_complete",
        promoted=promoted,
        demoted=demoted,
        purged=purged,
        errors=errors,
    )
    return TierSweepResult(
        promoted=promoted,
        demoted=demoted,
        purged=purged,
        errors=errors,
    )
