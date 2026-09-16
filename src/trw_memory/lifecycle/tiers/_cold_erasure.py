"""Cold-tier erasure — the GDPR ``forget`` path.

Belongs to the ``_cold.py`` facade, which keeps ``cold_remove`` as a one-line
delegator so ``TierManager.cold_remove`` and every test that patches
``trw_memory.lifecycle.tiers._cold.read_yaml`` keep working unchanged.

Extracted for the 350 effective-LOC gate: making the erasure gap VISIBLE (a
candidate file that cannot be read may hold the entry being erased) added the
code that pushed ``_cold.py`` from 347 to 353, and this is the smallest
self-contained unit in it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from trw_memory.exceptions import StorageError

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)

__all__ = ["cold_remove"]


def cold_remove(cold_base: Path, entry_id: str, search_cache: dict[str, Any] | Any) -> int:
    """Unlink every archived YAML whose ``id`` matches ``entry_id``.

    Args:
        cold_base: Root of the cold partition tree.
        entry_id: Memory entry identifier to erase from cold storage.
        search_cache: The tier's search cache; entries for unlinked files are
            evicted so a later search cannot resurrect a deleted memory.

    Returns:
        Count of cold YAML files removed (0 if the entry was not archived).

        NOT proof of erasure. A candidate that could not be READ is skipped and
        may have held the entry; those are logged at warning, per file and once
        in summary, because the count cannot express them. Unlink failures are
        likewise logged and counted as not removed.
    """
    # Resolved through the parent facade so the many tests that patch
    # `_cold.read_yaml` keep reaching this code path.
    from trw_memory.lifecycle.tiers import _cold as _facade

    if not cold_base.exists():
        return 0

    removed = 0
    unreadable: list[str] = []
    for yaml_file in sorted(cold_base.rglob("*.yaml")):
        try:
            data = _facade.read_yaml(yaml_file)
        except (OSError, StorageError):
            # Skipped without a trace until 2026-09-12: on an erasure path a file
            # we cannot READ may BE the entry, so silence here is the difference
            # between "not archived" and "possibly still on disk".
            unreadable.append(yaml_file.name)
            logger.warning("cold_remove_unreadable_candidate", entry_id=entry_id, path=str(yaml_file), exc_info=True)
            continue
        if str(data.get("id", "")) != entry_id:
            continue
        try:
            yaml_file.unlink(missing_ok=True)
        except OSError:
            logger.warning("cold_remove_unlink_failed", entry_id=entry_id, path=str(yaml_file), exc_info=True)
            continue
        search_cache.pop(str(yaml_file), None)
        removed += 1
        logger.debug("cold_remove", entry_id=entry_id, path=str(yaml_file))
    if unreadable:
        # UNVERIFIED, not merely partial: any of these may have held the entry.
        logger.warning("cold_remove_incomplete", entry_id=entry_id, removed=removed, unreadable=unreadable)
    return removed
