"""Cold-rebuild base-directory resolution.

Belongs to the ``sqlite_backend.py`` facade, which re-exports
``resolve_cold_rebuild_base`` as ``_resolve_cold_rebuild_base``. Extracted as
PRD-DIST-245 Phase 1 batch 90; the SQLCipher driver and pragma helpers that
shared this module were removed with encryption at rest (trw-memory 4.1).
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def resolve_cold_rebuild_base(db_path: Path) -> Path:
    """Return the base directory whose ``memory/cold`` subtree should rebuild.

    ``rebuild_from_cold(base_dir, conn)`` intentionally reads
    ``base_dir / "memory" / "cold"``. Two layouts are live:

    - standalone ``trw-memory`` tests/CLI: ``<base>/memory.db`` and
      ``<base>/memory/cold``.
    - ``trw-mcp`` runtime: ``<trw_dir>/memory/memory.db`` and
      ``<trw_dir>/memory/cold``.

    The second layout was the 2026-04-28 incident: using
    ``db_path.parent`` made recovery look under
    ``<trw_dir>/memory/memory/cold`` and rebuild zero rows. Prefer the
    production-shaped parent when it exists; otherwise preserve the
    standalone default.
    """
    standalone_base = db_path.parent
    candidates = [standalone_base]
    if db_path.parent.name == "memory" or (db_path.parent / "cold").exists():
        trw_dir_base = db_path.parent.parent
        if trw_dir_base != standalone_base:
            candidates.append(trw_dir_base)

    candidate_counts: list[tuple[Path, int]] = []
    for candidate in candidates:
        cold_dir = candidate / "memory" / "cold"
        yaml_count = sum(1 for _ in cold_dir.rglob("*.yaml")) if cold_dir.is_dir() else 0
        candidate_counts.append((candidate, yaml_count))

    non_empty = [item for item in candidate_counts if item[1] > 0]
    if non_empty:
        selected_base, selected_count = max(non_empty, key=lambda item: item[1])
    elif db_path.parent.name == "memory" and db_path.name == "memory.db":
        selected_base, selected_count = candidate_counts[-1]
    else:
        selected_base, selected_count = candidate_counts[0]

    logger.info(
        "cold_rebuild_base_selected",
        db_path=str(db_path),
        selected_base_dir=str(selected_base),
        selected_yaml_count=selected_count,
        candidates=[
            {
                "base_dir": str(candidate),
                "cold_dir": str(candidate / "memory" / "cold"),
                "yaml_count": yaml_count,
            }
            for candidate, yaml_count in candidate_counts
        ],
    )
    return selected_base
