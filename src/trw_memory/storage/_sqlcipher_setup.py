"""SQLCipher driver loading + pragma application + cold-rebuild path resolution.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — the parent module re-imports the names so its own
``_apply_sqlcipher_pragmas`` / ``_resolve_cold_rebuild_base`` /
``_import_sqlcipher_driver`` references continue working.

3 module-level helpers:

- ``import_sqlcipher_driver`` — try sqlcipher3.dbapi2 then
  pysqlcipher3.dbapi2; raise ``EncryptionUnavailableError`` on
  neither-available.
- ``apply_sqlcipher_pragmas`` — emit cipher / cipher_page_size /
  kdf_iter pragmas onto an opened connection.
- ``resolve_cold_rebuild_base`` — choose base directory whose
  ``memory/cold`` subtree should drive recovery rebuild.

Constants ``SQLCIPHER_REQUIRED_MESSAGE`` / ``SQLCIPHER_CIPHER`` /
``SQLCIPHER_CIPHER_PAGE_SIZE`` / ``SQLCIPHER_KDF_ITER`` re-exported
for back-compat.

Extracted as PRD-DIST-245 Phase 1 batch 90.
"""

from __future__ import annotations

import contextlib
import importlib
from pathlib import Path
from typing import Any

import structlog

from trw_memory.exceptions import EncryptionUnavailableError

logger = structlog.get_logger(__name__)

SQLCIPHER_REQUIRED_MESSAGE = (
    "SQLCipher driver not installed. Install one of: 'sqlcipher3-binary' "
    "or 'pysqlcipher3'. See trw-memory README §Encryption for details."
)
SQLCIPHER_CIPHER = "aes-256-cbc"
SQLCIPHER_CIPHER_PAGE_SIZE = 4096
SQLCIPHER_KDF_ITER = 256000


def import_sqlcipher_driver() -> Any:
    """Return a SQLCipher DB-API module or raise the standard startup error."""
    for module_name in ("sqlcipher3.dbapi2", "pysqlcipher3.dbapi2"):
        with contextlib.suppress(ImportError):
            return importlib.import_module(module_name)
    raise EncryptionUnavailableError(SQLCIPHER_REQUIRED_MESSAGE)


def apply_sqlcipher_pragmas(conn: Any) -> None:
    """Apply the explicit SQLCipher settings required by the PRD."""
    conn.execute(f"PRAGMA cipher = '{SQLCIPHER_CIPHER}'")
    conn.execute(f"PRAGMA cipher_page_size = {SQLCIPHER_CIPHER_PAGE_SIZE}")
    conn.execute(f"PRAGMA kdf_iter = {SQLCIPHER_KDF_ITER}")


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
