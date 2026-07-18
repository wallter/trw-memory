"""SEC-001 startup validation and anchored path resolution."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

import structlog

from trw_memory.exceptions import CanaryFixturesMissingError, SecurityDefaultUnresolvableError
from trw_memory.models.config import MemoryConfig

logger = structlog.get_logger(__name__)

_PACKAGE_FIXTURE_PACKAGE = "trw_memory.security"
_CANARY_FIXTURE_DIRNAME = "fixtures"


def _discover_anchor(config: MemoryConfig, *, trw_dir: Path | None = None) -> Path:
    if trw_dir is not None:
        return trw_dir.resolve()
    env_trw_dir = os.environ.get("TRW_DIR", "").strip()
    if env_trw_dir:
        return Path(env_trw_dir).expanduser().resolve()
    storage_root = Path(config.storage_path)
    if storage_root.is_absolute():
        return storage_root.resolve().parent
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        found = candidate / ".trw"
        if found.exists():
            return found.resolve()
    raise SecurityDefaultUnresolvableError("unable to resolve SEC-001 anchor; set TRW_DIR or use absolute config paths")


def resolve_security_path(
    config: MemoryConfig,
    field_name: str,
    *,
    trw_dir: Path | None = None,
    create_parent: bool = False,
    reject_leaf_symlink: bool = False,
) -> Path:
    """Resolve a SEC-001 path without depending on the current working directory."""
    raw_value = getattr(config, field_name)
    if not isinstance(raw_value, str) or not raw_value:
        raise SecurityDefaultUnresolvableError(f"{field_name} is empty")
    if raw_value.startswith("package:"):
        package_root = resources.files(_PACKAGE_FIXTURE_PACKAGE)
        resolved = Path(str(package_root / _CANARY_FIXTURE_DIRNAME)).resolve()
    else:
        candidate = Path(raw_value).expanduser()
        anchored = candidate if candidate.is_absolute() else _discover_anchor(config, trw_dir=trw_dir) / candidate
        if reject_leaf_symlink:
            resolved = anchored.parent.resolve() / anchored.name
            if resolved.is_symlink():
                raise SecurityDefaultUnresolvableError(f"{field_name} must not be a symlink: {resolved}")
        else:
            resolved = anchored.resolve()
    if create_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def verify_defaults(config: MemoryConfig, *, trw_dir: Path | None = None) -> None:
    """Fail loud if SEC-001 defaults do not resolve to real writable artifacts."""
    canary_dir = resolve_security_path(config, "canary_fixtures_path", trw_dir=trw_dir)
    if not canary_dir.exists() or not canary_dir.is_dir():
        raise CanaryFixturesMissingError(f"canary fixtures missing at {canary_dir}")

    quarantine_db = resolve_security_path(config, "quarantine_db_path", trw_dir=trw_dir, create_parent=True)
    try:
        quarantine_db.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SecurityDefaultUnresolvableError(f"quarantine DB path not creatable: {quarantine_db}") from exc

    signing_key = resolve_security_path(
        config,
        "provenance_signing_key_path",
        trw_dir=trw_dir,
        create_parent=True,
        reject_leaf_symlink=True,
    )
    try:
        signing_key.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SecurityDefaultUnresolvableError(f"provenance key path not creatable: {signing_key}") from exc

    logger.info(
        "memory_security_defaults_verified",
        component="memory_security",
        op="startup_verify_defaults",
        outcome="success",
        quarantine_db_path=str(quarantine_db),
        provenance_signing_key_path=str(signing_key),
        canary_fixtures_path=str(canary_dir),
    )
