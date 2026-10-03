"""The platform contact switch, resolved in one place for trw-memory and trw-mcp (B71-106, B71-107).

``platform_contact_enabled`` is the operator's kill switch for every contact with the TRW platform:
sync, telemetry, feedback, the update check and the learning stream. Both packages used to resolve it
themselves, and differently. trw-memory let a project ``null`` override a machine ``false`` and raised
on an invalid value, while trw-mcp fell back to defaults (contact on). Both read it once at startup,
so turning it off needed a restart. Every network boundary in both packages now calls
:func:`platform_contact_enabled` at the moment it would connect.

Layers, first explicit answer wins: ``TRW_PLATFORM_CONTACT_ENABLED`` in the environment, then the
project's ``.trw/config.yaml``, then ``~/.trw/config.yaml``, then on. A missing key, a ``null`` or a
blank value says nothing and falls through. Values follow pydantic's bool rules, so ``no`` and ``off``
mean off. A value that is not a bool, or a config file that cannot be read, resolves to OFF with a
warning: this switch only ever restricts, so an operator's unreadable intent must not become contact
(trw_assess fail_closed 0.99). The value itself is never logged.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from pathlib import Path

import structlog
from pydantic import TypeAdapter, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

__all__ = ["CONTACT_ENV", "CONTACT_KEY", "platform_contact_enabled", "team_sync_enabled"]

logger = structlog.get_logger(__name__)

CONTACT_KEY = "platform_contact_enabled"
CONTACT_ENV = "TRW_PLATFORM_CONTACT_ENABLED"
TEAM_SYNC_KEY = "team_sync_enabled"
TEAM_SYNC_ENV = "TRW_TEAM_SYNC_ENABLED"
_BOOL: TypeAdapter[bool] = TypeAdapter(bool)


def platform_contact_enabled(project_root: Path | str | None = None, env: Mapping[str, str] | None = None) -> bool:
    """Whether platform contact is allowed right now. Cheap enough to call at every connection.

    ``project_root`` names the project; a sender passes the root of the config that enabled it.
    Without it, ``TRW_PROJECT_ROOT`` does, else the project ``MemoryConfig`` loading finds (the
    nearest ``.trw`` at or above the working directory, never HOME's or a temp root's). Never raises: any
    fault while resolving (a working directory removed between retries, say) turns contact off.
    """
    try:
        return _resolve(project_root, os.environ if env is None else env)
    except Exception:  # justified: fail closed, a kill switch that cannot be read must not allow contact
        with contextlib.suppress(Exception):  # trw-fail-silent-allow: even a failing log must not escape
            logger.warning("platform_contact_resolution_failed", outcome="contact_off", exc_info=True)
        return False


def team_sync_enabled(project_root: Path | str | None = None, env: Mapping[str, str] | None = None) -> bool:
    """Whether team sync (the consent that turns on shared recall) is on right now, read live.

    Same layers as :func:`platform_contact_enabled` (environment, project config, machine config),
    but the default is OFF and every fault resolves to OFF: a recall query may leave the machine only
    on an explicit yes, never on a missing key or an unreadable file.
    """
    try:
        return _resolve(project_root, os.environ if env is None else env, TEAM_SYNC_KEY, TEAM_SYNC_ENV, default=False)
    except Exception:  # justified: fail closed, a consent that cannot be read grants nothing
        with contextlib.suppress(Exception):  # trw-fail-silent-allow: even a failing log must not escape
            logger.warning("team_sync_resolution_failed", outcome="query_stays_local", exc_info=True)
        return False


def _resolve(
    project_root: Path | str | None,
    environ: Mapping[str, str],
    key: str = CONTACT_KEY,
    env_name: str = CONTACT_ENV,
    *,
    default: bool = True,
) -> bool:
    root = Path(project_root) if project_root else _project_root(environ)
    layers = (("environment", None), ("project config", root), ("machine config", Path.home()))
    for layer, where in layers:
        try:
            raw = environ.get(env_name) if where is None else _config_value(where / ".trw" / "config.yaml", key)
        except (OSError, YAMLError, UnicodeDecodeError, ValueError):
            logger.warning("platform_contact_config_unreadable", layer=layer, outcome="contact_off")
            return False
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue  # says nothing: the next layer decides
        try:
            return _BOOL.validate_python(raw)
        except ValidationError:
            logger.warning(
                "platform_contact_value_invalid",
                layer=layer,
                key=key,
                outcome="contact_off" if key == CONTACT_KEY else "query_stays_local",
            )
            return False
    return default


def _project_root(environ: Mapping[str, str]) -> Path:
    if named := environ.get("TRW_PROJECT_ROOT"):
        return Path(named)
    from trw_memory.models._config_sources import _project_trw_dir  # the one finder config loading uses

    found = _project_trw_dir()
    return found.parent if found is not None else Path.cwd()


def _config_value(path: Path, key: str = CONTACT_KEY) -> object:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        loaded = YAML(typ="safe").load(handle)
    if loaded is not None and not isinstance(loaded, dict):  # a list or bare scalar is not a config (sol P2)
        raise ValueError(f"{path} is not a mapping")
    return loaded.get(key) if loaded else None
