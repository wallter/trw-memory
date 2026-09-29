"""Where a DEFAULT store and its security state live: beside the config that governs them.

``MemoryConfig`` records the ``.trw`` it loaded ``config.yaml`` from (``source_trw_dir``: the
nearest one at or above the cwd, never HOME's machine tier or a temp root's). A default store is ``<that project>/.memory`` and its derived
security files are ``<that .trw>/security/...``, so RBAC, sync and the contact switch read by
that same config object always govern the store it opens. ``TRW_DIR``/``TRW_PROJECT_ROOT`` never
steer a default store, and without a ``.trw`` a default write is refused before touching disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from trw_memory.exceptions import StorageRootUnresolvableError
from trw_memory.security.startup import resolve_security_path

if TYPE_CHECKING:
    from trw_memory.models.config import MemoryConfig

__all__ = ["resolve_state_path", "resolve_storage_root"]


def _is_default_storage(config: MemoryConfig) -> bool:
    return "storage_path" not in config.model_fields_set and not Path(config.storage_path).is_absolute()


def _governing_trw_dir(config: MemoryConfig) -> Path:
    if config.source_trw_dir is None:
        raise StorageRootUnresolvableError(
            f"the default storage_path {config.storage_path!r} needs a project anchor: run inside a project "
            "(a directory holding .trw, or below one; its config governs the store), or pass an explicit storage_path"
        )
    return config.source_trw_dir


def resolve_storage_root(config: MemoryConfig) -> Path:
    """The directory a store is written under -- the one waist every store writer goes through.

    An explicit ``storage_path`` (constructor or ``MEMORY_STORAGE_PATH``) is returned unchanged,
    relative or not. The default is ``<project>/.memory`` beside ``config.source_trw_dir``.
    """
    if not _is_default_storage(config):
        return Path(config.storage_path).expanduser()
    return _governing_trw_dir(config).parent / config.storage_path


def resolve_state_path(config: MemoryConfig, field_name: str) -> Path:
    """A security file derived from ``storage_path`` (audit log, rate-limit state, quarantine dir).

    A default-path config puts it under ``<source .trw>/`` (refused with no project); any other
    config resolves it through the SEC-001 anchor, unchanged.
    """
    raw = Path(getattr(config, field_name)).expanduser()
    if raw.is_absolute() or not _is_default_storage(config):
        return resolve_security_path(config, field_name)
    return _governing_trw_dir(config) / raw
