"""Where a namespace's tier files live now, and where a pre-fix orphan might still be.

**Responsibility.** Before the tier-root fix, a namespace's tier directory was
always ``resolve_storage_root(config)/<namespace>``, even when a set
``memory_single_store_path`` pointed the CANONICAL backend somewhere else. A
project moved onto a single store without also moving ``storage_path``
therefore left its tier files (``warm.db``, the cold ``entries/`` archive)
behind at the OLD root while new writes land at the single store's own
directory. Both roots are PER-NAMESPACE, before and after: the canonical
backend is safe to share across namespaces (rows keyed on
``(namespace, id)``), but :class:`~trw_memory.lifecycle.tiers._warm.WarmTierStore`
is not (see its module docstring), so this mapping must preserve that
boundary, never collapse it.

**No copy.** A leftover ``warm.db`` is never copied, moved, or deleted by
trw-memory: the warm tier rebuilds itself from the canonical backend on next
use (:func:`trw_memory.lifecycle.tiers._runtime.warmup_tier_manager`), so an
orphan file is simply unused, not lost data. A stranded cold-tier ``entries/``
archive is DATA (rows not yet visible to the current tier runtime) and is
never touched here either -- see the ``memory_warm_legacy`` doctor row in
trw-mcp, which reports both, read-only, and names trw-memory's
:func:`legacy_tier_dirs` as its one source of truth for the mapping.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory._project_anchor import resolve_storage_root
from trw_memory.models.config import MemoryConfig

__all__ = ["legacy_tier_dirs", "tier_root_dir"]


def tier_root_dir(config: MemoryConfig) -> Path:
    """The ONE base directory every namespace's tier subdirectory sits under.

    Under a set ``memory_single_store_path`` this is the directory CONTAINING
    that one file -- which can diverge from ``resolve_storage_root(config)``
    when the single store lives somewhere other than the per-namespace root.
    Otherwise it is ``resolve_storage_root(config)``, unchanged from every
    layout before ``memory_single_store_path`` existed.

    The single helper both :func:`trw_memory.lifecycle.tiers._runtime.namespace_storage_dir`
    and :func:`legacy_tier_dirs` resolve through, so the two can never drift.
    """
    if config.memory_single_store_path:
        return Path(config.memory_single_store_path).expanduser().resolve().parent
    return resolve_storage_root(config).resolve()


def legacy_tier_dirs(config: MemoryConfig) -> list[tuple[Path, Path]]:
    """Every ``(legacy_dir, target_dir)`` pair a namespace's tier files MIGHT be
    scattered across -- the single source of truth a reader (the trw-mcp doctor
    row) uses instead of re-deriving these paths itself.

    Returns ``[]`` when there is no possible divergence: no ``memory_single_store_path``
    is set, or the old and new roots happen to coincide (the common daemon
    layout) -- every per-namespace directory found is then the ACTIVE target,
    never a legacy orphan (HB-2: never point an operator at live data).

    A returned pair does not by itself mean a legacy file exists -- the caller
    still checks ``legacy_dir`` for ``memory/warm.db`` or ``entries/``.
    """
    if not config.memory_single_store_path:
        return []
    old_root = resolve_storage_root(config).resolve()
    new_root = tier_root_dir(config)
    if old_root == new_root or not old_root.exists():
        return []
    return [(candidate, new_root / candidate.name) for candidate in sorted(old_root.iterdir()) if candidate.is_dir()]
