"""Typed probe over the local Hugging Face cache (PRD-SEC-014-FR01).

The embedding loader must decide ``local_files_only`` *before* handing the model
name to sentence-transformers.  Until PRD-SEC-014 that decision read only the
config field and the two offline switches, so a machine with the whole snapshot
already on disk still permitted a huggingface.co revision check on the most
basic write path.

This module answers one narrow question — "is the configured model already
completely on this machine, where is it, and does its snapshot declare Python
modules?" — so the loader can load straight from that snapshot on a warm cache,
and so ``trw-mcp doctor`` can report cache state distinctly from egress posture.

Interface: :func:`probe_model_cache` returning a frozen :class:`CacheProbe`.
Everything else is private.  Expected degradations (huggingface_hub absent, an
unreadable or unrecognised cache layout) resolve to :attr:`CacheState.UNKNOWN`
rather than raising, because the probe must never fail a load that would
otherwise succeed (NFR02).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = ["CacheProbe", "CacheState", "probe_model_cache"]

# sentence-transformers resolves a bare model id against its own org before
# giving up, so ``all-MiniLM-L6-v2`` lives at ``sentence-transformers/...`` in
# the cache. Probing only the literal name would report ABSENT for the shipped
# default and re-enable the network fetch this PRD exists to remove.
_ST_ORG_PREFIX = "sentence-transformers/"

# Anchor file present in every transformers/sentence-transformers snapshot. Its
# cached path is also how the snapshot directory is located without enumerating
# the whole cache (``scan_cache_dir`` costs ~46 ms on a 35-repo cache; the
# anchor lookup costs ~0.1 ms and is the only path a warm load takes).
_ANCHOR_FILE = "config.json"

# ``auto_map`` is the transformers declaration that a repo's classes live in
# Python modules shipped by the repo itself — i.e. loading it executes code
# fetched from the Hub.
_REMOTE_CODE_CONFIG_KEY = "auto_map"

_PY_SUFFIX = ".py"


class CacheState(str, Enum):
    """Local-cache state of a configured embedding model."""

    COMPLETE = "complete"
    """Every file the snapshot declares is present and resolvable on disk."""

    INCOMPLETE = "incomplete"
    """The repo is cached but the snapshot is missing declared files/blobs."""

    ABSENT = "absent"
    """The repo has no cache entry on this machine."""

    UNKNOWN = "unknown"
    """The cache could not be inspected (dependency or layout unavailable)."""


@dataclass(frozen=True)
class CacheProbe:
    """Result of one cache inspection.

    Attributes:
        state: Typed cache state; ``COMPLETE`` is the only value that licenses
            forcing ``local_files_only=True`` on its own.
        declares_remote_code: ``True`` when the cached snapshot ships Python
            modules or declares ``auto_map`` — loading it would execute
            repo-supplied code.
        snapshot_path: Absolute path of the resolved snapshot directory when the
            state is ``COMPLETE``, else empty. Callers load from this path rather
            than from the repo id, because ``local_files_only=True`` is NOT
            sufficient to prevent a Hub request (see
            :meth:`~trw_memory.embeddings.local.LocalEmbeddingProvider._load_model`).
    """

    state: CacheState
    declares_remote_code: bool = False
    snapshot_path: str = ""


def _resolve_cache_dir() -> str | None:
    """Return the hub cache directory, honoring env overrides set after import.

    ``huggingface_hub.constants.HF_HUB_CACHE`` is frozen at import time, so a
    test (or a wrapper process) that exports ``HF_HOME`` afterwards would be
    silently ignored. Resolving the documented env precedence here keeps the
    probe answering about the cache the loader will actually read.
    """
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if override := os.environ.get(var, "").strip():
            return override
    if hf_home := os.environ.get("HF_HOME", "").strip():
        return os.path.join(hf_home, "hub")
    try:
        from huggingface_hub import constants
    except ImportError:
        return None
    return str(constants.HF_HUB_CACHE)


def _candidate_repo_ids(model_name: str) -> tuple[str, ...]:
    """Return the repo ids sentence-transformers would try, in its order."""
    name = model_name.strip().strip("/")
    if not name:
        return ()
    if "/" in name:
        return (name,)
    return (name, f"{_ST_ORG_PREFIX}{name}")


def _declares_remote_code(snapshot: Path, py_module_seen: bool) -> bool:
    if py_module_seen:
        return True
    config_path = snapshot / _ANCHOR_FILE
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and bool(payload.get(_REMOTE_CODE_CONFIG_KEY))


def _inspect_snapshot(snapshot: Path) -> CacheProbe:
    """Classify a snapshot directory by walking the files it declares.

    A snapshot entry is a symlink into ``blobs/``; a deleted blob leaves a
    dangling link. That is the "partially deleted blob" case the PRD requires
    to be treated as INCOMPLETE, never as COMPLETE — an offline load of it
    would fail rather than silently succeed.
    """
    complete = True
    saw_file = False
    py_module_seen = False
    try:
        entries = sorted(snapshot.rglob("*"))
    except OSError:
        return CacheProbe(CacheState.UNKNOWN)
    for entry in entries:
        if not entry.exists():  # dangling symlink -> declared file, missing blob
            complete = False
            continue
        if entry.is_dir():
            continue
        saw_file = True
        if entry.suffix == _PY_SUFFIX:
            py_module_seen = True
    if not saw_file:
        # A directory that exists but holds nothing is INCOMPLETE, not ABSENT:
        # ABSENT means "nothing for this model on this machine", which would let
        # a caller conclude the cache is simply cold. An empty snapshot/model
        # directory is the more alarming case — something IS there and is not
        # usable — and the two resolve to the same conservative behaviour
        # (no cache-first load), so the distinction only affects what the doctor
        # row tells the operator. It should not tell them "absent".
        return CacheProbe(CacheState.INCOMPLETE)
    if not complete:
        return CacheProbe(CacheState.INCOMPLETE, _declares_remote_code(snapshot, py_module_seen))
    return CacheProbe(
        CacheState.COMPLETE,
        _declares_remote_code(snapshot, py_module_seen),
        str(snapshot.resolve()),
    )


def _cached_snapshot_dir(repo_id: str, cache_dir: str | None) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(repo_id=repo_id, filename=_ANCHOR_FILE, cache_dir=cache_dir)
    if isinstance(hit, str) and os.path.isfile(hit):
        return Path(hit).parent
    return None


def _repo_is_cached(repo_ids: tuple[str, ...], cache_dir: str | None) -> bool:
    """Return True when any candidate repo has a cache entry at all.

    Only reached when the anchor lookup missed, i.e. on the cold path that is
    about to hit the network anyway — ``scan_cache_dir`` enumerates the whole
    cache and is far too slow for the warm path (NFR01).
    """
    from huggingface_hub import scan_cache_dir

    try:
        info = scan_cache_dir(cache_dir=cache_dir)
    except Exception:  # justified: CacheNotFound + any layout/IO error -> not cached
        return False
    wanted = {repo_id.lower() for repo_id in repo_ids}
    return any(repo.repo_id.lower() in wanted for repo in info.repos)


def probe_model_cache(model_name: str) -> CacheProbe:
    """Classify the local cache state of *model_name*.

    Args:
        model_name: HuggingFace model id (``org/name`` or a bare id resolved
            against the sentence-transformers org), or a local directory path.

    Returns:
        A :class:`CacheProbe`. ``UNKNOWN`` when huggingface_hub is unavailable
        or the cache cannot be located, so callers degrade to their prior
        resolution instead of failing (NFR02).
    """
    repo_ids = _candidate_repo_ids(model_name)
    if not repo_ids:
        return CacheProbe(CacheState.ABSENT)

    local_dir = Path(model_name).expanduser()
    if local_dir.is_dir():
        # An explicit on-disk model directory needs no Hub round-trip at all.
        return _inspect_snapshot(local_dir)

    cache_dir = _resolve_cache_dir()
    try:
        for repo_id in repo_ids:
            snapshot = _cached_snapshot_dir(repo_id, cache_dir)
            if snapshot is not None:
                return _inspect_snapshot(snapshot)
        cached = _repo_is_cached(repo_ids, cache_dir)
    except ImportError:
        return CacheProbe(CacheState.UNKNOWN)
    except OSError:
        return CacheProbe(CacheState.UNKNOWN)
    return CacheProbe(CacheState.INCOMPLETE if cached else CacheState.ABSENT)
