"""The daemon tools' one embedder resolution, with the contract C1 ``unavailable`` answer (PRD-CORE-302).

``get_local_embedder`` never caches a failed load (``_provider_cache``), so a model
fetched after a refusal is picked up by the next call without a daemon restart.
A remote-code refusal still raises: it is a configuration to fix, not a mode.
"""

from __future__ import annotations

import dataclasses
from importlib.util import find_spec

import structlog

from trw_memory.embeddings import get_local_embedder
from trw_memory.embeddings._declared_space import snapshot_revision
from trw_memory.embeddings._hf_cache import CacheState, probe_model_cache
from trw_memory.embeddings._provider_cache import loaded_local_embedder
from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.embeddings.local import FETCH_COMMAND
from trw_memory.embeddings.provenance import provider_embedding_space
from trw_memory.exceptions import ModelNotCachedError
from trw_memory.models.config import MemoryConfig
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)


def resolve_embedder(config: MemoryConfig, *, surface: str) -> EmbeddingProvider | dict[str, object]:
    """The embedder, or the ``{"status": "unavailable", "reason": ..., ["fix"]}`` answer to return instead."""
    try:
        embedder = get_local_embedder(model_name=config.embedding_model, dim=config.embedding_dim)
    except ModelNotCachedError as exc:
        logger.warning("embedder_unavailable", surface=surface, reason="model_not_cached", detail=str(exc))
        return {"status": "unavailable", "reason": "model_not_cached", "fix": FETCH_COMMAND}
    if embedder is None:
        logger.warning("embedder_unavailable", surface=surface, reason="embedder_error")
        return {"status": "unavailable", "reason": "embedder_error"}
    return embedder


def embedder_status(config: MemoryConfig) -> dict[str, object]:
    """Contract C3 ``embedder`` block. Probes the cache and peeks at the provider cache; never loads a model."""

    loaded = loaded_local_embedder((config.embedding_model, config.embedding_dim))
    space = provider_embedding_space(loaded) if loaded is not None else None
    probe = probe_model_cache(config.embedding_model)
    reason: str | None = None
    if loaded is None and find_spec("sentence_transformers") is None:
        reason = "embedder_error"
    elif loaded is None and probe.state is not CacheState.COMPLETE:
        reason = "model_not_cached"
    block: dict[str, object] = {
        "available": reason is None,
        "model": config.embedding_model,
        "revision": snapshot_revision(probe.snapshot_path) or None,
        "space": dataclasses.asdict(space) if space is not None else None,
        "loaded": loaded is not None,
        "reason": reason,
    }
    if reason == "model_not_cached":
        block["fix"] = FETCH_COMMAND
    return block


def coverage_status(backend: StorageBackend, namespace: str, config: MemoryConfig) -> dict[str, object] | None:
    """Contract C3 ``coverage`` block: *namespace*'s rows by where their vector stands, or ``None`` without a census.

    ``active_space`` and ``other_space`` are ``None`` until the model is loaded,
    because a measured space is known only from the loaded weights.
    ``outside_active_space`` counts the stored vectors dense recall refuses (other
    space or no provenance), which ``memory_reembed`` re-encodes; rows with no
    vector, canaries among them, stay under ``no_vector``.
    """
    census = backend.vector_space_census(namespace=namespace)
    if census is None:
        return None
    loaded = loaded_local_embedder((config.embedding_model, config.embedding_dim))
    space = provider_embedding_space(loaded) if loaded is not None else None
    unknown = census.get(None, 0)
    known = sum(count for key, count in census.items() if key is not None)
    active = census.get(space, 0) if space is not None else None
    return {
        "active_space": active,
        "other_space": known - active if active is not None else None,
        "unknown_provenance": unknown,
        "outside_active_space": known - active + unknown if active is not None else None,
        "no_vector": max(backend.count(namespace=namespace) - known - unknown, 0),
    }
