"""Warm the daemon's models in the background once it is serving (DAEMON-MODEL-WARMUP).

Belongs to :mod:`trw_memory.daemon._serve`, which starts it right after the loopback socket is bound.

The embedding model and the cross-encoder both load lazily, so the first recall after every daemon start paid
the torch and sentence-transformers import plus two model loads (about 6 s against about 1 s warm). Loading them
on a thread that starts after the bind keeps readiness unblocked; a recall that arrives first simply waits on the
same one-load-per-model locks the loaders already hold. Off when embeddings are off; a failed load only logs.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import structlog

from trw_memory.models.config import MemoryConfig
from trw_memory.retrieval.reranker import _get_model as load_reranker
from trw_memory.tools._embedder import resolve_embedder

__all__ = ["start_model_warmup"]

logger = structlog.get_logger(__name__)


def _warm(config: MemoryConfig) -> None:
    # Each load is independent: a missing embedder must not leave the reranker cold.
    loads: tuple[tuple[str, Callable[[], object]], ...] = (
        ("embedder", lambda: resolve_embedder(config, surface="warmup")),
        ("reranker", lambda: load_reranker(config.recall_rerank_model)),
    )
    for name, load in loads:
        try:
            load()
        except Exception:  # justified: best-effort; the first recall loads the model itself and says why it cannot
            logger.warning("daemon_model_warmup_failed", model=name, exc_info=True)


def start_model_warmup(config: MemoryConfig | None = None) -> threading.Thread | None:
    """Start loading the models on a daemon thread; ``None`` when embeddings are off."""
    resolved = config or MemoryConfig()
    if not resolved.embeddings_enabled:
        return None
    thread = threading.Thread(target=_warm, args=(resolved,), name="trw-model-warmup", daemon=True)
    thread.start()
    return thread
