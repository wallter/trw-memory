"""Embedding providers for trw-memory."""

import structlog

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.embeddings.local import LocalEmbeddingProvider
from trw_memory.exceptions import LocalOnlyViolationError

logger = structlog.get_logger(__name__)

__all__ = ["EmbeddingProvider", "LocalEmbeddingProvider", "get_local_embedder"]


def get_local_embedder(
    *,
    model_name: str | None = None,
    dim: int | None = None,
) -> EmbeddingProvider | None:
    """Return an available local embedding provider, or ``None`` on failure."""
    try:
        provider = LocalEmbeddingProvider(
            model_name=model_name or "all-MiniLM-L6-v2",
            dim=dim or 384,
        )
        if provider.available():
            return provider
    except LocalOnlyViolationError:
        raise
    except Exception:
        logger.debug("embedder_init_failed", exc_info=True)
    return None
