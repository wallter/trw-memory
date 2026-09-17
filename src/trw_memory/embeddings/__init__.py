"""Embedding providers for trw-memory."""

import structlog

from trw_memory.embeddings._provider_cache import cached_local_embedder, reset_provider_cache
from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.embeddings.local import LocalEmbeddingProvider
from trw_memory.exceptions import LocalOnlyViolationError, RemoteCodeNotPermittedError

logger = structlog.get_logger(__name__)

__all__ = [
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "get_local_embedder",
    "reset_provider_cache",
]

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIM = 384


def get_local_embedder(
    *,
    model_name: str | None = None,
    dim: int | None = None,
) -> EmbeddingProvider | None:
    """Return an available local embedding provider, or ``None`` on failure.

    The provider is cached for the life of the process, keyed by model name,
    dimension and the security/offline/snapshot-source policy that decided what
    could be loaded (PRD-CORE-279 FR01-FR03). A long-lived server therefore
    loads the model once rather than once per request. Nothing about the
    refusal contract changes: a security refusal still raises out of here and
    is never cached, and a provider that could not load is not cached either,
    so a later call retries.
    """
    key = (model_name or _DEFAULT_MODEL, dim or _DEFAULT_DIM)

    def _build() -> EmbeddingProvider | None:
        try:
            provider = LocalEmbeddingProvider(model_name=key[0], dim=key[1])
            if provider.available():
                return provider
        except (LocalOnlyViolationError, RemoteCodeNotPermittedError):
            # PRD-SEC-014 NFR02: a fail-closed security refusal is reported, never
            # swallowed into an indistinguishable "no embedder" None here. The
            # keyword-only degradation belongs one layer up, in trw-mcp's wrapper.
            raise
        except Exception:
            logger.debug("embedder_init_failed", exc_info=True)
        return None

    return cached_local_embedder(key, _build)
