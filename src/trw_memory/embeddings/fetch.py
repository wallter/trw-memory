"""The one code path that downloads a model (PLAN W40: one network story).

Runtime loaders (the embedder, the reranker) are cache-only and raise or degrade
with ``trw-mcp models fetch`` as the fix. That verb and the installer call
:func:`fetch_models`, which loads each configured model once through the same
library the runtime uses, at the revision ``_model_pin`` names, so the files the
runtime asks for are exactly the files fetched.
"""

from __future__ import annotations

import structlog

from trw_memory._model_pin import model_revision
from trw_memory.models.config import MemoryConfig

__all__ = ["fetch_models"]

logger = structlog.get_logger(__name__)


def fetch_models(*, embedding_model: str | None = None, rerank_model: str | None = None) -> dict[str, str]:
    """Download the embedding and rerank models; ``{model: revision}`` fetched.

    Each defaults to the configured one (``MemoryConfig``). Raises ``ImportError``
    without sentence-transformers (``trw-memory[embeddings]``) and whatever the Hub
    client raises when a download fails: a fetch the caller asked for must not
    fail quietly.
    """
    from sentence_transformers import CrossEncoder, SentenceTransformer

    cfg = MemoryConfig()
    trust_remote_code = bool(cfg.embedding_trust_remote_code)
    fetched: dict[str, str] = {}
    for model, load in (
        (embedding_model or cfg.embedding_model, SentenceTransformer),
        (rerank_model or cfg.recall_rerank_model, CrossEncoder),
    ):
        revision = model_revision(model)
        logger.info("model_fetch", model=model, revision=revision, source="huggingface.co")
        load(model, revision=revision, trust_remote_code=trust_remote_code)
        fetched[model] = revision
    return fetched
