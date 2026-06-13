"""Cross-encoder re-ranking for trw-memory (optional enhancement).

Provides a post-fusion re-ranking stage using a cross-encoder model
(``cross-encoder/ms-marco-MiniLM-L-6-v2`` by default) to score (query, passage)
pairs jointly.  Unlike bi-encoders that embed query and document independently,
cross-encoders attend over both at once, producing higher-quality relevance
scores at the cost of O(n) inference calls per query.

Usage
-----
Cross-encoding is applied AFTER RRF/CombMAX fusion:

1. Fused candidates are scored by the cross-encoder (query, entry_text).
2. Top-K candidates by cross-encoder score are returned.

This module is an **optional** enhancement — it requires ``sentence-transformers``
and a cached model.  When the dep or model is absent the function falls back
to returning the input list unchanged (graceful degradation).

Performance notes
-----------------
- Model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (66M params, ~80MB).
- Batch inference: all candidates scored in one forward pass.
- Expected latency on CPU: ~20-80ms for 25 candidates (128-token passages).
- Expected latency on GPU: <5ms.
- The model is lazily loaded on first call and cached as a module-level singleton
  so repeated recalls don't pay the load cost.
"""

from __future__ import annotations

import structlog

from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MAX_PASSAGE_CHARS = 512
_LOADED_MODELS: dict[str, object] = {}

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder

    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


def _get_model(model_name: str) -> object | None:
    """Lazy-load and cache a CrossEncoder model by name."""
    if not _CROSS_ENCODER_AVAILABLE:
        return None
    if model_name not in _LOADED_MODELS:
        try:
            _LOADED_MODELS[model_name] = _CrossEncoder(model_name, max_length=512)
            logger.debug("reranker_model_loaded", model=model_name)
        except Exception:
            logger.warning("reranker_model_load_failed", model=model_name)
            _LOADED_MODELS[model_name] = None
    return _LOADED_MODELS.get(model_name)


def _entry_text(entry: MemoryEntry) -> str:
    """Build a passage string from a MemoryEntry for cross-encoder input."""
    parts = [entry.content]
    if entry.detail:
        parts.append(entry.detail)
    if entry.tags:
        parts.append(" ".join(entry.tags))
    text = " ".join(parts)
    return text[:_MAX_PASSAGE_CHARS]


def cross_encode_rerank(
    query: str,
    entries: list[MemoryEntry],
    *,
    model_name: str = _DEFAULT_MODEL,
    top_k: int | None = None,
) -> list[MemoryEntry]:
    """Re-rank *entries* using cross-encoder (query, passage) scoring.

    Calls the cross-encoder on every entry in *entries* as a single batched
    forward pass and re-sorts by the predicted relevance score.  When the
    cross-encoder is unavailable (import error or model load failure) the
    input order is preserved exactly — the caller's fusion ranking remains
    intact.

    Args:
        query: The search query.
        entries: Fusion-ordered candidates to re-rank.  Typically the output
            of :func:`~trw_memory.retrieval.pipeline.hybrid_search` before the
            ``top_k`` slice.
        model_name: HuggingFace model id for the cross-encoder.  The default
            ``"cross-encoder/ms-marco-MiniLM-L-6-v2"`` is a 66M-param passage
            re-ranker trained on MS MARCO that transfers well to general
            short-text retrieval.
        top_k: When set, return only the top-K entries after re-ranking.

    Returns:
        *entries* re-ordered by cross-encoder relevance score descending, or
        the original order when the cross-encoder is unavailable.
    """
    if not entries:
        return entries

    model = _get_model(model_name)
    if model is None:
        logger.debug("cross_encode_rerank_skipped", reason="model_unavailable", count=len(entries))
        return entries[:top_k] if top_k is not None else entries

    passages = [_entry_text(e) for e in entries]
    pairs = [[query, p] for p in passages]

    try:
        scores = model.predict(pairs)  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("cross_encode_rerank_error", error=str(exc)[:120])
        return entries[:top_k] if top_k is not None else entries

    scored = sorted(zip(entries, scores), key=lambda x: float(x[1]), reverse=True)
    reranked = [e for e, _ in scored]

    logger.debug(
        "cross_encode_rerank_complete",
        query=query[:80],
        input_count=len(entries),
        returned=len(reranked) if top_k is None else min(len(reranked), top_k),
        model=model_name,
    )

    return reranked[:top_k] if top_k is not None else reranked
