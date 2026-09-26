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

import threading
import time
from typing import Any, NamedTuple

import structlog

from trw_memory._model_pin import model_revision
from trw_memory.embeddings.local import FETCH_COMMAND, INFERENCE_DEVICE
from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# ms-marco-MiniLM models accept up to 512 tokens (~2048 chars at 4 chars/token).
# The old 512-char limit wasted 75% of model capacity on long sessions.
_MAX_PASSAGE_CHARS = 2048


#: A failed load, and when: retried after ``_RETRY_AFTER_S`` so a daemon picks up a
#: model fetched after it started, without trying to load on every recall.
class _FailedLoad(NamedTuple):
    at: float


_RETRY_AFTER_S = 60.0
_LOADED_MODELS: dict[str, object] = {}
_LOAD_LOCK = threading.Lock()

# Lazy-import state for ``sentence_transformers.CrossEncoder``.
#
# ``sentence_transformers`` transitively imports ``torch`` (~2.5-5.6s), so
# importing it at module load time made *every* consumer of
# ``trw_memory.retrieval`` pay the torch tax — including the ``trw_mcp.server``
# boot path, which pushed MCP connect past clients' 30s timeout under
# contention (production feedback sub_psVs_nUWnLJGvOs3).  We defer the import to
# first use and cache the outcome so the import machinery runs at most once.
#
# ``_cross_encoder_available`` is ``None`` until the first probe, then a bool.
_cross_encoder_cls: Any = None
_cross_encoder_available: bool | None = None


def _import_cross_encoder() -> bool:
    """Attempt to import ``sentence_transformers.CrossEncoder`` lazily.

    On the first call this runs the (expensive) import and caches the resolved
    class — or ``None`` on :class:`ImportError` — plus an availability flag in
    module globals, so subsequent calls never re-enter the import machinery.

    Returns:
        ``True`` when the cross-encoder is importable, ``False`` otherwise.
    """
    global _cross_encoder_cls, _cross_encoder_available
    if _cross_encoder_available is None:
        try:
            from sentence_transformers import CrossEncoder

            _cross_encoder_cls = CrossEncoder
            _cross_encoder_available = True
        except ImportError:
            _cross_encoder_cls = None
            _cross_encoder_available = False
    return _cross_encoder_available


def __getattr__(name: str) -> object:
    """PEP 562 module attribute hook.

    Preserves the legacy ``reranker._CROSS_ENCODER_AVAILABLE`` module-level
    boolean (still consumed by tests and any external caller) without paying the
    eager import cost: it is resolved lazily on first access.
    """
    if name == "_CROSS_ENCODER_AVAILABLE":
        return _import_cross_encoder()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_model(model_name: str) -> object | None:
    """Lazy-load and cache a CrossEncoder model by name, from the local cache only.

    A runtime load never downloads (PLAN W40): an uncached model yields ``None``
    and callers keep fusion order, with a warning naming the fetch command. A
    failed load is retried once ``_RETRY_AFTER_S`` has passed, so a model fetched
    while the daemon runs is used without a restart.
    """
    if not _import_cross_encoder():
        return None
    # One load per model: the daemon's first concurrent recalls must not load it twice (W27).
    with _LOAD_LOCK:
        cached = _LOADED_MODELS.get(model_name)
        if isinstance(cached, _FailedLoad) and time.monotonic() - cached.at < _RETRY_AFTER_S:
            return None
        if cached is None or isinstance(cached, _FailedLoad):
            try:
                _LOADED_MODELS[model_name] = _cross_encoder_cls(
                    model_name,
                    max_length=512,
                    revision=model_revision(model_name),
                    local_files_only=True,
                    device=INFERENCE_DEVICE,
                )
                logger.debug("reranker_model_loaded", model=model_name)
            except Exception:  # trw-fail-silent-allow: logged at warning with the fetch command; recall keeps fusion order and the load is retried after _RETRY_AFTER_S
                logger.warning("reranker_model_load_failed", model=model_name, fix=FETCH_COMMAND)
                _LOADED_MODELS[model_name] = _FailedLoad(time.monotonic())
                return None
        return _LOADED_MODELS[model_name]


def _entry_text(entry: MemoryEntry) -> str:
    """Build a passage string from a MemoryEntry for cross-encoder input."""
    parts = [entry.content]
    if entry.detail:
        parts.append(entry.detail)
    if entry.tags:
        parts.append(" ".join(entry.tags))
    text = " ".join(parts)
    return text[:_MAX_PASSAGE_CHARS]


def cross_encode_scores(
    query: str,
    entries: list[MemoryEntry],
    *,
    model_name: str = _DEFAULT_MODEL,
) -> list[tuple[MemoryEntry, float]] | None:
    """Score every entry against *query* with the cross-encoder.

    Returns ``(entry, score)`` pairs sorted by score descending, on the model's
    native logit scale (ms-marco MiniLM: roughly -11 for unrelated text up to
    +10 for an exact answer). Returns ``None`` when the cross-encoder is
    unavailable or inference fails, so callers can keep their fusion order.
    """
    if not entries:
        return []
    model = _get_model(model_name)
    if model is None:
        logger.debug("cross_encode_scores_skipped", reason="model_unavailable", count=len(entries))
        return None
    pairs = [[query, _entry_text(e)] for e in entries]
    try:
        scores = model.predict(pairs)  # type: ignore[attr-defined]
        # A model that returns the wrong shape or non-numeric output counts as
        # "inference failed": callers must keep fusion order, never crash recall.
        scored = [(e, float(s)) for e, s in zip(entries, scores, strict=True)]
    except Exception as exc:  # trw-fail-silent-allow: documented degradation -- an uncached model, a load failure or malformed model output returns None so callers keep fusion order; the warning names the cause
        logger.warning("cross_encode_rerank_error", error=str(exc)[:120])
        return None
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


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
        scored = sorted(zip(entries, scores, strict=True), key=lambda x: float(x[1]), reverse=True)
    except Exception as exc:
        logger.warning("cross_encode_rerank_error", error=str(exc)[:120])
        return entries[:top_k] if top_k is not None else entries

    reranked = [e for e, _ in scored]

    logger.debug(
        "cross_encode_rerank_complete",
        query=query[:80],
        input_count=len(entries),
        returned=len(reranked) if top_k is None else min(len(reranked), top_k),
        model=model_name,
    )

    return reranked[:top_k] if top_k is not None else reranked
