"""Dense vector similarity search for trw-memory.

Computes cosine similarity between a query embedding and pre-computed
entry embeddings supplied by the caller.  The caller is responsible for
maintaining the ``stored_embeddings`` dict (typically populated by the
storage backend on write).

Requires an :class:`~trw_memory.embeddings.interface.EmbeddingProvider`
to be available.  When the provider is ``None`` or returns ``None`` for the
query the function returns an empty list, allowing the pipeline to degrade
gracefully to BM25-only mode.
"""

from __future__ import annotations

import math

import structlog

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.exceptions import DimensionMismatchError

logger = structlog.get_logger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector (must be the same length as *a*).

    Returns:
        Cosine similarity in the range ``[-1.0, 1.0]``.  Returns ``0.0``
        when either vector is the zero vector.
    """
    if len(a) != len(b):
        raise DimensionMismatchError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def dense_search(
    query: str,
    entry_ids: list[str],
    embedder: EmbeddingProvider | None = None,
    query_embedding: list[float] | None = None,
    stored_embeddings: dict[str, list[float]] | None = None,
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """Dense vector similarity search over pre-computed entry embeddings.

    The function first resolves the query embedding: if *query_embedding* is
    provided it is used directly; otherwise ``embedder.embed(query)`` is
    called.  Cosine similarity is then computed against every entry in
    *stored_embeddings* whose id appears in *entry_ids*.

    Graceful degradation:
    - Returns ``[]`` when *embedder* is ``None``.
    - Returns ``[]`` when *embedder* reports it is unavailable.
    - Returns ``[]`` when query embedding cannot be computed.
    - Silently skips entries that have no stored embedding.

    Args:
        query: The search query string (used only when *query_embedding* is
            ``None``).
        entry_ids: Candidate entry IDs to search over.  Results are restricted
            to IDs that appear in both this list and *stored_embeddings*.
        embedder: Optional embedding provider used to encode the query.
        query_embedding: Pre-computed query vector.  When supplied,
            *embedder* is not called.
        stored_embeddings: Mapping of ``entry_id`` → vector.  Entries missing
            from this mapping are skipped.
        top_k: Maximum number of results to return.

    Returns:
        List of ``(entry_id, score)`` pairs sorted by cosine similarity
        descending.  Returns an empty list on graceful-degradation paths.
    """
    # ------------------------------------------------------------------ guard
    if embedder is None and query_embedding is None:
        logger.debug("dense_search_skipped", reason="no_embedder")
        return []

    if embedder is not None and not embedder.available() and query_embedding is None:
        logger.debug("dense_search_skipped", reason="embedder_unavailable")
        return []

    if not entry_ids or not stored_embeddings:
        logger.debug("dense_search_skipped", reason="no_candidates")
        return []

    # --------------------------------------------------------- query embedding
    q_vec: list[float] | None = query_embedding
    if q_vec is None:
        # embedder is not None here (guarded above)
        assert embedder is not None  # noqa: S101 — mypy narrowing guard; embedder is not None here: the early-return guard above (line ~89) exits when embedder is None and query_embedding is also None
        try:
            q_vec = embedder.embed(query)
        except (RuntimeError, ValueError, TypeError) as exc:
            # Structural telemetry only — query text may carry secrets or
            # proprietary memory contents, so never log it (raw or preview).
            logger.warning(
                "dense_search_embed_failed",
                query_chars=len(query),
                candidates=len(entry_ids),
                error_class=type(exc).__name__,
            )
            return []

    if q_vec is None:
        logger.debug("dense_search_skipped", reason="null_query_embedding")
        return []

    # -------------------------------------------------- cosine similarity scan
    results: list[tuple[str, float]] = []
    for entry_id in entry_ids:
        stored = stored_embeddings.get(entry_id)
        if stored is None:
            continue
        try:
            score = cosine_similarity(q_vec, stored)
            results.append((entry_id, score))
        except (DimensionMismatchError, ZeroDivisionError):
            logger.debug("dense_search_entry_skipped", entry_id=entry_id)
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    top = results[:top_k]
    logger.debug(
        "dense_search_complete",
        query_chars=len(query),
        candidates=len(entry_ids),
        scored=len(results),
        returned=len(top),
    )
    return top
