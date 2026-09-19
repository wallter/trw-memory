"""Query-role instructions for asymmetric retrieval encoders.

Some embedding models are trained for asymmetric search: the QUERY carries a
short instruction and the stored DOCUMENT does not. Encoding a query without its
instruction (or a document with one) measurably lowers retrieval quality, so the
role of each text is part of how it must be encoded.

The table is keyed by model id. A model absent from it is symmetric: queries and
documents are encoded identically, which is the correct behaviour for
``all-MiniLM-L6-v2`` and the historical behaviour for everything else.

Interface: :func:`query_prefix` and :func:`embed_query`. The prefix applies to
query-role text only; documents are always encoded as given, so the vectors a
store holds do not depend on this table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trw_memory.embeddings.interface import EmbeddingProvider

__all__ = ["embed_query", "query_prefix"]

_BGE_EN_V15_QUERY = "Represent this sentence for searching relevant passages: "

#: Model id -> instruction prepended to query-role text. Keys are compared
#: case-insensitively against both the full id and its repo-local name, so
#: ``BAAI/bge-small-en-v1.5`` and ``bge-small-en-v1.5`` resolve alike.
_QUERY_PREFIXES: dict[str, str] = {
    "baai/bge-small-en-v1.5": _BGE_EN_V15_QUERY,
    "baai/bge-base-en-v1.5": _BGE_EN_V15_QUERY,
    "baai/bge-large-en-v1.5": _BGE_EN_V15_QUERY,
}


def query_prefix(model_name: str) -> str:
    """Return the query instruction for *model_name*, or ``""`` when symmetric."""
    key = model_name.strip().lower()
    if key in _QUERY_PREFIXES:
        return _QUERY_PREFIXES[key]
    for known, prefix in _QUERY_PREFIXES.items():
        if key == known.rsplit("/", 1)[-1]:
            return prefix
    return ""


def embed_query(embedder: EmbeddingProvider, text: str) -> list[float] | None:
    """Encode *text* in the query role when *embedder* distinguishes roles.

    ``embed_query`` is an opt-in extension of the ``EmbeddingProvider``
    protocol, like ``embedding_space``: a provider that does not implement it is
    symmetric, and its plain ``embed`` IS the query encoding.
    """
    role_aware = getattr(embedder, "embed_query", None)
    if callable(role_aware):
        vector: list[float] | None = role_aware(text)
        return vector
    return embedder.embed(text)
