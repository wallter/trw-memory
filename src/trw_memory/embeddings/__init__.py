"""Embedding providers for trw-memory."""

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.embeddings.local import LocalEmbeddingProvider

__all__ = ["EmbeddingProvider", "LocalEmbeddingProvider"]
