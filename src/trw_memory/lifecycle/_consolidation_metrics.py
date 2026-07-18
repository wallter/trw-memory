"""Similarity metrics used by lifecycle consolidation previews."""

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.dense import cosine_similarity


def mean_pairwise_similarity(cluster: list[MemoryEntry], embedder: EmbeddingProvider) -> float:
    """Compute mean pairwise cosine similarity, or zero without a valid pair."""
    if len(cluster) < 2:
        return 0.0
    vectors = embedder.embed_batch([entry.content + " " + entry.detail for entry in cluster])
    valid = [vector for vector in vectors if vector is not None]
    pairs = [cosine_similarity(valid[i], valid[j]) for i in range(len(valid)) for j in range(i + 1, len(valid))]
    return sum(pairs) / len(pairs) if pairs else 0.0
