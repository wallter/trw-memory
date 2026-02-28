"""EmbeddingProvider Protocol — defines the contract for all embedding backends.

Any class that implements :class:`EmbeddingProvider` can be used as a drop-in
replacement without subclassing.  Use :class:`LocalEmbeddingProvider` from
:mod:`trw_memory.embeddings.local` for the default sentence-transformers
implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Structural protocol for embedding providers.

    Implementations must supply all four methods below.  The protocol is
    ``@runtime_checkable`` so ``isinstance(obj, EmbeddingProvider)`` works
    for dependency injection checks.
    """

    def embed(self, text: str) -> list[float] | None:
        """Generate an embedding vector for a single text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats (length == :meth:`dim`), or ``None`` if the
            provider is unavailable or the text is blank.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Generate embedding vectors for multiple texts in one call.

        More efficient than repeated :meth:`embed` calls because providers
        can batch-encode internally.

        Args:
            texts: Texts to embed.

        Returns:
            A list of the same length as *texts*.  Each element is either a
            float vector or ``None`` for blank / failed entries.
        """
        ...

    def available(self) -> bool:
        """Return ``True`` if the provider is ready to produce embeddings.

        Callers use this for feature-detection without generating an actual
        embedding (e.g., deciding whether to write to the vector column).
        """
        ...

    def dim(self) -> int:
        """Return the dimensionality of vectors produced by this provider.

        All vectors returned by :meth:`embed` and :meth:`embed_batch` have
        exactly this many elements.
        """
        ...
