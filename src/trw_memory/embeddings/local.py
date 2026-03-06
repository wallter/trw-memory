"""Local sentence-transformers embedding provider.

Implements :class:`~trw_memory.embeddings.interface.EmbeddingProvider` using
the ``sentence-transformers`` library (optional ``[embeddings]`` extra).  When
the library is not installed the provider gracefully degrades — all methods
continue to work but :meth:`LocalEmbeddingProvider.available` returns
``False`` and embed calls return ``None``.

The model is lazy-loaded on first use and cached as an instance attribute.
Multiple :class:`LocalEmbeddingProvider` instances each maintain their own
cache, which keeps the class stateless at the module level and makes testing
straightforward (no global state to clean up).
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIM = 384


class LocalEmbeddingProvider:
    """Sentence-transformers embedding provider with lazy model loading.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
            ``"all-MiniLM-L6-v2"`` (384-dimensional, fast, good quality).
        dim: Expected output dimensionality.  Must match the chosen model.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        dim: int = _DEFAULT_DIM,
    ) -> None:
        self._model_name = model_name
        self._dim = dim
        self._model: object | None = None
        self._load_attempted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> object | None:
        """Load and cache the sentence-transformers model.

        Sets ``_load_attempted`` after the first attempt so subsequent calls
        skip the import overhead (both success and failure paths are cached).
        """
        if self._load_attempted:
            return self._model

        self._load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            self._model = SentenceTransformer(self._model_name)
            logger.debug(
                "embedding model loaded",
                model=self._model_name,
                dim=self._dim,
            )
        except ImportError:
            logger.debug(
                "sentence-transformers not installed — embeddings unavailable",
                hint="pip install trw-memory[embeddings]",
            )
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "failed to load embedding model",
                model=self._model_name,
            )

        return self._model

    # ------------------------------------------------------------------
    # EmbeddingProvider interface
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float] | None:
        """Generate a single embedding vector.

        Args:
            text: Text to embed.  Blank strings return ``None`` immediately.

        Returns:
            A list of floats of length :meth:`dim`, or ``None`` on failure.
        """
        if not text.strip():
            return None

        model = self._load_model()
        if model is None:
            return None

        try:
            vector = model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
            return [float(v) for v in vector]
        except (RuntimeError, ValueError, TypeError):
            logger.warning(
                "embedding generation failed",
                text_length=len(text),
            )
            return None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Generate embeddings for multiple texts in one model call.

        Blank strings within the list are skipped during encoding and receive
        ``None`` in the output.  Non-blank strings are batched together for
        efficiency.

        Args:
            texts: List of texts to embed.

        Returns:
            List of the same length as *texts*.  Each entry is a float vector
            or ``None``.
        """
        if not texts:
            return []

        model = self._load_model()
        if model is None:
            return [None] * len(texts)

        results: list[list[float] | None] = []
        try:
            non_blank = [t for t in texts if t.strip()]
            if not non_blank:
                return [None] * len(texts)
            vectors = model.encode(  # type: ignore[attr-defined]
                non_blank,
                normalize_embeddings=True,
                batch_size=32,
            )
            vec_idx = 0
            for text in texts:
                if not text.strip():
                    results.append(None)
                else:
                    results.append([float(v) for v in vectors[vec_idx]])
                    vec_idx += 1
        except (RuntimeError, ValueError, TypeError):
            logger.warning(
                "batch embedding failed",
                batch_size=len(texts),
            )
            return [None] * len(texts)

        return results

    def available(self) -> bool:
        """Return ``True`` if the model loaded successfully.

        Triggers a load attempt on first call; subsequent calls use the cache.
        """
        return self._load_model() is not None

    def dim(self) -> int:
        """Return the dimensionality of vectors produced by this provider."""
        return self._dim
