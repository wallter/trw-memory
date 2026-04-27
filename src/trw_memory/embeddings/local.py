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

import contextlib
import importlib.util
import sys
from collections.abc import Iterator

import structlog

from trw_memory.exceptions import LocalOnlyViolationError
from trw_memory.models.config import MemoryConfig

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIM = 384
_TORCHCODEC_MODULE_PREFIX = "torchcodec"
_MISSING = object()


def _torchcodec_installed() -> bool:
    """Return True when torchcodec is import-discoverable.

    Broken ``sys.modules`` sentinels can make ``find_spec`` raise ``ValueError``;
    treat that as installed so the masking path can repair the import attempt.
    """
    try:
        return importlib.util.find_spec(_TORCHCODEC_MODULE_PREFIX) is not None
    except ValueError:
        return True


def _torchcodec_decoders_broken() -> bool:
    """Return True when installed torchcodec cannot import its decoders.

    SentenceTransformers 5 imports optional audio/video helpers at package import
    time. Text embeddings do not need torchcodec, but a broken torchcodec wheel
    can raise ``RuntimeError`` during that optional import and prevent
    ``SentenceTransformer`` itself from importing.
    """
    if not _torchcodec_installed():
        return False
    try:
        from torchcodec import decoders as _decoders  # type: ignore[import-not-found, import-untyped, unused-ignore]

        del _decoders
        return False
    except ImportError:
        return False
    except Exception as exc:  # justified: optional dependency can raise RuntimeError/OSError at import time
        logger.debug(
            "torchcodec_decoders_unavailable_for_text_embeddings",
            error_type=type(exc).__name__,
        )
        return True


@contextlib.contextmanager
def _hide_broken_torchcodec_for_sentence_transformers() -> Iterator[None]:
    """Temporarily make a broken torchcodec look absent during ST import.

    SentenceTransformers catches ImportError/OSError for optional torchcodec, but
    not every torchcodec binary failure is surfaced as those types. Hiding only
    during import lets text-only embeddings work without uninstalling torchcodec
    for other application features.
    """
    if not _torchcodec_decoders_broken():
        yield
        return

    original: dict[str, object] = {
        name: sys.modules.get(name, _MISSING)
        for name in list(sys.modules)
        if name == _TORCHCODEC_MODULE_PREFIX or name.startswith(f"{_TORCHCODEC_MODULE_PREFIX}.")
    }
    for name in list(sys.modules):
        if name == _TORCHCODEC_MODULE_PREFIX or name.startswith(f"{_TORCHCODEC_MODULE_PREFIX}."):
            del sys.modules[name]
    sys.modules[_TORCHCODEC_MODULE_PREFIX] = None  # type: ignore[assignment]
    sys.modules[f"{_TORCHCODEC_MODULE_PREFIX}.decoders"] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == _TORCHCODEC_MODULE_PREFIX or name.startswith(f"{_TORCHCODEC_MODULE_PREFIX}."):
                del sys.modules[name]
        for name, value in original.items():
            if value is not _MISSING:
                sys.modules[name] = value  # type: ignore[assignment]


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
        self._last_load_error: str = ""

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
        config = MemoryConfig()
        try:
            with _hide_broken_torchcodec_for_sentence_transformers():
                from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, local_files_only=config.local_only)
            logger.debug(
                "embedding_model_loaded",
                model=self._model_name,
                dim=self._dim,
            )
        except ImportError:
            self._last_load_error = "sentence-transformers is not installed"
            logger.debug(
                "embedding_library_unavailable",
                hint="pip install trw-memory[embeddings]",
            )
        except OSError as exc:
            if config.local_only:
                raise LocalOnlyViolationError(
                    f"Model '{self._model_name}' not found in local cache. Download is blocked "
                    "(memory_local_only=True). Pre-download the model: "
                    f"python -m sentence_transformers download {self._model_name}"
                ) from exc
            self._last_load_error = f"sentence-transformers installed but model load failed: {exc}"
            logger.warning(
                "embedding_model_load_failed",
                model=self._model_name,
                exc_info=True,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_load_error = f"sentence-transformers installed but runtime dependency failed: {exc}"
            logger.warning(
                "embedding_model_load_failed",
                model=self._model_name,
                exc_info=True,
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
                "embedding_generation_failed",
                text_length=len(text),
                exc_info=True,
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
                "embedding_batch_failed",
                batch_size=len(texts),
                exc_info=True,
            )
            return [None] * len(texts)

        return results

    def available(self) -> bool:
        """Return ``True`` if the model loaded successfully.

        Triggers a load attempt on first call; subsequent calls use the cache.
        """
        return self._load_model() is not None

    def unavailable_reason(self) -> str:
        """Return the last model-load failure reason, if any."""
        return self._last_load_error

    def dim(self) -> int:
        """Return the dimensionality of vectors produced by this provider."""
        return self._dim
