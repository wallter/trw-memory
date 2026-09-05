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
import os
import sys
from collections.abc import Iterator

import structlog

from trw_memory.embeddings._hf_cache import CacheProbe, CacheState, probe_model_cache
from trw_memory.exceptions import LocalOnlyViolationError, RemoteCodeNotPermittedError
from trw_memory.models.config import MemoryConfig

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIM = 384
_TORCHCODEC_MODULE_PREFIX = "torchcodec"
_MISSING = object()
_HF_HOST = "huggingface.co"

# PRD-SEC-014-FR02: the single named consent for executing repo-supplied code.
_REMOTE_CODE_FIELD = "embedding_trust_remote_code"

# The loader's own refusal when pre-load detection was inconclusive: transformers
# says "requires you to execute ... set the option `trust_remote_code=True`", so
# the fail-closed message is reached either way (RISK-004). The markers are the
# refusal's DIRECTIVE, not the bare flag name: a TypeError from a stale call
# signature also contains "trust_remote_code" and must not be reported as a
# security refusal.
_REMOTE_CODE_ERROR_MARKERS = ("trust_remote_code=true", "requires you to execute")

# PRD-QUAL-110-FR04: offline switches that block the huggingface.co model
# download. ``TRW_OFFLINE`` is the TRW master switch; ``HF_HUB_OFFLINE`` is the
# upstream huggingface_hub convention. Any truthy value forces
# ``local_files_only=True`` even when the ``local_only`` config field is False.
_OFFLINE_ENV_VARS = ("TRW_OFFLINE", "HF_HUB_OFFLINE")
_TRUTHY = ("1", "true", "yes", "on")


def _offline_download_blocked() -> bool:
    """Return True when an env offline switch blocks the model download (FR04)."""
    return any(os.environ.get(name, "").strip().lower() in _TRUTHY for name in _OFFLINE_ENV_VARS)


def _blocked_by(config: MemoryConfig, offline: bool) -> str:
    """Name what prevented the download in a ``local_files_only`` failure.

    Since PRD-SEC-014-FR01 a cache probe can force ``local_files_only`` on its
    own, so the message must be able to say so (RISK-001): reporting an offline
    switch that is not set would send the operator to unset nothing.
    """
    if config.local_only:
        return "memory_local_only=True"
    if offline:
        return "TRW_OFFLINE/HF_HUB_OFFLINE"
    return "the local cache reported a complete snapshot, so no download was attempted"


def _is_remote_code_error(exc: BaseException) -> bool:
    """Return True when a loader failure was a refusal to execute remote code."""
    if isinstance(exc, TypeError):
        # A bad call signature is a programming error, never a policy refusal.
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _REMOTE_CODE_ERROR_MARKERS)


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


#: Device name handed to sentence-transformers when the CUDA load fails.
_CPU_DEVICE = "cpu"
#: How much of the CUDA error text the fallback warning carries.
_CUDA_ERROR_DETAIL_CHARS = 160


def _is_cuda_error(exc: BaseException) -> bool:
    """True when a load-time RuntimeError comes from CUDA (OOM, driver, device)."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    return "cuda" in text or "out of memory" in text


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

    def _probe_cache(self) -> CacheProbe:
        """Probe the local HF cache once per instance, failing open (NFR01/NFR02).

        The probe is evaluated inside :meth:`_load_model`, which is latched by
        ``_load_attempted``, so it runs at most once per provider. A probe that
        raises for any reason logs one degradation line and yields ``UNKNOWN``;
        it never fails a load that would otherwise succeed.
        """
        try:
            return probe_model_cache(self._model_name)
        except Exception as exc:  # justified: the probe must never fail a viable load (NFR02)
            logger.warning(
                "embedding_cache_probe_degraded",
                model=self._model_name,
                error_type=type(exc).__name__,
            )
            return CacheProbe(CacheState.UNKNOWN)

    def _remote_code_error(self, cause: str) -> RemoteCodeNotPermittedError:
        """Build the fail-closed error naming the one field that permits it."""
        return RemoteCodeNotPermittedError(
            f"Model '{self._model_name}' requires executing code shipped by the model "
            f"repository ({cause}), and {_REMOTE_CODE_FIELD} is False. To permit it, set "
            f"{_REMOTE_CODE_FIELD}: true in .trw/config.yaml (or export "
            f"MEMORY_{_REMOTE_CODE_FIELD.upper()}=1) — only if you trust that repository, "
            f"because its code runs with this process's privileges."
        )

    def _load_model(self) -> object | None:
        """Load and cache the sentence-transformers model.

        Sets ``_load_attempted`` after the first attempt so subsequent calls
        skip the import overhead (both success and failure paths are cached).
        """
        if self._load_attempted:
            return self._model

        self._load_attempted = True
        config = MemoryConfig()
        # PRD-QUAL-110-FR04: an env offline switch forces local-files-only even
        # when the config field is False, so an air-gapped deployer can prove
        # zero huggingface.co egress without editing config.
        offline = _offline_download_blocked()
        switch_forced = bool(config.local_only) or offline
        # PRD-SEC-014-FR01: the cache decides first. A complete local snapshot
        # needs no Hub round-trip at all, so the config/env expression is now the
        # FALLBACK for a cache we could not confirm, not the primary resolution.
        probe = self._probe_cache()
        cache_first = probe.state is CacheState.COMPLETE and bool(probe.snapshot_path)
        local_files_only = switch_forced or probe.state is CacheState.COMPLETE
        # ``local_files_only=True`` is NOT sufficient on its own: transformers'
        # AutoProcessor rebuilds its hub kwargs with
        # ``inspect.signature(cached_file).parameters``, and ``cached_file``'s
        # signature is ``(path_or_repo_id, filename, **kwargs)`` — so every hub
        # kwarg, local_files_only included, is dropped and the processor/feature
        # -extractor probes still reach huggingface.co. That is the source of the
        # two unauthenticated-request warnings reported against a warm cache.
        # Handing sentence-transformers the resolved snapshot DIRECTORY takes the
        # local-directory branch instead, which cannot make a request at all.
        model_ref = probe.snapshot_path if cache_first else self._model_name
        # PRD-SEC-014-FR02: one typed field, and nothing else, decides this.
        trust_remote_code = bool(config.embedding_trust_remote_code)
        if probe.declares_remote_code and not trust_remote_code:
            raise self._remote_code_error("its cached snapshot ships Python modules")
        if not local_files_only:
            # A network-capable load may fetch the model from huggingface.co —
            # disclose the potential egress before it happens (FR04).
            logger.info(
                "embedding_model_download_disclosure",
                model=self._model_name,
                source=_HF_HOST,
                cache_state=probe.state.value,
                detail=(
                    "Embedding model may be downloaded from huggingface.co on "
                    "first use. Set TRW_OFFLINE=1 (or HF_HUB_OFFLINE=1) to block."
                ),
            )
        try:
            with _hide_broken_torchcodec_for_sentence_transformers():
                from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(
                    model_ref,
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                )
            except RuntimeError as exc:
                # A busy or full GPU (another process holding CUDA memory) makes
                # the default device selection fail at load time. The encoder is
                # small enough to run on CPU, so a CUDA failure retries there
                # instead of silently disabling embeddings for the session.
                if not _is_cuda_error(exc):
                    raise
                logger.warning(
                    "embedding_model_cuda_fallback_cpu",
                    model=self._model_name,
                    detail=str(exc).splitlines()[0][:_CUDA_ERROR_DETAIL_CHARS],
                )
                self._model = SentenceTransformer(
                    model_ref,
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                    device=_CPU_DEVICE,
                )
            logger.debug(
                "embedding_model_loaded",
                model=self._model_name,
                dim=self._dim,
                cache_state=probe.state.value,
            )
        except ImportError:
            self._last_load_error = "sentence-transformers is not installed"
            logger.debug(
                "embedding_library_unavailable",
                hint="pip install trw-memory[embeddings]",
            )
        except OSError as exc:
            if not trust_remote_code and _is_remote_code_error(exc):
                raise self._remote_code_error("the loader refused to load it without that consent") from exc
            if local_files_only:
                raise LocalOnlyViolationError(
                    f"Model '{self._model_name}' not found in local cache. Download is blocked "
                    f"({_blocked_by(config, offline)}). Pre-download the model: "
                    f"python -m sentence_transformers download {self._model_name}"
                ) from exc
            self._last_load_error = f"sentence-transformers installed but model load failed: {exc}"
            logger.warning(
                "embedding_model_load_failed",
                model=self._model_name,
                exc_info=True,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            if not trust_remote_code and _is_remote_code_error(exc):
                raise self._remote_code_error("the loader refused to load it without that consent") from exc
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
        non_blank = [t for t in texts if t.strip()]
        if not non_blank:
            return [None] * len(texts)

        # Retry with progressively smaller batch sizes on OOM/RuntimeError.
        # Long sessions (10K+ chars) can exhaust GPU memory at batch_size=32
        # when the GPU is shared with other processes (e.g. a serving LLM).
        _batch_size = 32
        _min_batch = 1
        vectors = None
        while _batch_size >= _min_batch:
            try:
                vectors = model.encode(  # type: ignore[attr-defined]
                    non_blank,
                    normalize_embeddings=True,
                    batch_size=_batch_size,
                )
                break
            except (RuntimeError, ValueError, TypeError):
                if _batch_size == _min_batch:
                    logger.warning(
                        "embedding_batch_failed",
                        batch_size=_batch_size,
                        text_count=len(non_blank),
                        exc_info=True,
                    )
                    return [None] * len(texts)
                _batch_size = max(_min_batch, _batch_size // 4)
                logger.debug(
                    "embedding_batch_retry",
                    new_batch_size=_batch_size,
                    text_count=len(non_blank),
                )

        vec_idx = 0
        for text in texts:
            if not text.strip():
                results.append(None)
            else:
                results.append([float(v) for v in vectors[vec_idx]])  # type: ignore[index]
                vec_idx += 1

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
