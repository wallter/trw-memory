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
import threading
from collections.abc import Iterator

import structlog

from trw_memory._model_pin import DEFAULT_EMBEDDING_MODEL, model_revision
from trw_memory.embeddings._declared_space import declared_embedding_space, snapshot_revision
from trw_memory.embeddings._hf_cache import CacheProbe, CacheState, probe_model_cache
from trw_memory.embeddings._loaded_state import dependency_versions, loaded_state_digest
from trw_memory.embeddings._query_prompts import query_prefix
from trw_memory.embeddings._runtime_identity import capture_runtime_identity, runtime_identity_matches
from trw_memory.embeddings._similarity_calibration import register_space_model
from trw_memory.embeddings.provenance import EmbeddingSpace
from trw_memory.exceptions import ModelNotCachedError, RemoteCodeNotPermittedError
from trw_memory.models.config import MemoryConfig

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = DEFAULT_EMBEDDING_MODEL
_DEFAULT_DIM = 384
_TORCHCODEC_MODULE_PREFIX = "torchcodec"
_MISSING = object()

# PRD-SEC-014-FR02: the single named consent for executing repo-supplied code.
_REMOTE_CODE_FIELD = "embedding_trust_remote_code"

# The loader's own refusal when pre-load detection was inconclusive: transformers
# says "requires you to execute ... set the option `trust_remote_code=True`", so
# the fail-closed message is reached either way (RISK-004). The markers are the
# refusal's DIRECTIVE, not the bare flag name: a TypeError from a stale call
# signature also contains "trust_remote_code" and must not be reported as a
# security refusal.
_REMOTE_CODE_ERROR_MARKERS = ("trust_remote_code=true", "requires you to execute")

#: What a missing model's error names as the fix (runtime loads never download, PLAN W40).
#: Both routes: an SDK-only install has no ``trw-mcp`` command.
FETCH_COMMAND = "trw-mcp models fetch (SDK: trw_memory.embeddings.fetch_models())"

#: Every input is cut to this many characters before it reaches the model, so no caller (store,
#: correction, similar, reembed, consolidation) can hand the shared encode lock an unbounded text.
#: The model truncates to its token window anyway (512 tokens for the default), so the cut changes
#: no vector of ordinary text (rc9 sweep round 3).
MAX_EMBED_INPUT_CHARS = 8_000


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

#: The device every model in this process runs on; ``None`` lets torch choose.
#: macOS runs on CPU: two threads encoding on the Metal (MPS) device at once
#: aborted the daemon (``IOGPUMetalCommandBuffer setCurrentCommandEncoder``
#: assertion, 2026-09-25), and a process abort takes every client's memory down.
#: A lock would have to cover every torch call that touches the device, now and
#: later, and a gap is another abort. Measured on an M5 Pro: a query embed is
#: faster on CPU (5-8 vs 10-13 ms); a 50-pair re-rank costs ~20 ms more.
INFERENCE_DEVICE: str | None = _CPU_DEVICE if sys.platform == "darwin" else None
#: How much of the CUDA error text the fallback warning carries.
_CUDA_ERROR_DETAIL_CHARS = 160


#: OSErrors that name the machine, not the cache: never reported as a cache miss.
_NOT_A_CACHE_MISS = (PermissionError, IsADirectoryError, NotADirectoryError, InterruptedError)


def _is_cache_miss(exc: OSError, state: CacheState) -> bool:
    """Whether a local-files-only load failed because the model is not cached.

    A ``COMPLETE`` probe saw every declared file on disk, so a load error there is
    something else. Otherwise the loader's own miss (a plain ``OSError`` from
    transformers, or ``FileNotFoundError``) is a miss, unless the error names the
    machine: permissions, a path of the wrong kind, an interrupted read.
    """
    return state is not CacheState.COMPLETE and not isinstance(exc, _NOT_A_CACHE_MISS)


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
            ``"BAAI/bge-small-en-v1.5"`` (384-dimensional, 33M parameters).
        dim: Expected output dimensionality.  Must match the chosen model.

    Texts have a role. :meth:`embed` and :meth:`embed_batch` encode DOCUMENTS
    verbatim (the historical contract every external caller relies on);
    :meth:`embed_query` encodes a search QUERY, prepending the model's query
    instruction when it has one (see ``_query_prompts``). Symmetric models such
    as ``all-MiniLM-L6-v2`` encode both roles identically.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        dim: int = _DEFAULT_DIM,
    ) -> None:
        self._model_name = model_name
        self._dim = dim
        self._query_prefix = query_prefix(model_name)
        # PRD-CORE-279 NFR01: one provider is now shared by every worker
        # thread, and ``SentenceTransformer.encode`` is not a read-only call --
        # it moves the module to a device, flips it to eval, and mutates the
        # tokenizer's truncation/padding settings before encoding. Inference is
        # therefore serialised per provider. The lock is NOT held during the
        # model load; ``_provider_cache`` owns that serialisation.
        self._encode_lock = threading.Lock()
        self._model: object | None = None
        self._load_attempted: bool = False
        self._last_load_error: str = ""
        self._identity_model: object | None = None
        self._embedding_space: EmbeddingSpace | None = None
        self._identity_guard: object | None = None
        self._declared_space: EmbeddingSpace | None = None

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
        # PLAN W40: a runtime load never downloads. A complete snapshot is handed
        # over as its DIRECTORY: ``local_files_only=True`` alone is not enough,
        # because transformers' AutoProcessor rebuilds its hub kwargs from
        # ``inspect.signature(cached_file)`` and drops it, so the processor probes
        # would still reach huggingface.co. The local-directory branch cannot.
        probe = self._probe_cache()
        cache_first = probe.state is CacheState.COMPLETE and bool(probe.snapshot_path)
        model_ref = probe.snapshot_path if cache_first else self._model_name
        # PRD-SEC-014-FR02: one typed field, and nothing else, decides this.
        trust_remote_code = bool(config.embedding_trust_remote_code)
        if probe.declares_remote_code and not trust_remote_code:
            raise self._remote_code_error("its cached snapshot ships Python modules")
        try:
            with _hide_broken_torchcodec_for_sentence_transformers():
                from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(
                    model_ref,
                    revision=model_revision(self._model_name),
                    local_files_only=True,
                    trust_remote_code=trust_remote_code,
                    device=INFERENCE_DEVICE,
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
                    revision=model_revision(self._model_name),
                    local_files_only=True,
                    trust_remote_code=trust_remote_code,
                    device=_CPU_DEVICE,
                )
            # Identify what was loaded, not files that happened to be nearby.
            # No tree scans or exact dependency-version admission list.
            captured = None
            loaded_identity = None
            try:
                versions = dependency_versions()
                captured = (
                    capture_runtime_identity(self._model, self._dim, versions)
                    if versions and not trust_remote_code
                    else None
                )
                loaded_identity = loaded_state_digest(self._model) if captured is not None else None
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                logger.debug("embedding_identity_unavailable", error_type=type(exc).__name__)
            if captured is not None and loaded_identity is not None:
                manifest_digest, guard = captured
                self._identity_model = self._model
                self._identity_guard = guard
                self._embedding_space = EmbeddingSpace(
                    artifact_sha256=loaded_identity,
                    encoding=f"trw-loaded-encoder-v2:{manifest_digest}",
                    dimensions=self._dim,
                    model_id=self._model_name,
                )
            if self._embedding_space is None:
                # No measured identity (accelerator-resident or non-BERT encoder):
                # record the declared one so vectors are never written unqualified.
                # An uninspectable cache stays revision-less.
                revision = probe.snapshot_path if cache_first else ""
                self._declared_space = declared_embedding_space(
                    self._model_name, snapshot_revision(revision), self._dim
                )
            loaded_space = self._embedding_space or self._declared_space
            if loaded_space is not None:
                register_space_model(loaded_space, self._model_name)
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
            if not _is_cache_miss(exc, probe.state):
                # A permission, disk or corrupt-file error on a model that IS cached:
                # reporting it as "not in the local cache" sent operators to re-fetch
                # a model they already had (W07c). Surface it as itself.
                self._last_load_error = f"model load failed: {type(exc).__name__}: {exc}"
                logger.warning("embedding_model_load_failed", model=self._model_name, exc_info=True)
                return self._model
            raise ModelNotCachedError(
                f"Model '{self._model_name}' is not in the local cache, and runtime loads never "
                f"download. Fetch it: {FETCH_COMMAND}"
            ) from exc
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

    def embedding_space(self) -> EmbeddingSpace | None:
        """Return captured identity without model loads, file reads or inference.

        Only the exact loaded object and unchanged supported encoding settings
        retain the descriptor. Arbitrary in-place weight/tokenizer mutation is outside this immutable
        provider-lifetime contract; producers must not mutate loaded encoders.

        When the loaded state could not be measured at all, the declared space
        (model id + snapshot revision, ``_declared_space``) is returned instead.
        A measured identity that is later invalidated never falls back to it.
        """
        if self._embedding_space is None:
            return self._declared_space if self._model is not None else None
        if self._model is not self._identity_model:
            return None
        if not runtime_identity_matches(self._model, self._dim, self._identity_guard):
            return None
        return self._embedding_space

    def embed(self, text: str) -> list[float] | None:
        """Generate a single embedding vector.

        Args:
            text: Text to embed.  Blank strings return ``None`` immediately.

        Returns:
            A list of floats of length :meth:`dim`, or ``None`` on failure.
        """
        if not text.strip():
            return None
        text = text[:MAX_EMBED_INPUT_CHARS]

        model = self._load_model()
        if model is None:
            return None

        try:
            with self._encode_lock:
                vector = model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
            return [float(v) for v in vector]
        except (RuntimeError, ValueError, TypeError):
            logger.warning(
                "embedding_generation_failed",
                text_length=len(text),
                exc_info=True,
            )
            return None

    def embed_query(self, text: str) -> list[float] | None:
        """Embed *text* as a search query, with the model's query instruction."""
        if not text.strip():
            return None
        return self.embed(self._query_prefix + text)

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
        non_blank = [t[:MAX_EMBED_INPUT_CHARS] for t in texts if t.strip()]
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
                with self._encode_lock:
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

    @property
    def model_name(self) -> str:
        """Model id this provider encodes with (keys model-aware similarity thresholds)."""
        return self._model_name
