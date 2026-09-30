"""EMBED-PROBE-FAST-PATH: a model the cache probe proves absent is refused without importing the model runtime.

A daemon with sentence-transformers installed but no model cached used to import sentence-transformers and torch
(~4 s) on its first embed, only for the local-files-only load to miss. The probe already knew. The refusal, the
degraded-mode answer and its telemetry are the same either way; only the import is skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import structlog

from trw_memory.embeddings import _hf_cache
from trw_memory.embeddings._hf_cache import CacheProbe, CacheState, rules_out_local_load

_SRC = Path(__file__).resolve().parents[1] / "src"
_ABSENT_MODEL = "trw-test/no-such-embedding-model"


def _needs_the_runtime_installed() -> None:
    """The fast path is for installs that HAVE the runtime. find_spec, not importorskip: importing it is the ~4 s cost."""
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers is not installed")


@pytest.fixture
def empty_hub_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A hub cache with nothing in it, and no cache folder of sentence-transformers' own."""
    cache = tmp_path / "hf"
    cache.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache))
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        monkeypatch.delenv(var, raising=False)
    # A daemon started with HF_HOME set reads that cache; in this process huggingface_hub froze another one.
    monkeypatch.setattr(_hf_cache, "_loader_cache_dir", lambda: str(cache / "hub"))
    return cache


_CHILD = """
import json, sys
from trw_memory.embeddings.local import LocalEmbeddingProvider
from trw_memory.exceptions import ModelNotCachedError
try:
    LocalEmbeddingProvider(model_name=sys.argv[1], dim=384).available()
    refusal = None
except ModelNotCachedError as exc:
    refusal = str(exc)
print(json.dumps({"refusal": refusal, "loaded": [m for m in ("sentence_transformers", "torch") if m in sys.modules]}))
"""


@pytest.mark.integration
def test_an_absent_model_is_refused_in_a_fresh_interpreter_without_importing_the_model_runtime(
    empty_hub_cache: Path, tmp_path: Path
) -> None:
    _needs_the_runtime_installed()
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")}
    env |= {"HF_HOME": str(empty_hub_cache), "PYTHONPATH": str(_SRC), "HOME": str(tmp_path)}
    env.pop("SENTENCE_TRANSFORMERS_HOME", None)

    run = subprocess.run(
        [sys.executable, "-c", _CHILD, _ABSENT_MODEL], env=env, capture_output=True, text=True, timeout=120, check=False
    )

    assert run.returncode == 0, run.stderr[-3000:]
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert report["refusal"] is not None and "trw-mcp models fetch" in report["refusal"], report
    assert report["loaded"] == [], "the probe proved the model absent; nothing should have imported the runtime"


class _Refusing:
    """A stand-in sentence_transformers whose loader misses the cache the way the real one does."""

    loads: list[str] = []

    class SentenceTransformer:
        def __init__(self, model_ref: str, **_kwargs: object) -> None:
            _Refusing.loads.append(model_ref)
            raise OSError(f"{model_ref} is not a local folder and is not cached")


def test_the_probe_and_the_loader_give_the_same_refusal(empty_hub_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _needs_the_runtime_installed()
    from trw_memory.embeddings import local
    from trw_memory.exceptions import ModelNotCachedError

    with pytest.raises(ModelNotCachedError) as fast:
        local.LocalEmbeddingProvider(model_name=_ABSENT_MODEL, dim=384).available()

    monkeypatch.setattr(local, "rules_out_local_load", lambda _probe: False)  # force the real load
    monkeypatch.setitem(sys.modules, "sentence_transformers", _Refusing)
    _Refusing.loads.clear()
    with pytest.raises(ModelNotCachedError) as slow:
        local.LocalEmbeddingProvider(model_name=_ABSENT_MODEL, dim=384).available()

    assert _Refusing.loads, "the forced path must really have attempted the load"
    assert str(fast.value) == str(slow.value)


def test_the_degraded_mode_answer_and_telemetry_are_unchanged(empty_hub_cache: Path) -> None:
    """What the daemon answers and logs for an absent model: the model_not_cached block, as before."""
    _needs_the_runtime_installed()
    from trw_memory.embeddings.local import FETCH_COMMAND
    from trw_memory.models.config import MemoryConfig
    from trw_memory.tools._embedder import resolve_embedder

    with structlog.testing.capture_logs() as logs:
        answer = resolve_embedder(MemoryConfig(embedding_model=_ABSENT_MODEL), surface="memory_similar")

    assert answer == {"status": "unavailable", "reason": "model_not_cached", "fix": FETCH_COMMAND}
    warnings = [log for log in logs if log.get("event") == "embedder_unavailable"]
    assert [(w["surface"], w["reason"]) for w in warnings] == [("memory_similar", "model_not_cached")]


@pytest.mark.parametrize(
    ("state", "env", "version", "installed", "fast"),
    [
        pytest.param(CacheState.ABSENT, None, "6.1.0", True, True, id="absent-installed-v6"),
        pytest.param(CacheState.ABSENT, None, "3.0.0", True, True, id="absent-installed-v3"),
        pytest.param(CacheState.INCOMPLETE, None, "6.1.0", True, False, id="incomplete-keeps-the-load"),
        pytest.param(CacheState.UNKNOWN, None, "6.1.0", True, False, id="unknown-keeps-the-load"),
        pytest.param(CacheState.COMPLETE, None, "6.1.0", True, False, id="complete-loads"),
        pytest.param(CacheState.ABSENT, "/elsewhere", "6.1.0", True, False, id="own-cache-folder-set"),
        pytest.param(CacheState.ABSENT, None, "2.7.0", True, False, id="pre-v3-own-cache-folder"),
        pytest.param(CacheState.ABSENT, None, "not-a-version", True, False, id="unparseable-version"),
        pytest.param(CacheState.ABSENT, None, "6.1.0", False, False, id="not-installed-reports-itself"),
    ],
)
def test_only_a_proven_miss_skips_the_load(
    monkeypatch: pytest.MonkeyPatch, state: CacheState, env: str | None, version: str, installed: bool, fast: bool
) -> None:
    if env is None:
        monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising=False)
    else:
        monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", env)
    monkeypatch.setattr(_hf_cache.importlib.util, "find_spec", lambda _name: object() if installed else None)
    monkeypatch.setattr(_hf_cache.importlib.metadata, "version", lambda _name: version)
    monkeypatch.setattr(_hf_cache, "_resolve_cache_dir", lambda: "/cache/hub")  # the probe and the loader agree
    monkeypatch.setattr(_hf_cache, "_loader_cache_dir", lambda: "/cache/hub")

    assert rules_out_local_load(CacheProbe(state)) is fast


# -- the recall re-ranker takes the same fast path ---------------------------------------------------------------------

_RERANK_CHILD = """
import json, sys
from trw_memory.retrieval import reranker
missing = reranker._get_model(sys.argv[1]) is None
print(json.dumps({"missing": missing, "loaded": [m for m in ("sentence_transformers", "torch") if m in sys.modules]}))
"""


@pytest.mark.integration
def test_an_absent_reranker_model_is_skipped_in_a_fresh_interpreter_without_importing_the_runtime(
    empty_hub_cache: Path, tmp_path: Path
) -> None:
    _needs_the_runtime_installed()
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")}
    env |= {"HF_HOME": str(empty_hub_cache), "PYTHONPATH": str(_SRC), "HOME": str(tmp_path)}
    env.pop("SENTENCE_TRANSFORMERS_HOME", None)

    run = subprocess.run(
        [sys.executable, "-c", _RERANK_CHILD, "cross-encoder/no-such-reranker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert run.returncode == 0, run.stderr[-3000:]
    assert json.loads(run.stdout.strip().splitlines()[-1]) == {"missing": True, "loaded": []}


def test_an_absent_reranker_model_warns_as_before_and_is_probed_once_per_retry_window(
    empty_hub_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _needs_the_runtime_installed()
    from trw_memory.embeddings.local import FETCH_COMMAND
    from trw_memory.retrieval import reranker

    probes: list[str] = []
    real_probe = reranker.probe_model_cache
    monkeypatch.setattr(reranker, "probe_model_cache", lambda name: probes.append(name) or real_probe(name))
    monkeypatch.setattr(reranker, "_LOADED_MODELS", {})
    monkeypatch.setattr(reranker, "_import_cross_encoder", lambda: pytest.fail("the runtime import ran"))

    with structlog.testing.capture_logs() as logs:
        first, second = reranker._get_model("cross-encoder/x"), reranker._get_model("cross-encoder/x")

    assert (first, second) == (None, None)
    assert probes == ["cross-encoder/x"], "a recent failed load is not re-probed"
    warnings = [log for log in logs if log.get("event") == "reranker_model_load_failed"]
    assert [(w["model"], w["fix"]) for w in warnings] == [("cross-encoder/x", FETCH_COMMAND)]


def test_a_loaded_reranker_model_is_never_re_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.retrieval import reranker

    loaded = object()
    monkeypatch.setattr(reranker, "_LOADED_MODELS", {"cross-encoder/y": loaded})
    monkeypatch.setattr(reranker, "_import_cross_encoder", lambda: True)
    monkeypatch.setattr(reranker, "probe_model_cache", lambda _name: pytest.fail("probed a loaded model"))

    assert reranker._get_model("cross-encoder/y") is loaded


# -- codex r1 known issues (EMBED-PROBE-FAST-PATH) ---------------------------------------------------------------------


def test_cache_variables_are_expanded_the_way_huggingface_hub_expands_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("W4_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_HOME", "$W4_CACHE_ROOT/hf")
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(var, raising=False)

    assert _hf_cache._resolve_cache_dir() == str(tmp_path / "hf" / "hub")


@pytest.mark.parametrize(("loader_dir", "fast"), [("same", True), ("elsewhere", False), (None, False)])
def test_absent_counts_only_for_the_cache_the_loader_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader_dir: str | None, fast: bool
) -> None:
    """The probe honours variables set after huggingface_hub froze its cache path; the loader does not."""
    probed = tmp_path / "probed-hub"
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising=False)
    monkeypatch.setattr(_hf_cache.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_hf_cache.importlib.metadata, "version", lambda _name: "6.1.0")
    monkeypatch.setattr(_hf_cache, "_resolve_cache_dir", lambda: str(probed))
    loader = {"same": str(probed), "elsewhere": str(tmp_path / "frozen-hub"), None: None}[loader_dir]
    monkeypatch.setattr(_hf_cache, "_loader_cache_dir", lambda: loader)

    assert rules_out_local_load(CacheProbe(CacheState.ABSENT)) is fast


def test_a_reranker_probe_that_raises_falls_through_to_the_protected_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.retrieval import reranker

    attempts: list[str] = []

    class _Missing:
        def __init__(self, name: str, **_kwargs: object) -> None:
            attempts.append(name)
            raise OSError("not in the local cache")

    def _unreadable(_name: str) -> CacheProbe:
        raise RecursionError("a pathologically nested config.json")

    monkeypatch.setattr(reranker, "probe_model_cache", _unreadable)
    monkeypatch.setattr(reranker, "_import_cross_encoder", lambda: True)
    monkeypatch.setattr(reranker, "_cross_encoder_cls", _Missing)
    monkeypatch.setattr(reranker, "_LOADED_MODELS", {})

    with structlog.testing.capture_logs() as logs:
        model = reranker._get_model("cross-encoder/z")

    assert model is None
    assert attempts == ["cross-encoder/z"], "the protected loader decides when the probe cannot"
    assert [log["event"] for log in logs if log["log_level"] == "warning"] == [
        "reranker_cache_probe_degraded",
        "reranker_model_load_failed",
    ]
