"""PRD-SEC-014-FR06: a network-blocked regression on the REAL loader path.

``test_local_embedding_offline.py`` and ``test_embeddings.py`` only capture the
constructor arguments of a fake ``SentenceTransformer``; neither can observe a
Hub call, which is exactly why the reported warm-cache egress survived them.

Here every outbound connection attempt is intercepted at the socket layer and
counted, ``LocalEmbeddingProvider._load_model`` is the genuine implementation
(never patched). Since PLAN W40 there is no offline switch to set: a warm
cache loads from its snapshot directory, and a cold one raises
``ModelNotCachedError`` naming the fetch command, both with zero calls.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.embeddings import local as local_mod
from trw_memory.embeddings._hf_cache import CacheState, probe_model_cache
from trw_memory.exceptions import ModelNotCachedError

from ._test_hf_cache_support import (
    TINY_MODEL_DIM,
    NetworkSeam,
    build_loadable_model_cache,
    build_model_cache,
    install_fake_sentence_transformers,
    simulated_hub_request,
    use_fixture_cache,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def seam(monkeypatch: pytest.MonkeyPatch) -> Iterator[NetworkSeam]:
    installed = NetworkSeam()
    installed.install(monkeypatch)
    yield installed


def test_warm_cache_makes_zero_network_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    seam: NetworkSeam,
) -> None:
    """FR06: complete snapshot + no offline switch -> zero seam calls, no error."""
    use_fixture_cache(monkeypatch, tmp_path)
    build_model_cache(tmp_path)
    captured = install_fake_sentence_transformers(monkeypatch)

    # No offline switch exists any more; prove none is set anyway.
    assert os.environ.get("HF_HUB_OFFLINE") is None

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    assert provider.available() is True

    assert seam.calls == 0
    assert captured["local_files_only"] is True


def test_a_cold_cache_raises_the_fetch_command_with_zero_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    seam: NetworkSeam,
) -> None:
    """W40: no model on disk is an error naming ``trw-mcp models fetch``, never a download."""
    use_fixture_cache(monkeypatch, tmp_path)
    (tmp_path / "hub").mkdir()
    install_fake_sentence_transformers(monkeypatch, error=OSError("not in the local cache"))

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    with pytest.raises(ModelNotCachedError, match="trw-mcp models fetch"):
        provider.available()

    assert seam.calls == 0
    # Non-vacuity: the seam does count a dial when one happens.
    with pytest.raises(AssertionError, match="network seam invoked"):
        simulated_hub_request()
    assert seam.calls == 1


def test_real_sentence_transformers_warm_load_makes_zero_network_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    seam: NetworkSeam,
) -> None:
    """FR06: the invariant against the genuinely installed loader, no fakes.

    The fake-``SentenceTransformer`` cases above capture the arguments the
    resolution produced; they cannot see what the real stack DOES with them —
    and the real stack ignored one. transformers' ``AutoProcessor`` rebuilds its
    hub kwargs from ``inspect.signature(cached_file).parameters``
    (``path_or_repo_id``, ``filename``, ``**kwargs``), so ``local_files_only``
    is discarded and the processor probes reach huggingface.co regardless. That
    defect is invisible to every fake, which is why this case builds a real,
    tiny model into a fixture cache and loads it through the genuine loader with
    every socket refused.

    Offline and self-contained: the fixture model is constructed from
    transformers primitives, so this never depends on what the running machine
    happens to have downloaded and never needs the network.
    """
    pytest.importorskip("sentence_transformers")
    use_fixture_cache(monkeypatch, tmp_path)
    build_loadable_model_cache(tmp_path, repo_id=local_mod._DEFAULT_MODEL)

    assert probe_model_cache(local_mod._DEFAULT_MODEL).state is CacheState.COMPLETE

    provider = local_mod.LocalEmbeddingProvider(dim=TINY_MODEL_DIM)
    assert provider.available() is True
    vector = provider.embed("warm cache should never phone home")

    assert vector is not None
    assert len(vector) == TINY_MODEL_DIM
    assert seam.calls == 0


async def test_a_runtime_recall_with_the_cache_present_makes_zero_network_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    seam: NetworkSeam,
) -> None:
    """PLAN W40 acceptance: store and recall through the genuine stack, every socket refused.

    The embedder is cached; the re-ranker deliberately is not. Before W40 an
    uncached re-ranker was loaded network-capable on the first recall; now it is
    skipped and recall keeps fusion order.
    """
    pytest.importorskip("sentence_transformers")
    from trw_memory.client import MemoryClient
    from trw_memory.embeddings import reset_provider_cache
    from trw_memory.retrieval import reranker

    use_fixture_cache(monkeypatch, tmp_path)
    build_loadable_model_cache(tmp_path, repo_id=local_mod._DEFAULT_MODEL)
    monkeypatch.setenv("MEMORY_EMBEDDING_DIM", str(TINY_MODEL_DIM))
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "store"))
    monkeypatch.setattr(reranker, "_LOADED_MODELS", {})
    reset_provider_cache()

    client = MemoryClient(namespace="default", mode="local")
    try:
        await client.store("the deploy runbook lives in ops/deploy.md", importance=0.8)
        results = await client.recall("deploy runbook", limit=5)
    finally:
        await client.close()
        reset_provider_cache()

    assert [row["content"] for row in results] == ["the deploy runbook lives in ops/deploy.md"]
    # Tried the cache, skipped, and marked for a later retry.
    assert list(reranker._LOADED_MODELS) == ["cross-encoder/ms-marco-MiniLM-L-6-v2"]
    assert isinstance(reranker._LOADED_MODELS["cross-encoder/ms-marco-MiniLM-L-6-v2"], reranker._FailedLoad)
    assert seam.calls == 0


def test_transformers_still_discards_local_files_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the upstream premise the snapshot-directory workaround rests on.

    ``AutoProcessor.from_pretrained`` keeps only the kwargs named in
    ``inspect.signature(cached_file).parameters``, and ``cached_file`` takes
    ``(path_or_repo_id, filename, **kwargs)`` — so ``local_files_only`` never
    reaches the download path. If this assertion ever fails, upstream may have
    fixed the propagation: re-evaluate whether FR01 still needs to pass the
    resolved snapshot directory instead of the repo id.
    """
    hub = pytest.importorskip("transformers.utils.hub")
    params = inspect.signature(hub.cached_file).parameters
    assert "local_files_only" not in params, (
        "transformers.cached_file now names local_files_only explicitly — the "
        "AutoProcessor kwarg filter may no longer drop it; re-check FR01."
    )


# -- W07c: only a genuine cache miss says "not in the local cache" ---------------------------


def test_a_load_error_on_a_complete_cache_surfaces_as_itself(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every declared file is on disk, so the error is not a miss; re-fetching would not fix it."""
    use_fixture_cache(monkeypatch, tmp_path)
    build_model_cache(tmp_path)
    install_fake_sentence_transformers(monkeypatch, error=OSError("disk I/O error"))

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    assert provider.available() is False
    assert provider.unavailable_reason() == "model load failed: OSError: disk I/O error"


def test_a_permission_error_is_never_reported_as_a_cache_miss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_fixture_cache(monkeypatch, tmp_path)
    (tmp_path / "hub").mkdir()
    install_fake_sentence_transformers(monkeypatch, error=PermissionError(13, "Permission denied", "/hub"))

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    assert provider.available() is False
    assert provider.unavailable_reason().startswith("model load failed: PermissionError:")


@pytest.mark.parametrize("state", [CacheState.ABSENT, CacheState.INCOMPLETE, CacheState.UNKNOWN])
def test_a_loader_miss_without_a_complete_cache_names_the_fetch_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, state: CacheState
) -> None:
    """UNKNOWN (the cache could not be inspected) keeps the miss: a local-files-only load that
    finds nothing is the one failure the fetch command fixes."""
    from trw_memory.embeddings._hf_cache import CacheProbe

    use_fixture_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(local_mod, "probe_model_cache", lambda _name: CacheProbe(state))
    install_fake_sentence_transformers(monkeypatch, error=FileNotFoundError("config.json"))

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    with pytest.raises(ModelNotCachedError, match="trw-mcp models fetch"):
        provider.available()
