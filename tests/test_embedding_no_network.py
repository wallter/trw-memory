"""PRD-SEC-014-FR06: a network-blocked regression on the REAL loader path.

``test_local_embedding_offline.py`` and ``test_embeddings.py`` only capture the
constructor arguments of a fake ``SentenceTransformer``; neither can observe a
Hub call, which is exactly why the reported warm-cache egress survived them.

Here every outbound connection attempt is intercepted at the socket layer and
counted, ``LocalEmbeddingProvider._load_model`` is the genuine implementation
(never patched), and the two zero-call resolutions are distinguished: one
reached through the cache with **both offline switches unset**, one reached
through ``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.embeddings import local as local_mod
from trw_memory.embeddings._hf_cache import CacheState, probe_model_cache

from ._test_hf_cache_support import (
    TINY_MODEL_DIM,
    NetworkSeam,
    build_loadable_model_cache,
    build_model_cache,
    install_fake_sentence_transformers,
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

    # The invariant must hold WITHOUT the switches: prove they are unset.
    assert os.environ.get("TRW_OFFLINE") is None
    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert local_mod._offline_download_blocked() is False

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    assert provider.available() is True

    assert seam.calls == 0
    assert captured["local_files_only"] is True


def test_offline_switch_path_also_makes_zero_network_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    seam: NetworkSeam,
) -> None:
    """FR06 companion: the offline switch reaches zero calls by its own route."""
    use_fixture_cache(monkeypatch, tmp_path)
    build_model_cache(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    captured = install_fake_sentence_transformers(monkeypatch)

    assert local_mod._offline_download_blocked() is True

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    assert provider.available() is True

    assert seam.calls == 0
    assert captured["local_files_only"] is True


def test_absent_cache_without_switches_would_reach_the_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    seam: NetworkSeam,
) -> None:
    """Non-vacuity: the seam DOES fire when the cache cannot answer.

    Without this, a zero-call assertion could pass because nothing in the test
    is capable of dialing out at all.
    """
    use_fixture_cache(monkeypatch, tmp_path)
    (tmp_path / "hub").mkdir()
    install_fake_sentence_transformers(monkeypatch)

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    with pytest.raises(AssertionError, match="network seam invoked"):
        provider.available()

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
    build_loadable_model_cache(tmp_path)

    assert local_mod._offline_download_blocked() is False
    assert probe_model_cache(local_mod._DEFAULT_MODEL).state is CacheState.COMPLETE

    provider = local_mod.LocalEmbeddingProvider(dim=TINY_MODEL_DIM)
    assert provider.available() is True
    vector = provider.embed("warm cache should never phone home")

    assert vector is not None
    assert len(vector) == TINY_MODEL_DIM
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
