"""PRD-SEC-014-FR01/NFR01/NFR02: the local cache decides ``local_files_only``.

Before this PRD the loader resolved ``bool(config.local_only) or offline`` and
never looked at the cache, so a machine holding the entire model snapshot still
permitted a huggingface.co revision check on the most basic write path. These
tests pin the new resolution, the once-per-instance budget, and the fail-open
degradation when the probe cannot answer.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from trw_memory.embeddings import _hf_cache
from trw_memory.embeddings import local as local_mod
from trw_memory.embeddings._hf_cache import CacheProbe, CacheState, probe_model_cache

from ._test_hf_cache_support import (
    DEFAULT_REPO_ID,
    NetworkSeam,
    build_model_cache,
    delete_blob_behind,
    install_fake_sentence_transformers,
    use_fixture_cache,
)

pytestmark = pytest.mark.integration

_PROBE_BUDGET_SECONDS = 0.05  # NFR01: p95 <= 50 ms added to a cold load


@pytest.fixture
def seam(monkeypatch: pytest.MonkeyPatch) -> Iterator[NetworkSeam]:
    """Refuse (and count) every outbound connection for the whole test."""
    installed = NetworkSeam()
    installed.install(monkeypatch)
    yield installed


def test_complete_cache_forces_local_files_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    seam: NetworkSeam,
) -> None:
    """FR01: a complete snapshot forces local_files_only with no offline switch."""
    use_fixture_cache(monkeypatch, tmp_path)
    snapshot = build_model_cache(tmp_path)
    captured = install_fake_sentence_transformers(monkeypatch)

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    assert provider.available() is True

    assert captured["local_files_only"] is True
    # The loader hands over the resolved snapshot DIRECTORY, not the repo id:
    # local_files_only alone does not stop transformers' AutoProcessor from
    # probing the Hub, because it drops every hub kwarg it is given.
    assert captured["model_name"] == str(snapshot.resolve())
    assert seam.calls == 0


def test_absent_cache_stays_network_capable_and_discloses_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FR01: no snapshot + no offline switch -> one disclosure naming host + switch."""
    use_fixture_cache(monkeypatch, tmp_path)
    (tmp_path / "hub").mkdir()
    captured = install_fake_sentence_transformers(monkeypatch)
    # The fake would dial the Hub; record the attempt instead of refusing it so
    # the network-capable disclosure path can be observed end to end.
    attempts: list[object] = []
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: attempts.append(a))

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    with capture_logs() as logs:
        provider.available()

    disclosures = [entry for entry in logs if entry.get("event") == "embedding_model_download_disclosure"]
    assert len(disclosures) == 1
    assert disclosures[0]["source"] == "huggingface.co"
    assert "TRW_OFFLINE" in disclosures[0]["detail"]
    assert captured["local_files_only"] is False
    assert len(attempts) == 1


def test_cache_probe_runs_once_per_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    seam: NetworkSeam,
) -> None:
    """NFR01: the probe is latched to one evaluation per provider, inside budget."""
    use_fixture_cache(monkeypatch, tmp_path)
    build_model_cache(tmp_path)
    install_fake_sentence_transformers(monkeypatch)

    calls: list[str] = []

    def _counting_probe(model_name: str) -> CacheProbe:
        calls.append(model_name)
        return probe_model_cache(model_name)

    monkeypatch.setattr(local_mod, "probe_model_cache", _counting_probe)

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    assert provider.available() is True
    provider.available()
    provider.embed("hello")
    assert calls == ["all-MiniLM-L6-v2"]

    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        probe_model_cache("all-MiniLM-L6-v2")
        elapsed.append(time.perf_counter() - start)
    assert max(elapsed) < _PROBE_BUDGET_SECONDS


def test_probe_failure_degrades_to_prior_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """NFR02: a raising probe logs once and falls back to config/env resolution."""
    use_fixture_cache(monkeypatch, tmp_path)
    build_model_cache(tmp_path)
    captured = install_fake_sentence_transformers(monkeypatch)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: None)

    def _exploding_probe(model_name: str) -> CacheProbe:
        raise RuntimeError("cache layout changed under us")

    monkeypatch.setattr(local_mod, "probe_model_cache", _exploding_probe)

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    with capture_logs() as logs:
        assert provider.available() is True

    degraded = [entry for entry in logs if entry.get("event") == "embedding_cache_probe_degraded"]
    assert len(degraded) == 1
    # Prior resolution: no offline switch, local_only False -> network-capable.
    assert captured["local_files_only"] is False


def test_probe_failure_still_honors_offline_switch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    seam: NetworkSeam,
) -> None:
    """NFR02: degradation never loosens the offline switch resolution."""
    use_fixture_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("TRW_OFFLINE", "1")
    captured = install_fake_sentence_transformers(monkeypatch)
    monkeypatch.setattr(local_mod, "probe_model_cache", _raise_probe)

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    assert provider.available() is True
    assert captured["local_files_only"] is True
    assert seam.calls == 0


def _raise_probe(model_name: str) -> CacheProbe:
    raise OSError("unreadable cache")


class TestCacheProbeStates:
    """Contract tests for the probe module's narrow interface."""

    def test_complete_snapshot(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        use_fixture_cache(monkeypatch, tmp_path)
        snapshot = build_model_cache(tmp_path)
        probe = probe_model_cache("all-MiniLM-L6-v2")
        assert probe.state is CacheState.COMPLETE
        assert probe.declares_remote_code is False
        assert probe.snapshot_path == str(snapshot.resolve())

    def test_incomplete_snapshot_carries_no_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Only a COMPLETE probe may be loaded from directly."""
        use_fixture_cache(monkeypatch, tmp_path)
        snapshot = build_model_cache(tmp_path)
        delete_blob_behind(snapshot, "model.safetensors")
        assert probe_model_cache("all-MiniLM-L6-v2").snapshot_path == ""

    def test_deleted_blob_is_incomplete_not_complete(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        use_fixture_cache(monkeypatch, tmp_path)
        snapshot = build_model_cache(tmp_path)
        delete_blob_behind(snapshot, "model.safetensors")
        assert probe_model_cache("all-MiniLM-L6-v2").state is CacheState.INCOMPLETE

    def test_unknown_repo_is_absent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        use_fixture_cache(monkeypatch, tmp_path)
        build_model_cache(tmp_path)
        assert probe_model_cache("some-org/never-downloaded").state is CacheState.ABSENT

    def test_missing_cache_dir_is_absent_not_an_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        use_fixture_cache(monkeypatch, tmp_path / "does-not-exist")
        assert probe_model_cache("all-MiniLM-L6-v2").state is CacheState.ABSENT

    def test_cached_repo_without_anchor_is_incomplete(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        use_fixture_cache(monkeypatch, tmp_path)
        snapshot = build_model_cache(tmp_path)
        (snapshot / "config.json").unlink()
        assert probe_model_cache(DEFAULT_REPO_ID).state is CacheState.INCOMPLETE

    def test_huggingface_hub_absent_is_unknown(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        use_fixture_cache(monkeypatch, tmp_path)
        build_model_cache(tmp_path)

        def _no_hub(repo_id: str, cache_dir: str | None) -> Path | None:
            raise ImportError("No module named 'huggingface_hub'")

        monkeypatch.setattr(_hf_cache, "_cached_snapshot_dir", _no_hub)
        assert probe_model_cache("all-MiniLM-L6-v2").state is CacheState.UNKNOWN

    def test_local_directory_model_is_complete(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        use_fixture_cache(monkeypatch, tmp_path)
        snapshot = build_model_cache(tmp_path)
        probe = probe_model_cache(str(snapshot))
        assert probe.state is CacheState.COMPLETE
