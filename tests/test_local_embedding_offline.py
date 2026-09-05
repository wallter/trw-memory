"""PRD-QUAL-110-FR04: LocalEmbeddingProvider honors the offline switch.

The embedding init forces ``local_files_only=True`` (no huggingface.co
download) when ``TRW_OFFLINE`` or ``HF_HUB_OFFLINE`` is engaged, even if the
``local_only`` config field is False, and discloses the potential egress on a
network-capable first load.

PRD-SEC-014-FR01 narrowed what "network-capable" means: the disclosure is
emitted only when the cache cannot answer, so the egress-disclosure case now
pins ``HF_HOME`` at an empty directory instead of depending on whatever the
developer happens to have cached. The paired
``test_complete_cache_suppresses_the_disclosure`` covers the other side.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from trw_memory.embeddings import local as local_mod

from ._test_hf_cache_support import build_model_cache, use_fixture_cache


@pytest.fixture(autouse=True)
def _clear_offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("TRW_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    yield


def test_offline_helper_detects_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRW_OFFLINE", "1")
    assert local_mod._offline_download_blocked() is True
    monkeypatch.delenv("TRW_OFFLINE")
    monkeypatch.setenv("HF_HUB_OFFLINE", "yes")
    assert local_mod._offline_download_blocked() is True
    monkeypatch.delenv("HF_HUB_OFFLINE")
    assert local_mod._offline_download_blocked() is False


def test_offline_forces_local_files_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """With TRW_OFFLINE=1, the model is loaded with local_files_only=True."""
    monkeypatch.setenv("TRW_OFFLINE", "1")
    captured: dict[str, object] = {}

    class _FakeST:
        def __init__(self, model_name: str, local_files_only: bool = False, trust_remote_code: bool = False) -> None:
            captured["model_name"] = model_name
            captured["local_files_only"] = local_files_only

        def encode(self, *a: object, **k: object) -> list[float]:
            return [0.0]

    import sys
    import types

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = _FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    assert provider.available() is True
    assert captured["local_files_only"] is True


def _install_fake_st(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeST:
        def __init__(
            self,
            model_name: str,
            local_files_only: bool = False,
            trust_remote_code: bool = False,
        ) -> None:
            pass

        def encode(self, *a: object, **k: object) -> list[float]:
            return [0.0]

    import sys
    import types

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = _FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)


def test_online_load_discloses_egress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A network-capable load (no cache, not offline, not local_only) discloses egress."""
    use_fixture_cache(monkeypatch, tmp_path)
    (tmp_path / "hub").mkdir()
    _install_fake_st(monkeypatch)

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    with capture_logs() as logs:
        provider.available()
    events = {e.get("event") for e in logs}
    assert "embedding_model_download_disclosure" in events


def test_complete_cache_suppresses_the_disclosure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """PRD-SEC-014-FR01: no egress is possible on a warm cache, so none is disclosed."""
    use_fixture_cache(monkeypatch, tmp_path)
    build_model_cache(tmp_path)
    _install_fake_st(monkeypatch)

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    with capture_logs() as logs:
        provider.available()
    events = {e.get("event") for e in logs}
    assert "embedding_model_download_disclosure" not in events


def _install_cuda_failing_st(monkeypatch: pytest.MonkeyPatch, *, message: str) -> list[dict[str, object]]:
    """Fake SentenceTransformer whose default-device load raises; ``device="cpu"`` succeeds."""
    calls: list[dict[str, object]] = []

    class _FakeST:
        def __init__(
            self,
            model_name: str,
            local_files_only: bool = False,
            trust_remote_code: bool = False,
            device: str | None = None,
        ) -> None:
            calls.append({"model_name": model_name, "device": device})
            if device is None:
                raise RuntimeError(message)

        def encode(self, *a: object, **k: object) -> list[float]:
            return [0.0]

    import sys
    import types

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = _FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    return calls


def test_cuda_load_failure_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CUDA out-of-memory at load time retries on CPU instead of disabling embeddings."""
    monkeypatch.setenv("TRW_OFFLINE", "1")
    calls = _install_cuda_failing_st(monkeypatch, message="CUDA error: out of memory")

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")

    assert provider.available() is True
    assert [c["device"] for c in calls] == [None, "cpu"]


def test_non_cuda_runtime_error_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only CUDA failures earn the CPU retry; any other RuntimeError stays a load failure."""
    monkeypatch.setenv("TRW_OFFLINE", "1")
    calls = _install_cuda_failing_st(monkeypatch, message="tokenizer vocabulary mismatch")

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")

    assert provider.available() is False
    assert [c["device"] for c in calls] == [None]
    assert "runtime dependency failed" in provider._last_load_error
