"""PLAN W40: LocalEmbeddingProvider's runtime load is always cache-only.

No setting or environment variable turns a download on; models arrive through
``trw-mcp models fetch`` (``trw_memory.embeddings.fetch_models``). A busy GPU
still falls back to CPU.
"""

from __future__ import annotations

import pytest

from trw_memory.embeddings import local as local_mod


def test_every_load_is_local_files_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """No offline switch is set, and the load still never reaches the Hub."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    captured: dict[str, object] = {}

    class _FakeST:
        def __init__(
            self,
            model_name: str,
            revision: str = "main",
            local_files_only: bool = False,
            trust_remote_code: bool = False,
            device: str | None = None,
        ) -> None:
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


def _install_cuda_failing_st(monkeypatch: pytest.MonkeyPatch, *, message: str) -> list[dict[str, object]]:
    """Fake SentenceTransformer whose default-device load raises; ``device="cpu"`` succeeds."""
    # The CUDA retry is the non-macOS path: macOS already loads on CPU (INFERENCE_DEVICE).
    monkeypatch.setattr(local_mod, "INFERENCE_DEVICE", None)
    calls: list[dict[str, object]] = []

    class _FakeST:
        def __init__(
            self,
            model_name: str,
            revision: str = "main",
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
    calls = _install_cuda_failing_st(monkeypatch, message="CUDA error: out of memory")

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")

    assert provider.available() is True
    assert [c["device"] for c in calls] == [None, "cpu"]


def test_non_cuda_runtime_error_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only CUDA failures earn the CPU retry; any other RuntimeError stays a load failure."""
    calls = _install_cuda_failing_st(monkeypatch, message="tokenizer vocabulary mismatch")

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")

    assert provider.available() is False
    assert [c["device"] for c in calls] == [None]
    assert "runtime dependency failed" in provider._last_load_error
