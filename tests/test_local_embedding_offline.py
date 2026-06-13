"""PRD-QUAL-110-FR04: LocalEmbeddingProvider honors the offline switch.

The embedding init forces ``local_files_only=True`` (no huggingface.co
download) when ``TRW_OFFLINE`` or ``HF_HUB_OFFLINE`` is engaged, even if the
``local_only`` config field is False, and discloses the potential egress on a
network-capable first load.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from structlog.testing import capture_logs

from trw_memory.embeddings import local as local_mod


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
        def __init__(
            self, model_name: str, local_files_only: bool = False, trust_remote_code: bool = False
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


def test_online_load_discloses_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    """A network-capable load (not offline, not local_only) discloses egress."""

    class _FakeST:
        def __init__(self, model_name: str, local_files_only: bool = False) -> None:
            pass

        def encode(self, *a: object, **k: object) -> list[float]:
            return [0.0]

    import sys
    import types

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = _FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    provider = local_mod.LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    with capture_logs() as logs:
        provider.available()
    events = {e.get("event") for e in logs}
    assert "embedding_model_download_disclosure" in events
