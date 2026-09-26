"""PLAN W40: ``fetch_models`` is the one code path that downloads a model."""

from __future__ import annotations

import sys
import types

import pytest

from trw_memory._model_pin import DEFAULT_EMBEDDING_MODEL, pinned_revision
from trw_memory.embeddings import fetch_models

pytestmark = pytest.mark.unit


def _fake_library(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict[str, object]]]:
    loads: list[tuple[str, str, dict[str, object]]] = []

    def loader(kind: str) -> type:
        class _Loader:
            def __init__(self, model: str, **kwargs: object) -> None:
                loads.append((kind, model, kwargs))

        return _Loader

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = loader("embedder")  # type: ignore[attr-defined]
    module.CrossEncoder = loader("reranker")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return loads


def test_both_configured_models_are_fetched_at_their_pinned_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _fake_library(monkeypatch)

    fetched = fetch_models()

    pinned = pinned_revision(DEFAULT_EMBEDDING_MODEL)
    assert fetched == {DEFAULT_EMBEDDING_MODEL: pinned, "cross-encoder/ms-marco-MiniLM-L-6-v2": "main"}
    assert [(kind, model, kwargs["revision"]) for kind, model, kwargs in loads] == [
        ("embedder", DEFAULT_EMBEDDING_MODEL, pinned),
        ("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2", "main"),
    ]
    # A fetch is the one load that may reach the Hub.
    assert all("local_files_only" not in kwargs for _kind, _model, kwargs in loads)


def test_an_explicit_model_overrides_the_configured_one(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _fake_library(monkeypatch)

    assert fetch_models(embedding_model="org/other")["org/other"] == "main"
    assert loads[0][1] == "org/other"
