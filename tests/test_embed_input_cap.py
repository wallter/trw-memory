"""The embedding provider bounds every input before the shared encode lock (rc9 sweep round 3).

memory_update's correction path handed an 8M-char patch to ``embed`` on the daemon's serialized
lane; the cap lives in the provider so every caller is bounded, not each call site.
"""

from __future__ import annotations

from unittest.mock import patch

from trw_memory.embeddings.local import MAX_EMBED_INPUT_CHARS, LocalEmbeddingProvider

_HUGE = "x" * 8_000_000


class _RecordingModel:
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def encode(self, texts: str | list[str], **_kwargs: object) -> list[float] | list[list[float]]:
        if isinstance(texts, str):
            self.lengths.append(len(texts))
            return [0.0] * 4
        self.lengths.extend(len(text) for text in texts)
        return [[0.0] * 4 for _ in texts]


def _provider(model: _RecordingModel) -> LocalEmbeddingProvider:
    provider = LocalEmbeddingProvider()
    patch.object(provider, "_load_model", return_value=model).start()
    return provider


def test_embed_hands_the_model_at_most_the_cap() -> None:
    model = _RecordingModel()
    try:
        assert _provider(model).embed(_HUGE) is not None
        assert _provider(model).embed_query(_HUGE) is not None
    finally:
        patch.stopall()
    assert model.lengths == [MAX_EMBED_INPUT_CHARS, MAX_EMBED_INPUT_CHARS]


def test_embed_batch_caps_each_text_and_keeps_short_ones_whole() -> None:
    model = _RecordingModel()
    try:
        vectors = _provider(model).embed_batch([_HUGE, "short", "  "])
    finally:
        patch.stopall()
    assert model.lengths == [MAX_EMBED_INPUT_CHARS, len("short")]
    assert [vector is None for vector in vectors] == [False, False, True]
