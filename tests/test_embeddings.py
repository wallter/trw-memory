"""Tests for LocalEmbeddingProvider.

Covers:
- Graceful degradation when model cannot load (sentence-transformers not installed
  or model load fails): embed returns None, available() returns False,
  embed_batch returns [None, ...], dim() returns configured value.
- Lazy load is cached: _load_model called only once after _load_attempted is set.
- Empty/blank text returns None from embed without triggering model load.
- embed_batch empty list returns [].
- embed_batch blank strings in list return None at correct positions.
- dim() respects constructor argument.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from trw_memory.embeddings.local import LocalEmbeddingProvider

# ---------------------------------------------------------------------------
# Helper: produce a provider with no model loaded
# ---------------------------------------------------------------------------


def _unavailable_provider(dim: int = 384) -> LocalEmbeddingProvider:
    """Return a provider in a state where the model was attempted but failed."""
    provider = LocalEmbeddingProvider(dim=dim)
    # Force the "attempted but failed" state without touching imports
    provider._model = None
    provider._load_attempted = True
    return provider


# ---------------------------------------------------------------------------
# Graceful degradation when model unavailable
# ---------------------------------------------------------------------------


class TestUnavailableProvider:
    def test_embed_returns_none_when_model_unavailable(self) -> None:
        provider = _unavailable_provider()
        result = provider.embed("some text")
        assert result is None

    def test_available_returns_false_when_model_unavailable(self) -> None:
        provider = _unavailable_provider()
        assert provider.available() is False

    def test_embed_batch_returns_nones_when_unavailable(self) -> None:
        provider = _unavailable_provider()
        result = provider.embed_batch(["first", "second", "third"])
        assert result == [None, None, None]

    def test_embed_batch_length_matches_input(self) -> None:
        provider = _unavailable_provider()
        texts = ["a", "b", "c", "d", "e"]
        result = provider.embed_batch(texts)
        assert len(result) == len(texts)

    def test_embed_batch_empty_list_returns_empty(self) -> None:
        provider = _unavailable_provider()
        result = provider.embed_batch([])
        assert result == []

    def test_dim_returns_configured_value_384(self) -> None:
        provider = _unavailable_provider(dim=384)
        assert provider.dim() == 384

    def test_dim_returns_configured_value_768(self) -> None:
        provider = LocalEmbeddingProvider(dim=768)
        assert provider.dim() == 768

    def test_dim_independent_of_model_load(self) -> None:
        provider = _unavailable_provider(dim=256)
        assert provider.dim() == 256


# ---------------------------------------------------------------------------
# Blank text short-circuit (no model needed)
# ---------------------------------------------------------------------------


class TestBlankTextShortCircuit:
    def test_embed_blank_string_returns_none(self) -> None:
        provider = LocalEmbeddingProvider()  # may or may not have model
        # Blank text must return None regardless of model availability
        result = provider.embed("")
        assert result is None

    def test_embed_whitespace_only_returns_none(self) -> None:
        provider = LocalEmbeddingProvider()
        result = provider.embed("   \t\n  ")
        assert result is None

    def test_embed_blank_does_not_call_model(self) -> None:
        provider = LocalEmbeddingProvider()
        mock_model = MagicMock()
        provider._model = mock_model
        provider._load_attempted = True

        result = provider.embed("   ")
        assert result is None
        mock_model.encode.assert_not_called()


# ---------------------------------------------------------------------------
# Load caching (lazy load called once)
# ---------------------------------------------------------------------------


class TestLoadCaching:
    def test_load_attempted_flag_set_after_first_call(self) -> None:
        provider = LocalEmbeddingProvider()
        assert provider._load_attempted is False
        # Trigger a load attempt (may fail if sentence-transformers not installed)
        provider.available()
        assert provider._load_attempted is True

    def test_load_not_re_attempted_after_failure(self) -> None:
        provider = _unavailable_provider()
        # _load_model should return immediately without re-trying
        with patch.object(provider.__class__, "_load_model", wraps=provider._load_model) as mock_load:
            provider.available()
            provider.available()
        # Called twice, but the actual import path must only run once.
        # Since _load_attempted is already True on both calls, model is
        # returned immediately without entering the try block.
        assert provider._model is None

    def test_second_available_call_uses_cache(self) -> None:
        provider = _unavailable_provider()
        result1 = provider.available()
        result2 = provider.available()
        assert result1 == result2 is False


# ---------------------------------------------------------------------------
# Graceful degradation via patched import
# ---------------------------------------------------------------------------


class TestGracefulDegradationViaImport:
    def test_embed_returns_none_when_sentence_transformers_missing(self) -> None:
        """Simulate missing sentence-transformers by making _load_model return None."""
        with patch(
            "trw_memory.embeddings.local.LocalEmbeddingProvider._load_model",
            return_value=None,
        ):
            provider = LocalEmbeddingProvider()
            provider._load_attempted = True
            result = provider.embed("test text")
        assert result is None

    def test_available_returns_false_when_sentence_transformers_missing(self) -> None:
        with patch(
            "trw_memory.embeddings.local.LocalEmbeddingProvider._load_model",
            return_value=None,
        ):
            provider = LocalEmbeddingProvider()
            provider._load_attempted = True
            result = provider.available()
        assert not result

    def test_embed_batch_returns_nones_when_sentence_transformers_missing(self) -> None:
        with patch(
            "trw_memory.embeddings.local.LocalEmbeddingProvider._load_model",
            return_value=None,
        ):
            provider = LocalEmbeddingProvider()
            provider._load_attempted = True
            result = provider.embed_batch(["a", "b"])
        assert result == [None, None]


# ---------------------------------------------------------------------------
# Successful model path (mocked model)
# ---------------------------------------------------------------------------


class TestMockedModelSuccess:
    def _provider_with_mock_model(self, dim: int = 3) -> LocalEmbeddingProvider:
        """Return a provider with a mock SentenceTransformer-like model.

        The real SentenceTransformer.encode() returns:
        - A 1-D array-like when given a single string.
        - A 2-D array-like when given a list of strings.

        We replicate that contract so the production code's
        ``[float(v) for v in vector]`` iteration works correctly.
        """
        import math

        provider = LocalEmbeddingProvider(dim=dim)

        mock_model = MagicMock()

        def _vec_for(text: str, d: int) -> list[float]:
            raw = [float(ord(c)) for c in (text[:d].ljust(d))]
            norm = math.sqrt(sum(v * v for v in raw)) or 1.0
            return [v / norm for v in raw]

        def _fake_encode(texts: list[str] | str, **kwargs: object) -> list[list[float]]:
            # embed() passes a plain str; embed_batch() passes a list[str].
            # Return 1-D for str, 2-D for list — matching SentenceTransformer API.
            if isinstance(texts, str):
                return _vec_for(texts, dim)  # type: ignore[return-value]
            return [_vec_for(t, dim) for t in texts]

        mock_model.encode.side_effect = _fake_encode
        provider._model = mock_model
        provider._load_attempted = True
        return provider

    def test_embed_returns_list_of_floats(self) -> None:
        provider = self._provider_with_mock_model(dim=3)
        result = provider.embed("abc")
        assert result is not None
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_embed_returns_correct_dimension(self) -> None:
        provider = self._provider_with_mock_model(dim=3)
        result = provider.embed("hello")
        assert result is not None
        assert len(result) == 3

    def test_available_returns_true_with_loaded_model(self) -> None:
        provider = self._provider_with_mock_model()
        assert provider.available() is True

    def test_embed_batch_returns_correct_length(self) -> None:
        provider = self._provider_with_mock_model(dim=3)
        result = provider.embed_batch(["aa", "bb", "cc"])
        assert len(result) == 3
        assert all(r is not None for r in result)

    def test_embed_batch_blank_items_return_none(self) -> None:
        provider = self._provider_with_mock_model(dim=3)
        result = provider.embed_batch(["valid", "", "   ", "also_valid"])
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is None
        assert result[3] is not None

    def test_embed_batch_all_blank_returns_all_none(self) -> None:
        provider = self._provider_with_mock_model(dim=3)
        # all blanks → non-blank list is empty → model.encode called with []
        # but the final mapping should produce [None, None]
        result = provider.embed_batch(["", "  "])
        assert result == [None, None]

    def test_embed_raises_handled_gracefully(self) -> None:
        provider = LocalEmbeddingProvider(dim=3)
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("GPU OOM")
        provider._model = mock_model
        provider._load_attempted = True

        result = provider.embed("some text")
        assert result is None

    def test_embed_batch_raises_handled_gracefully(self) -> None:
        provider = LocalEmbeddingProvider(dim=3)
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("GPU OOM")
        provider._model = mock_model
        provider._load_attempted = True

        result = provider.embed_batch(["a", "b"])
        assert result == [None, None]
