"""Tests for the embedding-free lexical dedup fallback (check_duplicate).

When embeddings are unavailable, check_duplicate must still catch EXACT
normalized-text duplicates instead of silently no-op'ing — the gap that let a
project's store accumulate ~79% near-duplicates (identical summaries repeated
dozens of times). Exact-match only ⇒ zero false-positive risk.
"""

from __future__ import annotations

from trw_memory.lifecycle.dedup import (
    DedupResult,
    _lexical_duplicate,
    _normalize_text,
    check_duplicate,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus

from ._test_dedup_support import StubEmbedder, make_entry


class _AvailableButNoneEmbedder(StubEmbedder):
    """available()==True but embed()/embed_batch() return None (broken provider)."""

    def __init__(self) -> None:
        super().__init__(available=True)

    def embed(self, text: str) -> list[float] | None:
        return None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [None for _ in texts]


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------


def test_normalize_collapses_whitespace_and_casefolds() -> None:
    assert _normalize_text("  Use   Absolute\tPaths\n") == "use absolute paths"


def test_normalize_empty_is_empty() -> None:
    assert _normalize_text("   \n\t ") == ""


# ---------------------------------------------------------------------------
# _lexical_duplicate
# ---------------------------------------------------------------------------


def test_lexical_duplicate_matches_active_exact() -> None:
    entries = [make_entry("e1", "use absolute paths", detail="always")]
    result = _lexical_duplicate("use absolute paths", "always", entries)
    # merge (not skip) so re-learn metadata folds in + recurrence increments.
    assert result == DedupResult("merge", "e1", 1.0)


def test_lexical_duplicate_skips_non_active() -> None:
    entries = [make_entry("e1", "dup", detail="x", status=MemoryStatus.OBSOLETE)]
    assert _lexical_duplicate("dup", "x", entries) is None


def test_lexical_duplicate_empty_target_is_none() -> None:
    entries = [make_entry("e1", "something")]
    assert _lexical_duplicate("   ", "", entries) is None


# ---------------------------------------------------------------------------
# check_duplicate — embedder unavailable
# ---------------------------------------------------------------------------


def test_unavailable_embedder_merges_exact_duplicate() -> None:
    embedder = StubEmbedder(available=False)
    entries = [make_entry("e1", "phase field empty in JSONL logs")]
    result = check_duplicate("phase field empty in JSONL logs", entries, embedder)
    assert result == DedupResult("merge", "e1", 1.0)


def test_unavailable_embedder_normalizes_before_match() -> None:
    embedder = StubEmbedder(available=False)
    entries = [make_entry("e1", "Use   Absolute Paths", detail="Always")]
    result = check_duplicate("use absolute paths", entries, embedder, detail="always")
    assert result == DedupResult("merge", "e1", 1.0)


def test_unavailable_embedder_non_duplicate_returns_store() -> None:
    embedder = StubEmbedder(available=False)
    entries = [make_entry("e1", "some content")]
    result = check_duplicate("different content", entries, embedder)
    assert result == DedupResult("store", None, 0.0)


def test_unavailable_embedder_ignores_obsolete_duplicate() -> None:
    embedder = StubEmbedder(available=False)
    entries = [make_entry("e1", "dup content", status=MemoryStatus.OBSOLETE)]
    result = check_duplicate("dup content", entries, embedder)
    assert result == DedupResult("store", None, 0.0)


# ---------------------------------------------------------------------------
# check_duplicate — embed() returns None despite available()
# ---------------------------------------------------------------------------


def test_embed_returns_none_falls_back_to_lexical() -> None:
    embedder = _AvailableButNoneEmbedder()
    entries = [make_entry("e1", "broken embedder dup")]
    result = check_duplicate("broken embedder dup", entries, embedder)
    assert result == DedupResult("merge", "e1", 1.0)


# ---------------------------------------------------------------------------
# kill switch
# ---------------------------------------------------------------------------


def test_fallback_disabled_restores_legacy_noop() -> None:
    embedder = StubEmbedder(available=False)
    entries = [make_entry("e1", "exact dup")]
    config = MemoryConfig(dedup_lexical_fallback=False)
    result = check_duplicate("exact dup", entries, embedder, config=config)
    assert result == DedupResult("store", None, 0.0)


def test_config_default_enables_fallback() -> None:
    assert MemoryConfig().dedup_lexical_fallback is True
