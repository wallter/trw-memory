"""Query-side stopwords and suffix stemming in BM25.

Function words are dropped from the query, never from documents; inflected
forms on either side meet their stem.
"""

from __future__ import annotations

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.bm25 import _QUERY_STOPWORDS, _stem_token, bm25_search


def _entries() -> list[MemoryEntry]:
    return [
        MemoryEntry(id="q1", content="Caroline: Cool! What did it look like?", tags=[]),
        MemoryEntry(id="q2", content="Caroline: I did some research on adoption agencies last week", tags=[]),
        MemoryEntry(id="q3", content="Melanie: what a day, did you see that?", tags=[]),
    ]


def test_content_terms_outrank_function_word_overlap() -> None:
    results = bm25_search("What did Caroline research?", _entries())
    assert results[0][0] == "q2"


def test_all_stopword_query_keeps_its_tokens() -> None:
    results = bm25_search("what did", _entries())
    assert {eid for eid, _ in results} >= {"q1", "q3"}


def test_stopword_list_is_lowercase_single_tokens() -> None:
    assert all(w == w.lower() and " " not in w for w in _QUERY_STOPWORDS)
    assert {"what", "did", "the"} <= _QUERY_STOPWORDS


def test_stem_token_strips_common_suffixes_but_not_identifiers() -> None:
    assert [_stem_token(t) for t in ("researched", "agencies", "paintings", "quickly", "v2", "sqlite3", "fts")] == [
        "research",
        "agency",
        "painting",
        "quick",
        "v2",
        "sqlite3",
        "fts",
    ]


def test_inflected_document_meets_query_stem() -> None:
    entries = [
        MemoryEntry(id="s1", content="Caroline researched adoption agencies", tags=[]),
        MemoryEntry(id="s2", content="Melanie painted a sunrise", tags=[]),
    ]
    assert bm25_search("research agency", entries)[0][0] == "s1"
    assert bm25_search("painting a sunrise", entries)[0][0] == "s2"


def test_se_plurals_meet_their_singular_and_sibilants_still_strip_es() -> None:
    pairs = [("houses", "house"), ("phrases", "phrase"), ("responses", "response"), ("databases", "database")]
    assert [(_stem_token(a), _stem_token(b)) for a, b in pairs] == [(b, b) for _, b in pairs]
    assert [_stem_token(t) for t in ("classes", "churches", "wishes", "boxes", "quizzes")] == [
        "class",
        "church",
        "wish",
        "box",
        "quizz",
    ]
