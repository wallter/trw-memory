"""Whole-word lexical relevance — the one tokenizer the recall path shares.

PRD-CORE-278 FR02/FR04. This is a LEAF module on purpose: the retrieval pipeline
and the lifecycle ranker both need the same notion of "does this query word
appear in this entry", and neither may import the other (``lifecycle`` already
imports ``retrieval`` for the expiry predicate, so the reverse edge would close
a cycle).

Two corrections to what the ranker did before:

- **Whole words, not substrings.** ``rank_by_utility`` used ``token in content``,
  so the query token ``is`` matched ``this``, ``list`` and ``distilled``. Tokens
  are now compared against the entry's own token set.
- **Stopwords removed.** A query like ``who is my boss`` carried three tokens
  that match almost any English text, so relevance was dominated by noise. A
  query that is ONLY stopwords keeps them, because ranking by nothing is worse
  than ranking by a weak signal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

#: Split on anything that is not alphanumeric, so ``favourite_language`` yields
#: ``favourite`` and ``language`` (the defect FR04 measures) and ``trw-memory``
#: yields ``trw`` and ``memory``.
_WORD_RE = re.compile(r"[a-z0-9]+")
#: The bounds on caller query text every ranking leg shares: FTS MATCH, BM25 and lexical (C12 rc7).
MAX_QUERY_CHARS = 1000
MAX_QUERY_TERMS = 64

#: CamelCase boundary, mirroring ``retrieval.bm25._CAMEL_RE`` so both surfaces
#: index ``hybridSearch`` as ``hybrid`` and ``search``.
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

#: Closed-class English words that carry no retrieval signal. Deliberately
#: short: a long list starts deleting domain words (``not``, ``no`` and ``all``
#: are already borderline in an engineering corpus).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }
)


def tokenize(text: str) -> list[str]:
    """Return lowercased whole-word tokens, splitting identifiers and CamelCase."""
    return _WORD_RE.findall(_CAMEL_RE.sub(" ", text).lower())


def bounded_query(query: str) -> str:
    """*query* itself within the bounds, else the first ``MAX_QUERY_TERMS`` whitespace chunks of its first ``MAX_QUERY_CHARS``."""
    chunks = query[:MAX_QUERY_CHARS].split()
    return (
        query
        if len(query) <= MAX_QUERY_CHARS and len(chunks) <= MAX_QUERY_TERMS
        else " ".join(chunks[:MAX_QUERY_TERMS])
    )


def tokenize_query(query: str) -> list[str]:
    """Return query tokens with stopwords removed.

    A query made entirely of stopwords keeps them: an empty token list would
    make every entry equally (ir)relevant, which is exactly the failure mode
    this function exists to remove.
    """
    tokens = tokenize(bounded_query(query))
    meaningful = [token for token in tokens if token not in _STOPWORDS]
    return meaningful or tokens


def lexical_relevance(entry: Mapping[str, object], query_tokens: list[str]) -> float:
    """Return whole-word relevance in ``[0, 1]`` for a serialised entry dict.

    Content hits weigh 3, tag hits 2 and detail hits 1 — the weighting the
    substring ranker used, kept so this change is a tokenization fix and not a
    silent re-weighting. An empty token list is the wildcard query and scores
    ``1.0``, preserving the documented wildcard behaviour.
    """
    if not query_tokens:
        return 1.0
    content = set(tokenize(str(entry.get("content", ""))))
    detail = set(tokenize(str(entry.get("detail", ""))))
    raw_tags = entry.get("tags", [])
    tags: set[str] = set()
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            tags.update(tokenize(str(tag)))
    weighted = sum(
        (3 if token in content else 0) + (2 if token in tags else 0) + (1 if token in detail else 0)
        for token in query_tokens
    )
    return min(1.0, weighted / (len(query_tokens) * 3))


__all__ = ["lexical_relevance", "tokenize", "tokenize_query"]
