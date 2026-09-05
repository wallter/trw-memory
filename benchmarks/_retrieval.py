"""Benchmark-local retrieval helpers.

These helpers prefer the shipped hybrid retrieval pipeline when it is available,
but fall back to deterministic token-overlap ranking so benchmark runs remain
meaningful in minimal offline environments without optional extras installed.
"""

from __future__ import annotations

import re

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.pipeline import hybrid_search
from trw_memory.security.namespace_scope import authorize_namespaces
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import StorageBackend

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Return normalized lowercase tokens for deterministic lexical ranking."""
    return _TOKEN_RE.findall(text.lower())


def rank_entries(
    query: str,
    entries: list[MemoryEntry],
    *,
    top_k: int = 10,
) -> list[MemoryEntry]:
    """Rank benchmark entries using hybrid retrieval or a deterministic fallback."""
    if not query.strip() or not entries:
        return []

    # The benchmark corpus is single-namespace; the scope is minted through the
    # authorizer like any other caller so the harness cannot silently grade a
    # retrieval policy the product cannot run (PRD-CORE-245 FR04).
    scope = authorize_namespaces(MemoryConfig(), {entry.namespace for entry in entries}, Permission.READ, "benchmark")
    ranked = hybrid_search(query=query, entries=entries, scope=scope, top_k=top_k)
    if ranked:
        return ranked[:top_k]

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    query_token_set = set(query_tokens)
    normalized_query = " ".join(query_tokens)
    scored: list[tuple[float, float, str, MemoryEntry]] = []

    for entry in entries:
        normalized_text = " ".join(_tokenize(f"{entry.content} {entry.detail} {' '.join(entry.tags)}"))
        if not normalized_text:
            continue

        text_tokens = set(normalized_text.split())
        overlap = len(query_token_set & text_tokens)
        if overlap == 0:
            continue

        coverage = overlap / len(query_token_set)
        phrase_bonus = 2.0 if normalized_query in normalized_text else 0.0
        tag_tokens = set(_tokenize(" ".join(entry.tags)))
        tag_bonus = len(query_token_set & tag_tokens) / len(query_token_set)
        score = coverage * 10.0 + phrase_bonus + tag_bonus + entry.importance
        scored.append((score, entry.importance, entry.id, entry))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [entry for *_ignored, entry in scored[:top_k]]


def search_backend_entries(
    backend: StorageBackend,
    query: str,
    *,
    namespace: str,
    candidate_limit: int,
    top_k: int = 10,
) -> list[MemoryEntry]:
    """Load benchmark candidates from a backend, then rank them for recall."""
    entries = backend.list_entries(namespace=namespace, limit=candidate_limit)
    return rank_entries(query, entries, top_k=top_k)
