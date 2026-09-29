"""Shared fixture for PRD-CORE-336 FR01: one store, both recall routes.

Seeds one SQLite store with rows of every source family, with fixed ids and
timestamps so a response is reproducible byte for byte, and runs the SAME query
through the library route (``MemoryClient.recall``) and the daemon route
(``memory_recall_impl``). The embedder is off and the cross-encoder is stubbed on
both routes, so the ranking is BM25 + the pipeline only and is deterministic.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.source_policy import classify_source_family
from trw_memory.tools import recall as recall_tool

QUERY = "retry backoff queue"
_STAMP = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
_GIT = {"source": "distilled:git:aaa..bbb"}

#: (id, content, metadata). Distilled rows match the query strongly, so a weight
#: below 1 moves them across non-distilled rows.
NON_DISTILLED_ROWS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("M-ins", "retry backoff policy for the worker queue", {"source_kind": "instruction_rule"}),
    ("M-sem", "queue retry semantics", {"source_kind": "semantic_memory"}),
    ("M-life", "retry bulletin for the queue rollout", {"source_kind": "lifecycle"}),
    ("M-epi", "one retry after a timeout", {"source_kind": "episodic"}),
    ("M-unk-a", "retry backoff queue jitter ceiling", {}),
    ("M-unk-b", "backoff queue depth alarm", {}),
    ("M-unk-c", "queue consumer lag", {}),
    ("M-unk-d", "retry budget", {}),
    ("M-unk-e", "backoff for flaky network calls", {}),
)
DISTILLED_ROWS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("M-git-a", "retry backoff queue: cap the backoff and add jitter to the queue retry", _GIT),
    ("M-git-b", "retry backoff queue drain order on shutdown retry", _GIT),
    ("M-git-c", "queue retry backoff reset after success", _GIT),
)


def _graded(entry_id: str, retries: int, metadata: dict[str, str]) -> tuple[str, str, dict[str, str]]:
    """Equal-length rows whose only difference is how often "retry" occurs.

    BM25 then orders them by that count whatever the corpus size, so the
    daemon's pool (which also holds the security canary rows) and the library's
    pool rank them identically before any weight is applied.
    """
    return entry_id, " ".join(["retry"] * retries + ["backoff", "queue"] + ["note"] * (8 - retries)), metadata


#: The FR01 parity fixture: 3 git_distilled and 5 ``unknown`` rows, the
#: distilled ones spread across the order so a weight of 0.5 moves them past
#: ``unknown`` rows on either score basis.
PARITY_ROWS: tuple[tuple[str, str, dict[str, str]], ...] = (
    _graded("M-git-a", 8, _GIT),
    _graded("M-unk-a", 7, {}),
    _graded("M-unk-b", 6, {}),
    _graded("M-git-b", 5, _GIT),
    _graded("M-unk-c", 4, {}),
    _graded("M-unk-d", 3, {}),
    _graded("M-git-c", 2, _GIT),
    _graded("M-unk-e", 1, {}),
)
#: Rows matching no query term. Without them every candidate contains "retry",
#: its unsmoothed BM25 IDF is <= 0 on the library's pool (the daemon's pool
#: also holds the canary rows, so its IDF stays positive), and the two routes
#: would rank the fixture in opposite orders before any weight applies.
PARITY_PADDING: tuple[tuple[str, str, dict[str, str]], ...] = tuple(
    (f"M-pad-{index}", f"unrelated gardening note {index}", {}) for index in range(6)
)


def seed(client: MemoryClient, rows: tuple[tuple[str, str, dict[str, str]], ...]) -> None:
    backend = client._get_backend()
    for entry_id, content, metadata in rows:
        backend.store(
            MemoryEntry(
                id=entry_id,
                content=content,
                namespace="default",
                created_at=_STAMP,
                updated_at=_STAMP,
                last_accessed_at=_STAMP,
                valid_from=_STAMP,
                metadata=dict(metadata),
            )
        )


def make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    client = MemoryClient(namespace="default", mode="local")
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    monkeypatch.setattr(client, "_record_recall_access", AsyncMock(return_value=None))
    monkeypatch.setattr(client, "_apply_pending_remote_retirements", AsyncMock(return_value=None))
    monkeypatch.setattr(recall_tool, "get_local_embedder", lambda **_: None)
    return client


def stub_cross_encoder(monkeypatch: pytest.MonkeyPatch, score: Callable[[MemoryEntry], float] | None) -> None:
    """``None`` makes the cross-encoder unavailable (fusion order kept, ``fused`` basis)."""

    def cross_encode_scores(
        _query: str, entries: list[MemoryEntry], **_: Any
    ) -> list[tuple[MemoryEntry, float]] | None:
        if score is None:
            return None
        return sorted(((entry, score(entry)) for entry in entries), key=lambda pair: pair[1], reverse=True)

    monkeypatch.setattr("trw_memory.retrieval.reranker.cross_encode_scores", cross_encode_scores)


def reverse_id_rerank(entry: MemoryEntry) -> float:
    """A deterministic reordering rerank: descending by id, above every rerank floor."""
    return 10.0 + sum(ord(ch) for ch in entry.id) / 1000.0


def fewest_retries_rerank(entry: MemoryEntry) -> float:
    """A rerank that inverts the parity fixture's BM25 order, above every rerank floor."""
    return 20.0 - entry.content.split().count("retry")


async def library_recall(client: MemoryClient) -> list[dict[str, Any]]:
    rows = await client.recall(QUERY, limit=20, include_org_memories=False)
    return [dict(row) for row in rows]


def daemon_recall(client: MemoryClient, query: str = QUERY, **options: Any) -> dict[str, Any]:
    return recall_tool.memory_recall_impl(
        query,
        "default",
        backend=client._get_backend(),
        config=client._config,
        limit=20,
        include_org_memories=False,
        record_access=False,
        **options,
    )


def is_distilled(row: dict[str, Any]) -> bool:
    return classify_source_family(row) == "git_distilled"


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("memory_id") or row.get("id"))


#: Row fields added to ``MemoryEntry`` AFTER the goldens were recorded. They are not ranking
#: output, so the byte-for-byte golden drops them. Every other field (id, order, score and
#: the rest of the row) still has to match exactly. Add an entry here only for a new
#: schema field, with its PRD, never to absorb a ranking or score change.
FIELDS_ADDED_AFTER_GOLDENS: dict[str, str] = {"evidence_level": "PRD-CORE-312 FR01"}


def _without_later_fields(value: object) -> object:
    if isinstance(value, dict):
        return {k: _without_later_fields(v) for k, v in value.items() if k not in FIELDS_ADDED_AFTER_GOLDENS}
    if isinstance(value, list):
        return [_without_later_fields(v) for v in value]
    return value


def canonical(response: object) -> str:
    """The byte form a golden compares: sorted keys, floats by ``repr``, later schema fields dropped."""
    return json.dumps(_without_later_fields(response), sort_keys=True, default=str)


def non_distilled_view(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(id, repr(score)) of every non-distilled row, in response order."""
    return [(row_id(row), repr(float(row["score"]))) for row in rows if not is_distilled(row)]


#: Recorded on int b4a83e07f, before the pipeline weighted distilled rows.
GOLDENS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "data" / "source_weighting_goldens.json").read_text(encoding="utf-8")
)


def expected_daemon_weighted(preweight: list[list[str]], weight: float) -> list[tuple[str, str]]:
    """The pre-weight daemon response with only git_distilled scores multiplied, stably re-sorted."""
    scored = [
        (row_id, float(score) * weight if row_id.startswith("M-git") else float(score)) for row_id, score in preweight
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [(row_id, repr(score)) for row_id, score in scored]


def golden_view(key: str) -> list[tuple[str, str]]:
    """A recorded (id, repr(score)) view, as ``non_distilled_view`` returns it."""
    return [(row_id, score) for row_id, score in GOLDENS[key]]
