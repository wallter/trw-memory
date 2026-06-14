"""PRD-CORE-195 FR04 — recall-path parent fusion, dedup, no-leak invariant.

Uses a controllable embedder: any text containing a ``@label`` token embeds to
the basis vector for that label, so a query and the sibling that paraphrases it
land at cosine 1.0 while everything else is orthogonal. This isolates the
collapse/fusion machinery from a real embedding model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from trw_memory._client_hype import is_hype_id
from trw_memory.client import MemoryClient

_DIM = 384


class _LabelEmbedder:
    """Embed ``@label`` tokens to orthogonal basis vectors; else a constant."""

    def __init__(self) -> None:
        self._labels: dict[str, int] = {}

    def _basis(self, idx: int) -> list[float]:
        vec = [0.0] * _DIM
        vec[idx % _DIM] = 1.0
        return vec

    def embed(self, text: str) -> list[float] | None:
        if not text.strip():
            return None
        for token in text.split():
            if token.startswith("@"):
                self._labels.setdefault(token, len(self._labels) + 1)
                return self._basis(self._labels[token])
        return self._basis(0)

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(t) for t in texts]

    def available(self) -> bool:
        return True

    def dim(self) -> int:
        return _DIM


class _MapGenerator:
    """Yield a fixed question list per entry id."""

    def __init__(self, by_id: dict[str, list[str]]) -> None:
        self._by_id = by_id

    def generate(self, entry: object) -> list[str]:
        return list(self._by_id.get(entry.id, []))  # type: ignore[attr-defined]


def _client(tmp_path: Path, generator: object, embedder: _LabelEmbedder) -> MemoryClient:
    client = MemoryClient(
        "default",
        mode="local",
        db_path=tmp_path / "recall.db",
        question_generator=generator,  # type: ignore[arg-type]
    )
    client._config.hype_enabled = True
    client._config.recall_auto_temporal = False
    client._get_embedder = lambda: embedder  # type: ignore[method-assign]
    return client


async def test_query_reaches_parent_via_sibling_not_synthetic_id(tmp_path: Path) -> None:
    emb = _LabelEmbedder()
    # Entry content has label @doc; its HyPE question has label @ask. A query
    # with @ask matches ONLY the sibling vector (content is orthogonal), so the
    # parent can only surface via HyPE collapse.
    gen = _MapGenerator({"D1": ["paraphrased question @ask here"]})
    client = _client(tmp_path, gen, emb)
    try:
        await client.store("statement content @doc", entry_id="D1")
        results = await client.recall("@ask", limit=5)
        ids = [r["memory_id"] for r in results]
        assert "D1" in ids  # reached via the sibling
        assert all(not is_hype_id(i) for i in ids)  # FR04 no-leak invariant
        assert ids.count("D1") == 1  # appears exactly once
    finally:
        await client.close()


async def test_parent_deduped_across_multiple_siblings(tmp_path: Path) -> None:
    emb = _LabelEmbedder()
    # Two siblings both share label @ask with the query, plus the primary
    # vector also matches (@ask in content) → parent must appear exactly once.
    gen = _MapGenerator({"D2": ["first paraphrase @ask one", "second paraphrase @ask two"]})
    client = _client(tmp_path, gen, emb)
    try:
        await client.store("content mentioning @ask directly", entry_id="D2")
        results = await client.recall("@ask", limit=5)
        ids = [r["memory_id"] for r in results]
        assert ids.count("D2") == 1
        assert all(not is_hype_id(i) for i in ids)
    finally:
        await client.close()


async def test_forgotten_parent_sibling_does_not_resurface(tmp_path: Path) -> None:
    emb = _LabelEmbedder()
    gen = _MapGenerator({"D3": ["paraphrase @ask gone"]})
    client = _client(tmp_path, gen, emb)
    try:
        await client.store("content @doc here", entry_id="D3")
        await client.forget("D3")
        results = await client.recall("@ask", limit=5)
        ids = [r["memory_id"] for r in results]
        assert "D3" not in ids
        assert all(not is_hype_id(i) for i in ids)
    finally:
        await client.close()
