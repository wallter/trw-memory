"""PRD-CORE-195 FR03 — store-time HyPE expansion (siblings, cap, fail-open).

NFR02: assert no ``event=`` kwarg on the HyPE log sites.
NFR03: guarded with importorskip for the optional vector extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from trw_memory._client_hype import hype_sibling_id
from trw_memory.client import MemoryClient


class _FakeEmbedder:
    """Deterministic 384-d embedder: hashes text to a stable unit-ish vector."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float] | None:
        if not text.strip():
            return None
        h = abs(hash(text))
        return [float((h >> (i % 31)) & 0xFF) / 255.0 for i in range(self._dim)]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(t) for t in texts]

    def available(self) -> bool:
        return True

    def dim(self) -> int:
        return self._dim


class _ListGenerator:
    def __init__(self, questions: list[str]) -> None:
        self._questions = questions

    def generate(self, entry: object) -> list[str]:
        return list(self._questions)


class _RaisingGenerator:
    def generate(self, entry: object) -> list[str]:
        raise RuntimeError("generator boom")


def _make_client(tmp_path: Path, generator: object, *, hype_enabled: bool = True) -> MemoryClient:
    client = MemoryClient(
        "default",
        mode="local",
        db_path=tmp_path / "hype.db",
        question_generator=generator,  # type: ignore[arg-type]
    )
    client._config.hype_enabled = hype_enabled
    client._get_embedder = lambda: _FakeEmbedder()  # type: ignore[method-assign]
    return client


async def test_siblings_written_for_each_question(tmp_path: Path) -> None:
    gen = _ListGenerator(["how do I enforce strict typing?", "what enforces strict mode?"])
    client = _make_client(tmp_path, gen)
    try:
        await client.store("Pydantic v2 strict mode", entry_id="E1")
        backend = client._get_backend()
        siblings = backend.hype_sibling_ids("E1")
        assert set(siblings) == {hype_sibling_id("E1", 0), hype_sibling_id("E1", 1)}
        # The primary vector for the parent is still present.
        assert backend.vector_exists("E1")
    finally:
        await client.close()


async def test_short_questions_skipped(tmp_path: Path) -> None:
    gen = _ListGenerator(["short", "a long enough question here"])  # first < 8 chars
    client = _make_client(tmp_path, gen)
    try:
        await client.store("content", entry_id="E2")
        siblings = client._get_backend().hype_sibling_ids("E2")
        assert siblings == [hype_sibling_id("E2", 0)]
    finally:
        await client.close()


async def test_cap_honoured(tmp_path: Path) -> None:
    gen = _ListGenerator([f"question number {i} padded out" for i in range(8)])
    client = _make_client(tmp_path, gen)
    client._config.hype_questions_per_entry = 3
    try:
        await client.store("content", entry_id="E3")
        siblings = client._get_backend().hype_sibling_ids("E3")
        assert len(siblings) == 3
    finally:
        await client.close()


async def test_duplicate_questions_deduped(tmp_path: Path) -> None:
    gen = _ListGenerator(["same question repeated", "same question repeated", "another distinct one"])
    client = _make_client(tmp_path, gen)
    try:
        await client.store("content", entry_id="E4")
        siblings = client._get_backend().hype_sibling_ids("E4")
        assert len(siblings) == 2
    finally:
        await client.close()


async def test_disabled_writes_no_siblings(tmp_path: Path) -> None:
    gen = _ListGenerator(["a perfectly good question string"])
    client = _make_client(tmp_path, gen, hype_enabled=False)
    try:
        result = await client.store("content", entry_id="E5")
        assert result["status"] == "stored"
        assert client._get_backend().hype_sibling_ids("E5") == []
    finally:
        await client.close()


async def test_fail_open_when_generator_raises(tmp_path: Path) -> None:
    client = _make_client(tmp_path, _RaisingGenerator())
    try:
        # Store must SUCCEED despite the generator blowing up (fail-open).
        result = await client.store("Pydantic v2 strict mode", entry_id="E6")
        assert result["status"] == "stored"
        backend = client._get_backend()
        assert backend.vector_exists("E6")  # primary vector committed
        assert backend.hype_sibling_ids("E6") == []  # no siblings
    finally:
        await client.close()


async def test_empty_generator_writes_no_siblings(tmp_path: Path) -> None:
    # Generator yields nothing → no siblings, store still succeeds (line 108-110).
    client = _make_client(tmp_path, _ListGenerator([]))
    try:
        result = await client.store("content", entry_id="E7")
        assert result["status"] == "stored"
        assert client._get_backend().hype_sibling_ids("E7") == []
    finally:
        await client.close()


async def test_none_embedding_question_skipped(tmp_path: Path) -> None:
    # An embedder that returns None for a specific question skips that sibling
    # only (line 119-120), keeping the others.
    class _SelectiveEmbedder(_FakeEmbedder):
        def embed(self, text: str) -> list[float] | None:
            if "skipme" in text:
                return None
            return super().embed(text)

    gen = _ListGenerator(["this one skipme please", "this one is kept fine"])
    client = _make_client(tmp_path, gen)
    client._get_embedder = lambda: _SelectiveEmbedder()  # type: ignore[method-assign]
    try:
        await client.store("content", entry_id="E8")
        siblings = client._get_backend().hype_sibling_ids("E8")
        # Only index 1 survives (index 0 embedded to None).
        assert siblings == [hype_sibling_id("E8", 1)]
    finally:
        await client.close()


async def test_no_embedder_writes_no_siblings(tmp_path: Path) -> None:
    # embedding_has_consumer gates embedder=None → HyPE branch returns early
    # (line 105-106) after purging; store succeeds.
    gen = _ListGenerator(["a perfectly good question string"])
    client = _make_client(tmp_path, gen)
    client._get_embedder = lambda: None  # type: ignore[method-assign]
    try:
        result = await client.store("content", entry_id="E9")
        assert result["status"] == "stored"
        assert client._get_backend().hype_sibling_ids("E9") == []
    finally:
        await client.close()


async def test_no_event_kwarg_on_hype_log_sites() -> None:
    # NFR02: the store-side HyPE module must not use the reserved `event=` kwarg.
    source = Path(__file__).resolve().parents[1] / "src" / "trw_memory" / "_client_hype.py"
    text = source.read_text(encoding="utf-8")
    assert "event=" not in text
