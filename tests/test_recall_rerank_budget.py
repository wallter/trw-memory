"""The cross-encoder's latency levers on memory_recall: a per-call opt-out and a passage cap.

On the real TRW store one daemon recall page cost ~2 s, almost all of it the
cross-encoder re-rank and its bridge hop over long passages. ``rerank=False``
skips both for a caller with a latency budget; ``recall_rerank_passage_chars``
bounds what the model reads whenever it does run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client, FastMCP

from tests.conftest import make_entry
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.retrieval import reranker
from trw_memory.tools import recall as recall_tool

_NS = "project:rerank-33333333"
_LONG_DETAIL = "zephyrine eviction detail " * 40  # ~1,000 chars, past the cap under test


class _RecordingModel:
    """A stand-in cross-encoder that records every (query, passage) pair it scores."""

    def __init__(self) -> None:
        self.pairs: list[list[str]] = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.pairs.extend(pairs)
        return [0.0] * len(pairs)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> _RecordingModel:
    recording = _RecordingModel()
    monkeypatch.setattr(reranker, "_get_model", lambda _name: recording)
    return recording


@pytest.fixture
def config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_path=str(tmp_path), memory_single_store_path=str(tmp_path / "memory.db"))


def _recall(config: MemoryConfig, **kwargs: Any) -> dict[str, object]:
    with create_backend_from_config(config, _NS) as backend:
        for index in range(3):
            backend.store(
                make_entry(entry_id=f"M-{index}", content=f"zephyrine rule {index}", detail=_LONG_DETAIL, namespace=_NS)
            )
        return recall_tool.memory_recall_impl(
            "zephyrine eviction", _NS, backend=backend, limit=3, config=config, include_org_memories=False, **kwargs
        )


def test_rerank_false_never_calls_the_cross_encoder(config: MemoryConfig, model: _RecordingModel) -> None:
    result = _recall(config, rerank=False)

    assert model.pairs == []
    assert {row["id"] for row in result["memories"]} == {"M-0", "M-1", "M-2"}  # type: ignore[union-attr, index]


def test_the_default_still_reranks(config: MemoryConfig, model: _RecordingModel) -> None:
    """The opt-out is the only change: a caller that says nothing keeps the re-rank."""
    _recall(config)

    assert model.pairs


def test_every_reranked_passage_is_cut_to_the_configured_cap(tmp_path: Path, model: _RecordingModel) -> None:
    config = MemoryConfig(
        storage_path=str(tmp_path),
        memory_single_store_path=str(tmp_path / "memory.db"),
        recall_rerank_passage_chars=64,
    )
    _recall(config)

    assert model.pairs
    assert all(len(passage) <= 64 for _query, passage in model.pairs)
    assert any(len(passage) == 64 for _query, passage in model.pairs)  # the long rows were cut, not dropped


def test_cross_encode_scores_truncates_to_passage_chars(model: _RecordingModel) -> None:
    entry = make_entry(content="head", detail="x" * 5000)

    reranker.cross_encode_scores("q", [entry], passage_chars=100)
    reranker.cross_encode_scores("q", [entry])

    assert [len(passage) for _query, passage in model.pairs] == [100, reranker._MAX_PASSAGE_CHARS]


def test_the_default_cap_keeps_the_previous_cut() -> None:
    """The cap is an operator knob; its default is the 2048-char cut recall always used."""
    assert MemoryConfig().recall_rerank_passage_chars == reranker._MAX_PASSAGE_CHARS == 2048


async def test_the_served_tool_accepts_and_honours_rerank_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: _RecordingModel
) -> None:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("MEMORY_SINGLE_STORE_PATH", str(tmp_path / "memory.db"))
    with create_backend_from_config(MemoryConfig(), _NS) as backend:
        backend.store(make_entry(entry_id="M-0", content="zephyrine rule", detail=_LONG_DETAIL, namespace=_NS))
    mcp = FastMCP("daemon-under-test")
    recall_tool.register_recall_tool(mcp)

    async with Client(mcp) as client:
        answer = await client.call_tool(
            "memory_recall",
            {"query": "zephyrine", "namespace": _NS, "include_org_memories": False, "rerank": False},
        )

    assert [row["id"] for row in answer.data["memories"]] == ["M-0"]
    assert model.pairs == []
