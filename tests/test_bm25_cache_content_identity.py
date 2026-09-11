"""Updating recalled knowledge must invalidate its cached lexical evidence."""

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval import bm25


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    pytest.importorskip("rank_bm25")
    monkeypatch.setattr(bm25, "_bm25_cache", None)


@pytest.mark.parametrize("field", ["content", "detail", "tags"])
def test_same_ids_updated_lexical_field_changes_winner(field):
    def corpus(swapped):
        values = ["apples", "telemetry"] if swapped else ["telemetry", "apples"]
        return [
            MemoryEntry(id=entry_id, **({"content": "incident"} | {field: [value] if field == "tags" else value}))
            for entry_id, value in zip(["one", "two"], values, strict=True)
        ]

    assert bm25.bm25_search("telemetry", corpus(False))[0][0] == "one"
    assert bm25.bm25_search("telemetry", corpus(True))[0][0] == "two"


def test_unchanged_reordered_corpus_reuses_model():
    entries = [MemoryEntry(id="one", content="telemetry"), MemoryEntry(id="two", content="apples")]
    before = bm25.bm25_search("telemetry", entries)
    model = bm25._bm25_cache[1]
    assert bm25.bm25_search("telemetry", list(reversed(entries))) == before
    assert bm25._bm25_cache[1] is model


def test_nonlexical_update_does_not_rebuild_model():
    entries = [MemoryEntry(id="one", content="telemetry"), MemoryEntry(id="two", content="apples")]
    before = bm25.bm25_search("telemetry", entries)
    model = bm25._bm25_cache[1]
    changed = [entry.model_copy(update={"importance": 0.99}) for entry in entries]
    assert bm25.bm25_search("telemetry", changed) == before
    assert bm25._bm25_cache[1] is model
