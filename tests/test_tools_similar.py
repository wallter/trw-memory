"""memory_similar: the trw_learn dedup verdict, decided by the daemon from text (PRD-CORE-302 FR01, contract C1)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from trw_memory.embeddings.local import FETCH_COMMAND
from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.exceptions import ModelNotCachedError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools import similar
from trw_memory.tools.similar import memory_similar_impl

pytestmark = pytest.mark.unit

SPACE_A = EmbeddingSpace("a" * 64, "test-encoder:a", 3)
SPACE_B = EmbeddingSpace("b" * 64, "test-encoder:b", 3)
NEW_TEXT = "retry the flaky upload with backoff"
NEAR = [1.0, 0.0, 0.0]
FAR = [0.0, 0.0, 1.0]


class _Embedder:
    """Encodes by lookup, in SPACE_A; a text it does not know lands FAR from everything."""

    model_name = "test-encoder"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.encoded: list[str] = []

    def available(self) -> bool:
        return True

    def embedding_space(self) -> EmbeddingSpace:
        return SPACE_A

    def embed(self, text: str) -> list[float] | None:
        self.encoded.append(text)
        return self.vectors.get(text, FAR)

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    def dim(self) -> int:
        return 3


@pytest.fixture
def backend(tmp_path: Path) -> SQLiteBackend:
    pytest.importorskip("sqlite_vec")
    return SQLiteBackend(tmp_path / "m.db", dim=3)


def _put(
    backend: SQLiteBackend,
    entry_id: str,
    vector: list[float] | None,
    space: EmbeddingSpace | None,
    **fields: object,
) -> None:
    entry = MemoryEntry(id=entry_id, content=f"content {entry_id}", namespace="default", **fields)  # type: ignore[arg-type]
    backend.store(entry)
    if vector is not None:
        proof = VectorProvenance.for_vector(space, f"{entry.content} {entry.detail}", vector) if space else None
        backend.upsert_vector(entry_id, vector, namespace="default", provenance=proof)


def _similar(
    backend: SQLiteBackend,
    embedder: object,
    monkeypatch: pytest.MonkeyPatch,
    *,
    text: str = NEW_TEXT,
    skip: float = 0.95,
    merge: float = 0.85,
) -> dict[str, object]:
    monkeypatch.setattr("trw_memory.tools._embedder.get_local_embedder", lambda **_kw: embedder)
    return memory_similar_impl("default", text, skip, merge, 10, backend=backend, config=MemoryConfig())


def test_a_near_copy_in_the_active_space_is_skipped_from_the_knn_window(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(backend, "L1", NEAR, SPACE_A)
    embedder = _Embedder({NEW_TEXT: NEAR})

    result = _similar(backend, embedder, monkeypatch)

    assert (result["status"], result["action"], result["existing_id"], result["mode"]) == ("ok", "skip", "L1", "knn")
    assert embedder.encoded == [NEW_TEXT]  # the window's stored vector was used, nothing re-encoded


def test_another_space_in_the_namespace_forces_exhaustive_mode_which_encodes_that_rows_text(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(backend, "L-a", FAR, SPACE_A)
    _put(backend, "L-b", [0.0, 1.0, 0.0], SPACE_B)  # its stored vector is meaningless here
    embedder = _Embedder({NEW_TEXT: NEAR, "content L-b ": NEAR})

    result = _similar(backend, embedder, monkeypatch)

    assert (result["action"], result["existing_id"], result["mode"]) == ("skip", "L-b", "exhaustive")
    assert result["examined"] == 2
    assert "content L-b " in embedder.encoded and "content L-a " not in embedder.encoded


def test_a_vectorless_namespace_is_still_scanned_as_6_1_0_did(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(backend, "L-plain", None, None)
    embedder = _Embedder({NEW_TEXT: NEAR, "content L-plain ": NEAR})

    result = _similar(backend, embedder, monkeypatch)

    assert (result["action"], result["existing_id"], result["mode"]) == ("skip", "L-plain", "exhaustive")


def test_exhaustive_mode_pages_through_every_row(backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(similar, "EXHAUSTIVE_PAGE", 2)
    for n in range(5):
        _put(backend, f"L{n}", None, None)
    embedder = _Embedder({NEW_TEXT: NEAR, "content L0 ": NEAR})

    result = _similar(backend, embedder, monkeypatch)

    assert (result["existing_id"], result["examined"]) == ("L0", 5)


def test_an_exhaustive_pass_past_its_row_budget_is_unavailable_not_a_store(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rc9 sweep B1: the shared daemon never encodes a whole namespace for one caller."""
    monkeypatch.setattr(similar, "EXHAUSTIVE_PAGE", 2)
    monkeypatch.setattr(similar, "EXHAUSTIVE_MAX_ROWS", 3, raising=False)
    for n in range(5):
        _put(backend, f"L{n}", None, None)
    embedder = _Embedder({NEW_TEXT: NEAR})

    result = _similar(backend, embedder, monkeypatch)

    assert result == {"status": "unavailable", "reason": "similar_budget", "fix": "trw-mcp memory reembed"}
    assert len(embedder.encoded) == 1 + 2  # the new text and the first page, never the rows past the budget


def test_an_empty_namespace_and_an_exactly_full_last_page_still_answer(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _similar(backend, _Embedder({NEW_TEXT: NEAR}), monkeypatch)["action"] == "store"

    monkeypatch.setattr(similar, "EXHAUSTIVE_PAGE", 2)
    for n in range(4):
        _put(backend, f"L{n}", None, None)
    result = _similar(backend, _Embedder({NEW_TEXT: NEAR, "content L3 ": NEAR}), monkeypatch)

    assert (result["action"], result["existing_id"], result["examined"]) == ("skip", "L3", 4)


def test_a_read_that_spends_the_budget_ends_the_pass_before_any_encode(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(backend, "L-plain", None, None)
    embedder = _Embedder({NEW_TEXT: NEAR, "content L-plain ": NEAR})
    real_list = backend.list_entries
    monkeypatch.setattr(similar, "EXHAUSTIVE_SECONDS", 0.05)

    def slow_list(**kwargs: object) -> object:
        time.sleep(0.1)
        return real_list(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(backend, "list_entries", slow_list)

    assert _similar(backend, embedder, monkeypatch)["reason"] == "similar_budget"
    assert embedder.encoded == [NEW_TEXT]


def test_an_exhaustive_pass_past_its_time_budget_is_unavailable(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(similar, "EXHAUSTIVE_SECONDS", -1.0, raising=False)
    _put(backend, "L-plain", None, None)
    embedder = _Embedder({NEW_TEXT: NEAR, "content L-plain ": NEAR})

    assert _similar(backend, embedder, monkeypatch)["reason"] == "similar_budget"
    assert embedder.encoded == [NEW_TEXT]


@pytest.mark.parametrize(
    ("status", "expected"),
    [(MemoryStatus.ACTIVE, ("merge", "L1")), (MemoryStatus.OBSOLETE, ("store", None))],
)
def test_a_merge_band_match_merges_only_into_an_active_row(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch, status: MemoryStatus, expected: tuple[str, str | None]
) -> None:
    _put(backend, "L1", [0.6, 0.8, 0.0], SPACE_A, status=status)  # cosine 0.6 with NEAR

    result = _similar(backend, _Embedder({NEW_TEXT: NEAR}), monkeypatch, skip=0.95, merge=0.59)

    assert (result["action"], result["existing_id"]) == expected


def test_skip_holds_against_an_obsolete_row(backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    _put(backend, "L1", NEAR, SPACE_A, status=MemoryStatus.OBSOLETE)

    result = _similar(backend, _Embedder({NEW_TEXT: NEAR}), monkeypatch)

    assert (result["action"], result["existing_id"]) == ("skip", "L1")


def test_nothing_close_is_stored(backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    _put(backend, "L1", FAR, SPACE_A)

    result = _similar(backend, _Embedder({NEW_TEXT: NEAR}), monkeypatch)

    assert (result["action"], result["existing_id"], result["mode"]) == ("store", None, "knn")


def test_empty_text_and_out_of_range_thresholds_are_refused_with_a_code(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = _Embedder({})

    assert _similar(backend, embedder, monkeypatch, text="  \n")["code"] == "empty_text"
    assert _similar(backend, embedder, monkeypatch, skip=1.5)["code"] == "bad_thresholds"
    assert embedder.encoded == []


async def test_an_oversized_text_is_refused_before_the_model_is_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """The served tool refuses a text past ``MAX_SIMILAR_TEXT_CHARS`` at the argument bound (daemon/_arg_bounds.py)."""
    from fastmcp import Client

    from trw_memory.server import mcp

    def _no_model(**_kw: object) -> None:
        raise AssertionError("the model must not be resolved for an oversized text")

    monkeypatch.setattr("trw_memory.tools._embedder.get_local_embedder", _no_model)
    text = "x" * (similar.MAX_SIMILAR_TEXT_CHARS + 1)

    async with Client(mcp) as client:
        result = (await client.call_tool("memory_similar", {"namespace": "project:t", "text": text})).data

    assert (result["error"], result["argument"], result["limit"]) == (
        "argument_too_large",
        "text",
        similar.MAX_SIMILAR_TEXT_CHARS,
    )


def test_no_embedder_is_unavailable_not_a_store_verdict(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _similar(backend, None, monkeypatch) == {"status": "unavailable", "reason": "embedder_error"}


def test_a_model_that_is_not_cached_names_the_fetch_command(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _refuse(**_kw: object) -> None:
        raise ModelNotCachedError("model not in the cache")

    monkeypatch.setattr("trw_memory.tools._embedder.get_local_embedder", _refuse)

    result = memory_similar_impl("default", NEW_TEXT, 0.95, 0.85, 10, backend=backend, config=MemoryConfig())

    assert result == {"status": "unavailable", "reason": "model_not_cached", "fix": FETCH_COMMAND}


def test_a_resolved_row_whose_vector_was_pruned_is_still_a_skip(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal status prunes the vector; the empty window goes exhaustive and re-encodes it (6.1.0 parity)."""
    _put(backend, "L-done", NEAR, SPACE_A)
    backend.update("L-done", namespace="default", status=MemoryStatus.RESOLVED.value)
    embedder = _Embedder({NEW_TEXT: NEAR, "content L-done ": NEAR})

    result = _similar(backend, embedder, monkeypatch)

    assert (result["action"], result["existing_id"], result["mode"]) == ("skip", "L-done", "exhaustive")


@pytest.mark.parametrize(
    ("model_name", "similarity", "expected"),
    [
        ("all-MiniLM-L6-v2", 0.87, "merge"),  # the reference scale: 0.87 >= 0.85
        ("BAAI/bge-small-en-v1.5", 0.87, "store"),  # bge's 0.85 equivalent is 0.89
        ("BAAI/bge-small-en-v1.5", 0.93, "merge"),
        ("BAAI/bge-small-en-v1.5", 0.99, "skip"),
    ],
)
def test_the_callers_reference_thresholds_are_calibrated_to_the_loaded_model(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch, model_name: str, similarity: float, expected: str
) -> None:
    """Moved from trw-mcp with the calibration itself (PRD-CORE-302 C1)."""
    import math

    _put(backend, "L1", [similarity, math.sqrt(1 - similarity**2), 0.0], SPACE_A)
    embedder = _Embedder({NEW_TEXT: NEAR})
    embedder.model_name = model_name

    assert _similar(backend, embedder, monkeypatch)["action"] == expected
