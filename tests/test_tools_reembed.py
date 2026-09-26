"""memory_reembed: the daemon runs the SDK re-embed contract on one namespace (PRD-CORE-302 FR07)."""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.embeddings._space_gate import select_space_vectors
from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.reembed import memory_reembed_impl

pytestmark = pytest.mark.integration

OLD = EmbeddingSpace("a" * 64, "test-encoder:old", 3)
ACTIVE = EmbeddingSpace("b" * 64, "test-encoder:active", 3)


class _ActiveEmbedder:
    """Encodes into ACTIVE; a text containing "unencodable" fails, as a blank one does."""

    def available(self) -> bool:
        return True

    def embedding_space(self) -> EmbeddingSpace:
        return ACTIVE

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [None if "unencodable" in text else [0.0, 1.0, 0.0] for text in texts]


@pytest.fixture
def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SQLiteBackend:
    pytest.importorskip("sqlite_vec")
    monkeypatch.setattr("trw_memory.tools._embedder.get_local_embedder", lambda **_kw: _ActiveEmbedder())
    monkeypatch.setattr("trw_memory.tools._embedder.loaded_local_embedder", lambda _key: _ActiveEmbedder())
    store = SQLiteBackend(tmp_path / "memory.db", dim=3)
    for entry_id, content, space in [
        ("clean", "clean summary", OLD),
        ("trailing", "summary with trailing space ", OLD),
        ("newline", "summary\nwith a newline", OLD),
        ("no-proof", "written before provenance", None),
    ]:
        entry = MemoryEntry(id=entry_id, content=content, namespace="default")
        store.store(entry)
        # The stored proof names the text as it was encoded then, stripped: not the row's text now.
        proof = VectorProvenance.for_vector(space, content.strip(), [1.0, 0.0, 0.0]) if space else None
        store.upsert_vector(entry_id, [1.0, 0.0, 0.0], namespace="default", provenance=proof)
    return store


_COUNTS = ("examined", "reembedded", "already_current", "skipped", "warm_examined", "warm_reembedded")


def _reembed(backend: SQLiteBackend) -> dict[str, object]:
    """Every bounded pass, as trw-mcp runs them: the last answer with the passes' counts summed."""
    totals = dict.fromkeys(_COUNTS, 0)
    cursor: str | None = None
    for _ in range(100):
        answer = memory_reembed_impl("default", cursor, backend=backend, config=MemoryConfig())
        if answer["status"] != "ok":
            return answer
        totals = {key: totals[key] + int(answer[key]) for key in _COUNTS}  # type: ignore[call-overload]
        if (cursor := answer["cursor"]) is None:  # type: ignore[assignment]
            return {**answer, **totals}
    raise AssertionError("the passes never finished")


def _admitted(backend: SQLiteBackend, ids: list[str]) -> set[str]:
    return set(select_space_vectors(backend.get_vector_records(ids, namespace="default"), ACTIVE).vectors)


def test_every_row_lands_in_the_active_space_and_a_rerun_is_a_no_op(backend: SQLiteBackend) -> None:
    ids = ["clean", "trailing", "newline", "no-proof"]
    assert _admitted(backend, ids) == set()

    first = _reembed(backend)
    assert (first["status"], first["examined"], first["reembedded"], first["outside_active_space"]) == ("ok", 4, 4, 0)
    assert _admitted(backend, ids) == set(ids)

    second = _reembed(backend)
    assert (second["reembedded"], second["already_current"]) == (0, 4)


def test_a_row_that_cannot_be_encoded_is_counted_as_still_outside(backend: SQLiteBackend) -> None:
    backend.store(MemoryEntry(id="stuck", content="unencodable row", namespace="default"))
    backend.upsert_vector("stuck", [1.0, 0.0, 0.0], namespace="default")

    answer = _reembed(backend)

    assert (answer["skipped"], answer["outside_active_space"]) == (1, 1)
    assert "stuck" not in _admitted(backend, ["stuck"])


def test_no_embedder_answers_unavailable_and_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trw_memory.tools._embedder.get_local_embedder", lambda **_kw: None)
    store = SQLiteBackend(tmp_path / "memory.db", dim=3)

    assert memory_reembed_impl("default", None, backend=store, config=MemoryConfig()) == {
        "status": "unavailable",
        "reason": "embedder_error",
    }


def test_one_call_stops_at_its_budget_and_its_cursor_resumes_the_rest(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rc9 sweep B2: a call never walks a whole namespace on the shared daemon."""
    monkeypatch.setattr("trw_memory._client_reembed.REEMBED_CALL_ROWS", 3)

    first = memory_reembed_impl("default", None, backend=backend, config=MemoryConfig())

    assert (first["status"], first["examined"], first["outside_active_space"]) == ("ok", 3, None)
    assert isinstance(first["cursor"], str)
    rest = memory_reembed_impl("default", first["cursor"], backend=backend, config=MemoryConfig())  # type: ignore[arg-type]
    assert rest["examined"] == 1
    assert _admitted(backend, ["clean", "trailing", "newline", "no-proof"]) == {
        "clean",
        "trailing",
        "newline",
        "no-proof",
    }


@pytest.mark.parametrize("cursor", ["not json", '["rows", "x"]', '["elsewhere", "", ""]', '["rows", 1, 2]'])
def test_a_malformed_cursor_is_invalid_and_writes_nothing(backend: SQLiteBackend, cursor: str) -> None:
    answer = memory_reembed_impl("default", cursor, backend=backend, config=MemoryConfig())

    assert answer["status"] == "invalid"
    assert _admitted(backend, ["clean"]) == set()


def test_reembed_rows_rejects_a_batch_size_above_max(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared implementation enforces the same cap independently of the tool
    boundary, so any other caller (SDK, CLI) cannot bypass it."""
    from trw_memory._client_reembed import MAX_REEMBED_BATCH, reembed_rows

    store = SQLiteBackend(tmp_path / "memory.db", dim=3)

    with pytest.raises(ValueError, match=str(MAX_REEMBED_BATCH)):
        reembed_rows(
            store,
            _ActiveEmbedder(),
            namespace="default",
            config=MemoryConfig(),
            batch_size=MAX_REEMBED_BATCH + 1,
        )


def test_a_row_corrected_while_it_was_being_encoded_keeps_its_own_vector(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C12 (7.0 freeze): rows are encoded outside the write transaction, so a correction can land
    in between. Its text and vector must survive; the stale encoding of the old text must not."""
    corrected = [0.0, 0.0, 1.0]

    class _CorrectedMidEncode(_ActiveEmbedder):
        def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
            if any(text.startswith("clean summary") for text in texts):
                backend.update("clean", namespace="default", content="clean summary, corrected")
                proof = VectorProvenance.for_vector(ACTIVE, "clean summary, corrected", corrected)
                backend.upsert_vector("clean", corrected, namespace="default", provenance=proof)
            return super().embed_batch(texts)

    monkeypatch.setattr("trw_memory.tools._embedder.get_local_embedder", lambda **_kw: _CorrectedMidEncode())
    monkeypatch.setattr("trw_memory.tools._embedder.loaded_local_embedder", lambda _key: _CorrectedMidEncode())

    answer = _reembed(backend)

    assert answer["status"] == "ok", answer
    assert backend.get_stored_embeddings(["clean"], namespace="default")["clean"] == pytest.approx(corrected)
    assert _admitted(backend, ["trailing", "newline", "no-proof"]) == {"trailing", "newline", "no-proof"}


def test_a_newer_vector_written_while_encoding_is_kept_even_when_the_text_is_unchanged(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C12 sol review: the text alone cannot tell a stale encoding from a fresh one."""
    newer = [0.0, 0.0, 1.0]

    class _RevectoredMidEncode(_ActiveEmbedder):
        def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
            if any(text.startswith("clean summary") for text in texts):
                proof = VectorProvenance.for_vector(ACTIVE, "clean summary", newer)
                backend.upsert_vector("clean", newer, namespace="default", provenance=proof)
            return super().embed_batch(texts)

    monkeypatch.setattr("trw_memory.tools._embedder.get_local_embedder", lambda **_kw: _RevectoredMidEncode())
    monkeypatch.setattr("trw_memory.tools._embedder.loaded_local_embedder", lambda _key: _RevectoredMidEncode())

    answer = _reembed(backend)

    assert answer["status"] == "ok", answer
    assert backend.get_stored_embeddings(["clean"], namespace="default")["clean"] == pytest.approx(newer)
