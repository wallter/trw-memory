"""PRD-SEC-023 FR07 (daemon-side rules): a row's label survives the operations that rewrite rows.

Slice 2 covers the two that rewrite a stored row: a dedup merge keeps the maximum stamp of its inputs, and a correction can only raise it.
Slice 3 adds the two that derive data from rows: cross-project validation applies a match in neither direction when either row is above
``team``, and consolidation never clusters such a row (so no consolidated row or LLM summary is built from it).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory import _graph_sibling_index, graph
from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.labels import LabelPolicy, Level
from trw_memory.lifecycle.consolidation import find_clusters
from trw_memory.lifecycle.correction import LearningPatch, _collect
from trw_memory.lifecycle.dedup import merge_entries
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import StorageBackend

from ._test_consolidation_support import _V1, _V2, _V3, _InMemoryBackend, _make_embedder


@pytest.fixture(autouse=True)
def _user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "user-base"
    base.mkdir()
    monkeypatch.setenv("TRW_USER_DIR", str(base))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)


def _row(entry_id: str, **metadata: str) -> MemoryEntry:
    return MemoryEntry(id=entry_id, content=f"learning {entry_id}", namespace="default", metadata=dict(metadata))


def test_a_merge_keeps_the_maximum_stamp_of_its_inputs() -> None:
    existing, incoming = _row("a", trw_label="personal"), _row("b", trw_label="sensitive")

    merged = merge_entries(existing, incoming)

    assert merged.metadata["trw_label"] == "sensitive"
    assert LabelPolicy.current().label_of(merged) is Level.SENSITIVE


def test_a_stamped_incoming_row_cannot_be_laundered_by_merging_into_an_unstamped_one() -> None:
    merged = merge_entries(_row("a"), _row("b", trw_label="personal"))

    assert merged.metadata["trw_label"] == "personal"


def test_a_merge_of_unstamped_rows_leaves_metadata_exactly_as_it_was() -> None:
    existing = _row("a", source="agent")

    merged = merge_entries(existing, _row("b", other="x"))

    assert merged.metadata == {"source": "agent"}, "NFR01: no stamp on either side, no change"


def test_a_correction_cannot_lower_a_stamp() -> None:
    entry = _row("a", trw_label="personal")

    fields, changes = _collect(entry, LearningPatch(metadata_add={"trw_label": "team"}))

    assert "metadata" not in fields, "the stamp stays personal, so the stored metadata does not change at all"
    assert changes == ["metadata updated"] or changes == []


def test_a_correction_can_raise_a_stamp_and_an_unknown_value_raises_to_sensitive() -> None:
    entry = _row("a", trw_label="personal")

    raised, _ = _collect(entry, LearningPatch(metadata_add={"trw_label": "sensitive"}))
    junk, _ = _collect(entry, LearningPatch(metadata_add={"trw_label": "nonsense"}))

    assert raised["metadata"]["trw_label"] == "sensitive"
    assert junk["metadata"]["trw_label"] == "sensitive"


def test_a_correction_that_touches_no_stamp_changes_no_stamp() -> None:
    entry = _row("a", trw_label="personal")

    fields, _ = _collect(entry, LearningPatch(metadata_add={"note": "x"}))

    assert fields["metadata"] == {"trw_label": "personal", "note": "x"}


# ── slice 3: cross-project validation and consolidation ─────────────────────

_DIM = 8
_SPACE = EmbeddingSpace("c" * 64, "test-encoder:labels-derived", _DIM)
_SAME = [0.0] * (_DIM - 1) + [1.0]


def _put(backend: StorageBackend, entry_id: str, namespace: str, **metadata: str) -> MemoryEntry:
    entry = MemoryEntry(id=entry_id, content=f"content {entry_id}", namespace=namespace, metadata=dict(metadata))
    backend.store(entry)
    backend.upsert_vector(
        entry_id, _SAME, namespace=namespace, provenance=VectorProvenance.for_vector(_SPACE, entry.content, _SAME)
    )
    return entry


def _cross_validate(
    tmp_path: Path, *, written: dict[str, str], sibling: dict[str, str]
) -> tuple[int, MemoryEntry, MemoryEntry]:
    """Write an identical row into project:alpha and project:beta, cross-validate the alpha one; (matches, alpha row, beta row) after."""
    pytest.importorskip("sqlite_vec")
    _graph_sibling_index.SIBLING_CACHE.clear()
    cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path / "store"), embedding_dim=_DIM)
    with create_backend_from_config(cfg, "project:beta") as remote:
        _put(remote, "M-beta", "project:beta", **sibling)
    with create_backend_from_config(cfg, "project:alpha") as backend:
        entry = _put(backend, "M-alpha", "project:alpha", **written)
        matched = graph.cross_validate_entries([(entry, _SAME, _SPACE)], backend, config=cfg)["M-alpha"]
        alpha = backend.get("M-alpha", namespace="project:alpha")
    with create_backend_from_config(cfg, "project:beta") as remote:
        beta = remote.get("M-beta", namespace="project:beta")
    _graph_sibling_index.SIBLING_CACHE.clear()
    assert alpha is not None and beta is not None
    return matched, alpha, beta


def test_cross_validation_of_two_team_rows_still_applies_in_both_directions(tmp_path: Path) -> None:
    """The control: without a label, identical rows in two projects validate each other (so the cases below test the label, not a miss)."""
    matched, alpha, beta = _cross_validate(tmp_path, written={}, sibling={})

    assert matched == 1
    assert alpha.outcome_history and beta.outcome_history


def test_a_team_row_identical_to_a_personal_sibling_applies_no_match_in_either_direction(tmp_path: Path) -> None:
    matched, alpha, beta = _cross_validate(tmp_path, written={}, sibling={"trw_label": "personal"})

    assert matched == 0
    assert alpha.outcome_history == [] and beta.outcome_history == []
    assert alpha.importance == 0.5 and beta.importance == 0.5, "no importance boost on either row"


def test_a_personal_written_row_identical_to_a_team_sibling_applies_no_match_in_either_direction(
    tmp_path: Path,
) -> None:
    matched, alpha, beta = _cross_validate(tmp_path, written={"trw_label": "personal"}, sibling={})

    assert matched == 0
    assert alpha.outcome_history == [] and beta.outcome_history == []
    assert alpha.importance == 0.5 and beta.importance == 0.5


def _cluster_rows(**metadata: str) -> _InMemoryBackend:
    storage = _InMemoryBackend()
    for i in range(3):
        storage.store(
            MemoryEntry(id=f"e{i}", content=f"near duplicate {i}", namespace="default", metadata=dict(metadata))
        )
    return storage


def test_three_near_duplicate_team_rows_cluster() -> None:
    """The control for the case below."""
    clusters = find_clusters(_cluster_rows(), _make_embedder(vectors=[_V1, _V2, _V3]), similarity_threshold=0.5)

    assert [len(c) for c in clusters] == [3]


def test_three_near_duplicate_personal_rows_are_not_consolidated_or_even_embedded() -> None:
    embedder = _make_embedder(vectors=[_V1, _V2, _V3])

    clusters = find_clusters(_cluster_rows(trw_label="personal"), embedder, similarity_threshold=0.5)

    assert clusters == []
    embedder.embed_batch.assert_not_called()


def test_a_personal_row_is_left_out_of_a_cluster_of_team_rows() -> None:
    storage = _InMemoryBackend()
    storage.store(MemoryEntry(id="p", content="near duplicate p", namespace="user:alice"))
    for entry in _cluster_rows().list_entries():
        storage.store(entry)
    embedder = _make_embedder()
    embedder.embed_batch.side_effect = lambda texts: [_V1] * len(texts)

    clusters = find_clusters(storage, embedder, similarity_threshold=0.5)

    assert [sorted(e.id for e in c) for c in clusters] == [["e0", "e1", "e2"]]
    assert not any("near duplicate p" in text for text in embedder.embed_batch.call_args.args[0]), (
        "its text never reaches the embedder"
    )
