"""PRD-CORE-195 NFR05 — disabled-arm parity: HyPE off == pre-HyPE behaviour.

Regression pin: with hype_enabled=False, the store path writes zero sibling
vectors and the recall pipeline runs no collapse pass, so recall id-ordering is
byte-identical to a run where the collapse code does not exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from trw_memory.client import MemoryClient
from trw_memory.retrieval.pipeline import hybrid_search
from tests.conftest import make_entry
from tests.test_hype_store import _FakeEmbedder, _ListGenerator


async def test_disabled_store_writes_no_siblings(tmp_path: Path) -> None:
    # Even with a question generator wired, hype_enabled=False stores nothing.
    client = MemoryClient(
        "default",
        mode="local",
        db_path=tmp_path / "compat.db",
        question_generator=_ListGenerator(["a perfectly fine question string"]),
    )
    client._config.hype_enabled = False
    client._get_embedder = lambda: _FakeEmbedder()  # type: ignore[method-assign]
    try:
        await client.store("Pydantic strict mode content", entry_id="C1")
        backend = client._get_backend()
        # No #hype rows at all.
        assert backend.hype_sibling_ids("C1") == []
        assert all("#hype" not in vid for vid in backend.existing_vector_ids())
    finally:
        await client.close()


def test_hybrid_search_collapse_off_matches_default() -> None:
    # The collapse_hype=False arm (default) must equal a call that never passes
    # the flag — bit-for-bit id ordering on the same inputs.
    entries = [
        make_entry(entry_id="a", content="alpha beta gamma"),
        make_entry(entry_id="b", content="beta gamma delta"),
        make_entry(entry_id="c", content="gamma delta epsilon"),
    ]
    baseline = hybrid_search("beta gamma", entries, top_k=10)
    with_flag_off = hybrid_search("beta gamma", entries, top_k=10, collapse_hype=False)
    assert [e.id for e in baseline] == [e.id for e in with_flag_off]


def test_hybrid_search_collapse_off_ignores_sibling_embeddings() -> None:
    # When collapse_hype is False, even if stored_embeddings carries #hype keys
    # they are never injected into the dense pool → no synthetic id can surface.
    entries = [make_entry(entry_id="a", content="alpha beta")]
    stored = {"a": [0.1] * 384, "a#hype0": [0.1] * 384}
    result = hybrid_search(
        "alpha",
        entries,
        stored_embeddings=stored,
        collapse_hype=False,
        top_k=10,
    )
    assert all("#hype" not in e.id for e in result)
