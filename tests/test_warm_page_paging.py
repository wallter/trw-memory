"""``warm_page`` / ``commit_warm_page`` copy only the page's payloads, not the whole sidecar's.

B71-84: ``warm_page`` used to call ``_warm_sidecar_entries_by_id()``, which
parses the sidecar and copies EVERY row's payload dict before sorting ids and
slicing off ``limit`` of them. ``commit_warm_page`` hydrated the whole sidecar
again. Both run under the process-wide ``_TIER_MANAGER_CACHE_LOCK``
(``trw_memory._client_reembed._reembed_warm``), so each re-embed page's lock
hold scaled with the whole warm sidecar.

The fix selects the page's ids off the cached parse (``WarmTierStore._warm_rows``)
before copying any payload, then copies payloads only for those ids
(``WarmTierStore._entries_by_id(rows, only=...)``). These
tests assert that shape directly (payload-copy count bounded by the page,
not the sidecar) and that the observable behaviour -- page contents, resume
cursor, staleness skip, keyword-only mirrors -- is unchanged.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.embeddings.provenance import EmbeddingSpace
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.lifecycle.tiers._warm_space import WarmSnapshot, commit_warm_page, warm_page

SPACE = EmbeddingSpace("a" * 64, "test-encoder:warm-page", 3)


class _FakeEmbedder:
    """Minimal embedder stand-in for ``generation_provenance_kwargs``."""

    def embedding_space(self) -> EmbeddingSpace:
        return SPACE


def _make_store(tmp_path: Path, n: int) -> WarmTierStore:
    store = WarmTierStore(base_dir=tmp_path)
    for i in range(n):
        entry_id = f"id-{i:05d}"
        store.warm_add(entry_id, {"content": f"content {i}", "detail": "d", "tags": []}, embedding=None)
    return store


class TestWarmPageCostsOnlyThePage:
    """Failing-first: payload hydration must not touch the whole sidecar."""

    def test_warm_page_copies_only_the_page_not_the_whole_sidecar(self, tmp_path: Path) -> None:
        n = 500
        limit = 5
        store = _make_store(tmp_path, n)

        copy_calls: list[list] = []
        original = WarmTierStore._entries_by_id

        def counting_entries_by_id(rows, only=None):  # type: ignore[no-untyped-def]
            copied = original(rows, only)
            copy_calls.append(list(copied))  # the payloads actually copied
            return copied

        store._entries_by_id = staticmethod(counting_entries_by_id)  # type: ignore[method-assign]

        page = warm_page(store, SPACE, after=None, limit=limit)

        assert len(page) == limit
        total_rows_hydrated = sum(len(call) for call in copy_calls)
        # The whole sidecar has `n` rows; a fixed correctly must hydrate at
        # most the page (`limit`) rows, never anywhere near `n`.
        assert total_rows_hydrated <= limit, (
            f"warm_page hydrated {total_rows_hydrated} row payloads for a page of "
            f"{limit} out of {n} total rows -- it must hydrate only the page"
        )

    def test_commit_warm_page_copies_only_the_snapshot_ids(self, tmp_path: Path) -> None:
        n = 500
        store = _make_store(tmp_path, n)
        snapshot = [WarmSnapshot(f"id-{i:05d}", f"content {i} d", None) for i in range(5)]
        vectors: list[list[float] | None] = [[0.1, 0.2, 0.3] for _ in snapshot]

        copy_calls: list[list] = []
        original = WarmTierStore._entries_by_id

        def counting_entries_by_id(rows, only=None):  # type: ignore[no-untyped-def]
            copied = original(rows, only)
            copy_calls.append(list(copied))  # the payloads actually copied
            return copied

        store._entries_by_id = staticmethod(counting_entries_by_id)  # type: ignore[method-assign]

        written = commit_warm_page(store, _FakeEmbedder(), SPACE, snapshot, vectors)

        assert written == len(snapshot)
        total_rows_hydrated = sum(len(call) for call in copy_calls)
        assert total_rows_hydrated <= len(snapshot), (
            f"commit_warm_page hydrated {total_rows_hydrated} row payloads for a "
            f"{len(snapshot)}-item commit out of {n} total rows"
        )


class TestWarmPageEquivalence:
    """Same pages, same commits, as the O(whole-sidecar) implementation produced."""

    def test_pages_are_sorted_by_id_and_resume_after_cursor(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, 12)
        page1 = warm_page(store, SPACE, after=None, limit=5)
        ids1 = [s.entry_id for s in page1]
        assert ids1 == sorted(ids1)
        assert ids1 == [f"id-{i:05d}" for i in range(5)]

        page2 = warm_page(store, SPACE, after=ids1[-1], limit=5)
        ids2 = [s.entry_id for s in page2]
        assert ids2 == [f"id-{i:05d}" for i in range(5, 10)]

        page3 = warm_page(store, SPACE, after=ids2[-1], limit=5)
        ids3 = [s.entry_id for s in page3]
        assert ids3 == [f"id-{i:05d}" for i in range(10, 12)]

    def test_page_smaller_than_limit_at_the_tail(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, 3)
        page = warm_page(store, SPACE, after=None, limit=10)
        assert [s.entry_id for s in page] == ["id-00000", "id-00001", "id-00002"]

    def test_snapshot_text_matches_stored_payload(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, 3)
        page = warm_page(store, SPACE, after=None, limit=10)
        for snap in page:
            assert "content" in snap.text

    def test_commit_skips_rows_whose_text_changed_since_snapshot(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, 3)
        page = warm_page(store, SPACE, after=None, limit=10)
        # Mutate one row's content after the snapshot was taken.
        store.warm_add("id-00000", {"content": "changed", "detail": "d", "tags": []}, embedding=None)

        vectors: list[list[float] | None] = [[0.1, 0.2, 0.3] for _ in page]
        written = commit_warm_page(store, _FakeEmbedder(), SPACE, page, vectors)

        # id-00000's text changed, so it must be skipped; the other two commit.
        assert written == 2

    def test_mirror_row_without_vector_stays_keyword_only(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, 1)
        page = warm_page(store, SPACE, after=None, limit=10)
        assert len(page) == 1
        assert page[0].record is None  # no vector was ever stored for this mirror row

    def test_commit_none_vector_is_skipped(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, 2)
        page = warm_page(store, SPACE, after=None, limit=10)
        vectors: list[list[float] | None] = [None, [0.1, 0.2, 0.3]]
        written = commit_warm_page(store, _FakeEmbedder(), SPACE, page, vectors)
        assert written == 1


def test_duplicate_and_malformed_sidecar_ids_page_like_the_old_dict(tmp_path: Path) -> None:
    """sol review: the last row per id wins, and a row with an empty or missing id is never paged."""
    store = _make_store(tmp_path, 3)
    store.warm_add("id-00001", {"content": "rewritten", "detail": "d", "tags": []}, embedding=None)
    sidecar = store._warm_sidecar_path()
    with sidecar.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "", "summary": "no id"}\n{"summary": "missing id"}\n')

    page = warm_page(store, SPACE, after=None, limit=10)

    assert [item.entry_id for item in page] == ["id-00000", "id-00001", "id-00002"]
    assert page[1].text.startswith("rewritten")
