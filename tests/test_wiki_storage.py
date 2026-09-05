"""SQLite persistence tests for wiki page references."""

from __future__ import annotations

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.wiki.models import WikiPage, WikiReference


def _wiki_entry(entry_id: str, page: WikiPage, *, namespace: str = "default") -> MemoryEntry:
    return MemoryEntry(id=entry_id, content=page.title, namespace=namespace, metadata=page.to_memory_metadata())


def test_wiki_page_storage_round_trips_without_changing_non_wiki_memories(
    sqlite_memory_backend: SQLiteBackend,
) -> None:
    page = WikiPage(
        kind="topic",
        slug="topic/wiki-storage",
        title="Wiki Storage",
        confidence="medium",
        outbound_refs=[WikiReference(target_slug="topic/wiki-lint", ref_type="related")],
    )
    wiki_entry = _wiki_entry("M-wiki", page)
    plain_entry = MemoryEntry(id="M-plain", content="plain memory")

    sqlite_memory_backend.store(wiki_entry)
    sqlite_memory_backend.store(plain_entry)

    restored_wiki = sqlite_memory_backend.get("M-wiki", namespace="default")
    restored_plain = sqlite_memory_backend.get("M-plain", namespace="default")

    assert restored_wiki is not None
    assert restored_plain is not None
    assert WikiPage.from_memory_metadata(restored_wiki.metadata) == page
    assert WikiPage.from_memory_metadata(restored_plain.metadata) is None
    assert restored_plain.metadata == {}


def test_wiki_refs_persist_and_query_outbound_and_inbound_deterministically(
    sqlite_memory_backend: SQLiteBackend,
) -> None:
    source_page = WikiPage(
        kind="topic",
        slug="topic/source",
        title="Source",
        outbound_refs=[
            WikiReference(target_slug="topic/target", ref_type="supports", label="supports"),
            WikiReference(target_slug="topic/other", ref_type="related", label="related"),
        ],
    )
    target_page = WikiPage(kind="topic", slug="topic/target", title="Target")

    sqlite_memory_backend.store(_wiki_entry("M-source", source_page))
    sqlite_memory_backend.store(_wiki_entry("M-target", target_page))

    outbound = sqlite_memory_backend.query_wiki_outbound_refs("topic/source")
    inbound = sqlite_memory_backend.query_wiki_inbound_refs("topic/target")

    assert [(ref.target_slug, ref.ref_type) for ref in outbound] == [
        ("topic/other", "related"),
        ("topic/target", "supports"),
    ]
    assert [(ref.source_slug, ref.target_slug, ref.ref_type, ref.label) for ref in inbound] == [
        ("topic/source", "topic/target", "supports", "supports")
    ]


def test_wiki_ref_persistence_replaces_stale_refs_on_update(
    sqlite_memory_backend: SQLiteBackend,
) -> None:
    page = WikiPage(
        kind="topic",
        slug="topic/source",
        title="Source",
        outbound_refs=[WikiReference(target_slug="topic/old", ref_type="related")],
    )
    sqlite_memory_backend.store(_wiki_entry("M-source", page))

    updated_page = WikiPage(
        kind="topic",
        slug="topic/source",
        title="Source",
        outbound_refs=[WikiReference(target_slug="topic/new", ref_type="related")],
    )
    sqlite_memory_backend.update("M-source", metadata=updated_page.to_memory_metadata(), namespace="default")

    outbound = sqlite_memory_backend.query_wiki_outbound_refs("topic/source")

    assert [(ref.target_slug, ref.ref_type) for ref in outbound] == [("topic/new", "related")]


def test_wiki_ref_persistence_deletes_refs_for_removed_entries(
    sqlite_memory_backend: SQLiteBackend,
) -> None:
    page = WikiPage(
        kind="topic",
        slug="topic/source",
        title="Source",
        outbound_refs=[WikiReference(target_slug="topic/target", ref_type="related")],
    )
    sqlite_memory_backend.store(_wiki_entry("M-source", page))

    assert sqlite_memory_backend.delete("M-source", namespace="default") is True

    assert sqlite_memory_backend.query_wiki_outbound_refs("topic/source") == []
    assert sqlite_memory_backend.query_wiki_inbound_refs("topic/target") == []
