"""Tests for wiki page/ref typed models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from trw_memory.models.memory import MemoryEntry
from trw_memory.wiki.models import (
    WikiPage,
    WikiPageKind,
    WikiProvenance,
    WikiReference,
    validate_wiki_path,
    validate_wiki_slug,
)


def test_wiki_page_round_trips_all_prd_fields_via_json() -> None:
    page = WikiPage(
        kind=WikiPageKind.TOPIC,
        slug="memory/wiki-lint",
        title="Memory Wiki Lint",
        provenance=[WikiProvenance(source="human", source_id="curator")],
        confidence="high",
        evidence=["tests/test_wiki_models.py::test_wiki_page_round_trips_all_prd_fields_via_json"],
        outbound_refs=[WikiReference(target_slug="sources/prd-core-169", ref_type="source", bidirectional=False)],
    )

    restored = WikiPage.model_validate_json(page.model_dump_json())

    assert restored.kind == "topic"
    assert restored.slug == "memory/wiki-lint"
    assert restored.title == "Memory Wiki Lint"
    assert restored.provenance[0].source == "human"
    assert restored.confidence == "high"
    assert restored.evidence == ["tests/test_wiki_models.py::test_wiki_page_round_trips_all_prd_fields_via_json"]
    assert restored.outbound_refs[0].target_slug == "sources/prd-core-169"


def test_wiki_metadata_helpers_preserve_non_wiki_memory_behavior() -> None:
    non_wiki = MemoryEntry(id="M-plain", content="ordinary memory")
    assert WikiPage.from_memory_metadata(non_wiki.metadata) is None

    page = WikiPage(
        kind="analysis",
        slug="analysis/wiki-indexing",
        title="Wiki Indexing Analysis",
        provenance=[WikiProvenance(source="agent", source_id="sprint-125")],
        confidence="medium",
        evidence=["PRD-CORE-169"],
        outbound_refs=[WikiReference(target_slug="sources/prd-core-169", ref_type="source", bidirectional=False)],
    )
    entry = MemoryEntry(id="M-wiki", content=page.title, metadata=page.to_memory_metadata())

    restored = WikiPage.from_memory_metadata(entry.metadata)

    assert restored == page
    assert non_wiki.metadata == {}


@pytest.mark.parametrize(
    "slug",
    ["Project/Upper", "../escape", "topic//double", "topic/$shell", "/absolute", "topic/with space", ""],
)
def test_wiki_slug_rejects_traversal_shell_and_invalid_segments(slug: str) -> None:
    with pytest.raises(ValueError):
        validate_wiki_slug(slug)


def test_wiki_slug_accepts_stable_lowercase_path_segments() -> None:
    assert validate_wiki_slug("project/trw-memory") == "project/trw-memory"


@pytest.mark.parametrize("path", ["../outside.md", "/absolute.md", "docs/../secret.md", "docs/wiki;rm.md"])
def test_wiki_path_rejects_traversal_absolute_and_shell_like_paths(path: str) -> None:
    with pytest.raises(ValueError):
        validate_wiki_path(path)


def test_reference_deduplication_is_deterministic() -> None:
    page = WikiPage(
        kind="topic",
        slug="topic/references",
        title="References",
        outbound_refs=[
            WikiReference(target_slug="topic/b", ref_type="related"),
            WikiReference(target_slug="topic/a", ref_type="related"),
            WikiReference(target_slug="topic/b", ref_type="related"),
        ],
    )

    assert [ref.target_slug for ref in page.outbound_refs] == ["topic/a", "topic/b"]


def test_wiki_page_rejects_blank_title() -> None:
    with pytest.raises(ValidationError):
        WikiPage(kind="topic", slug="topic/blank", title="   ")


def test_wiki_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WikiPage(kind="topic", slug="topic/strict", title="Strict", unexpected="ignored-no-more")

    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


def test_memory_metadata_payload_is_json_string_for_storage_compatibility() -> None:
    page = WikiPage(kind="project", slug="project/trw", title="TRW")

    metadata = page.to_memory_metadata()
    payload = json.loads(metadata["wiki.page"])

    assert metadata["wiki.slug"] == "project/trw"
    assert metadata["wiki.kind"] == "project"
    assert payload["title"] == "TRW"
