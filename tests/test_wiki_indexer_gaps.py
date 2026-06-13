"""Wave 13: coverage gap-fill for wiki/indexer.py."""
from __future__ import annotations

import pytest

from trw_memory.wiki.indexer import WikiIndexCandidate, propose_wiki_pages
from trw_memory.wiki.models import WikiPageKind


class TestWikiIndexCandidateValidation:
    def test_valid_kind_string_is_accepted(self) -> None:
        """Passing a valid kind string hits the success branch (line 44)."""
        candidate = WikiIndexCandidate(title="My Topic", kind="topic")  # type: ignore[arg-type]
        assert candidate.kind == WikiPageKind.TOPIC

    def test_invalid_kind_string_raises_value_error(self) -> None:
        """Invalid kind string → ValueError listing valid kinds (lines 47-50)."""
        with pytest.raises(ValueError, match="kind must be one of"):
            WikiIndexCandidate(title="My Topic", kind="unknown_kind")  # type: ignore[arg-type]

    def test_non_string_kind_raises_value_error(self) -> None:
        """Non-string, non-WikiPageKind kind → ValueError (line 50)."""
        with pytest.raises(ValueError, match="kind must be a string or WikiPageKind enum"):
            WikiIndexCandidate(title="My Topic", kind=42)  # type: ignore[arg-type]

    def test_blank_title_raises_value_error(self) -> None:
        """Title that is only whitespace → ValueError blank text (line 57)."""
        with pytest.raises(ValueError, match="must not be blank"):
            WikiIndexCandidate(title="   ")

    def test_non_empty_source_slug_is_validated(self) -> None:
        """Non-empty source_slug passes through validate_wiki_slug (line 64)."""
        candidate = WikiIndexCandidate(title="My Topic", source_slug="topic/parent")
        assert candidate.source_slug == "topic/parent"

    def test_invalid_source_slug_raises_value_error(self) -> None:
        """Invalid source_slug → validate_wiki_slug raises ValueError."""
        with pytest.raises(ValueError):
            WikiIndexCandidate(title="My Topic", source_slug="INVALID SLUG!")


class TestProposeWikiPages:
    def _make_candidate(self, title: str, kind: str = "topic") -> WikiIndexCandidate:
        return WikiIndexCandidate(title=title, kind=kind)  # type: ignore[arg-type]

    def test_max_proposals_break_stops_early(self) -> None:
        """When proposals reach max_proposals, loop breaks (line 115)."""
        candidates = [self._make_candidate(f"Topic {i}") for i in range(5)]
        result = propose_wiki_pages(candidates, enabled=True, dry_run=False, max_proposals=2)
        assert len(result.proposals) == 2

    def test_duplicate_slug_is_skipped(self) -> None:
        """Two candidates normalizing to the same slug → second is skipped (line 118)."""
        c1 = WikiIndexCandidate(title="My Topic")
        c2 = WikiIndexCandidate(title="MY TOPIC")  # normalizes to same slug
        result = propose_wiki_pages([c1, c2], enabled=True, dry_run=False)
        slugs = [p.slug for p in result.proposals]
        assert len(slugs) == len(set(slugs))  # no duplicates
        assert len(result.proposals) == 1

    def test_existing_slug_excluded_from_proposals(self) -> None:
        """Slug matching existing_slugs → skipped (line 118 via existing check)."""
        candidate = self._make_candidate("My Topic")
        result = propose_wiki_pages(
            [candidate], enabled=True, dry_run=False, existing_slugs=["topic/my-topic"]
        )
        assert result.proposals == []

    def test_title_with_all_non_slug_chars_becomes_untitled(self) -> None:
        """Title normalizing to empty string → 'untitled' fallback (line 143)."""
        candidate = WikiIndexCandidate(title="---")
        result = propose_wiki_pages([candidate], enabled=True, dry_run=False)
        assert any("untitled" in p.slug for p in result.proposals)

    def test_source_slug_creates_outbound_ref(self) -> None:
        """Candidate with source_slug produces an outbound WikiReference."""
        candidate = WikiIndexCandidate(title="My Topic", source_slug="topic/parent")
        result = propose_wiki_pages([candidate], enabled=True, dry_run=False)
        assert len(result.proposals) == 1
        assert any(ref.target_slug == "topic/parent" for ref in result.proposals[0].outbound_refs)

    def test_disabled_returns_no_proposals(self) -> None:
        """enabled=False returns WikiIndexResult with empty proposals."""
        result = propose_wiki_pages([self._make_candidate("Foo")], enabled=False, dry_run=True)
        assert result.enabled is False
        assert result.proposals == []
