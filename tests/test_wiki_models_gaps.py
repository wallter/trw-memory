"""Wave 13: coverage gap-fill for wiki/models.py (lines 81, 107-110, 120-123, 137, 150, 184, 204, 207, 215, 221, 223)."""
from __future__ import annotations

import pytest

from trw_memory.wiki.models import (
    WikiPage,
    WikiPageKind,
    WikiReference,
    validate_wiki_path,
    validate_wiki_slug,
)


class TestWikiReferenceValidation:
    def test_shell_metachar_in_ref_type_raises_value_error(self) -> None:
        """ref_type with shell metachar → ValueError (line 81)."""
        with pytest.raises(ValueError, match="shell metacharacters"):
            WikiReference(target_slug="topic/foo", ref_type="bad;type")

    def test_shell_metachar_in_label_raises_value_error(self) -> None:
        """label with shell metachar → ValueError (line 81)."""
        with pytest.raises(ValueError, match="shell metacharacters"):
            WikiReference(target_slug="topic/foo", label="bad|label")


class TestWikiPageKindCoercion:
    def test_invalid_kind_string_raises_value_error(self) -> None:
        """WikiPage with invalid kind string → ValueError with valid list (lines 107-110)."""
        with pytest.raises(ValueError, match="kind must be one of"):
            WikiPage(kind="nonexistent", slug="topic/foo", title="Foo")  # type: ignore[arg-type]

    def test_non_string_non_enum_kind_raises_value_error(self) -> None:
        """WikiPage with non-string kind → ValueError (line 110)."""
        with pytest.raises(ValueError, match="kind must be a string or WikiPageKind enum"):
            WikiPage(kind=99, slug="topic/foo", title="Foo")  # type: ignore[arg-type]


class TestWikiPageConfidenceCoercion:
    def test_invalid_confidence_string_raises_value_error(self) -> None:
        """WikiPage with invalid confidence string → ValueError with valid list (lines 120-123)."""
        with pytest.raises(ValueError, match="confidence must be one of"):
            WikiPage(
                kind=WikiPageKind.TOPIC,
                slug="topic/foo",
                title="Foo",
                confidence="super-high",  # type: ignore[arg-type]
            )

    def test_non_string_confidence_raises_value_error(self) -> None:
        """WikiPage with non-string confidence → ValueError (line 123)."""
        with pytest.raises(ValueError, match="confidence must be a string or Confidence enum"):
            WikiPage(
                kind=WikiPageKind.TOPIC,
                slug="topic/foo",
                title="Foo",
                confidence=99,  # type: ignore[arg-type]
            )


class TestWikiPageTitleValidation:
    def test_shell_metachar_in_title_raises_value_error(self) -> None:
        """Title with shell metachar → ValueError (line 137)."""
        with pytest.raises(ValueError, match="shell metacharacters"):
            WikiPage(kind=WikiPageKind.TOPIC, slug="topic/foo", title="Bad|Title")


class TestWikiPagePathValidation:
    def test_non_empty_valid_path_is_accepted(self) -> None:
        """Non-empty valid path passes through validate_wiki_path (line 150)."""
        page = WikiPage(kind=WikiPageKind.TOPIC, slug="topic/foo", title="Foo", path="docs/wiki/foo.md")
        assert page.path == "docs/wiki/foo.md"

    def test_empty_path_defaults_to_empty_string(self) -> None:
        """Empty path string returns '' without calling validate_wiki_path."""
        page = WikiPage(kind=WikiPageKind.TOPIC, slug="topic/foo", title="Foo", path="")
        assert page.path == ""


class TestWikiPageEvidenceValidation:
    def test_blank_evidence_entry_raises_value_error(self) -> None:
        """Blank evidence string → ValueError (line 221)."""
        with pytest.raises(ValueError, match="must not be blank"):
            WikiPage(
                kind=WikiPageKind.TOPIC,
                slug="topic/foo",
                title="Foo",
                evidence=["   "],
            )

    def test_shell_metachar_in_evidence_raises_value_error(self) -> None:
        """Shell metachar in evidence entry → ValueError (line 223)."""
        with pytest.raises(ValueError, match="shell metacharacters"):
            WikiPage(
                kind=WikiPageKind.TOPIC,
                slug="topic/foo",
                title="Foo",
                evidence=["valid evidence", "bad;evidence"],
            )


class TestValidateWikiSlug:
    def test_non_string_input_raises_type_error(self) -> None:
        """Non-string slug → TypeError (line 184)."""
        with pytest.raises(TypeError, match="wiki slug must be a string"):
            validate_wiki_slug(42)  # type: ignore[arg-type]


class TestValidateWikiPath:
    def test_non_string_input_raises_type_error(self) -> None:
        """Non-string path → TypeError (line 204)."""
        with pytest.raises(TypeError, match="wiki path must be a string"):
            validate_wiki_path(42)  # type: ignore[arg-type]

    def test_empty_string_raises_value_error(self) -> None:
        """Empty path after strip → ValueError (line 207)."""
        with pytest.raises(ValueError, match="wiki path must not be empty"):
            validate_wiki_path("   ")

    def test_valid_relative_path_returns_posix(self) -> None:
        """Valid relative path returns pure POSIX form (line 215)."""
        result = validate_wiki_path("docs/wiki/foo.md")
        assert result == "docs/wiki/foo.md"

    def test_leading_double_dot_in_path_raises_value_error(self) -> None:
        """Path starting with ../  → ValueError for traversal (existing ValidatePath behaviour)."""
        with pytest.raises(ValueError, match="traversal|relative"):
            validate_wiki_path("../secret.md")
