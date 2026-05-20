"""Pure proactive wiki indexing proposal helpers.

These helpers never write to storage. Integration layers may consume proposals
and decide whether to persist them, but disabled and dry-run modes always return
``mutated=False``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from trw_memory.wiki.models import WikiPageKind, WikiReference, validate_wiki_slug

__all__ = [
    "WikiIndexCandidate",
    "WikiIndexProposal",
    "WikiIndexResult",
    "propose_wiki_pages",
]

_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")
_DUPLICATE_DASHES = re.compile(r"-+")


class WikiIndexCandidate(BaseModel):
    """Input candidate for explicit wiki indexing proposal generation."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True, validate_default=True)

    title: str = Field(min_length=1, max_length=200)
    kind: WikiPageKind = WikiPageKind.TOPIC
    source_slug: str = Field(default="")
    evidence: list[str] = Field(default_factory=list)
    reason: str = Field(default="candidate selected for wiki indexing")

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value: object) -> WikiPageKind:
        if isinstance(value, WikiPageKind):
            return value
        if isinstance(value, str):
            try:
                return WikiPageKind(value)
            except ValueError as err:
                valid = ", ".join(kind.value for kind in WikiPageKind)
                raise ValueError(f"kind must be one of {valid}") from err
        raise ValueError(f"kind must be a string or WikiPageKind enum, got {type(value).__name__}")

    @field_validator("title", "reason")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("candidate text must not be blank")
        return stripped

    @field_validator("source_slug")
    @classmethod
    def _validate_source_slug(cls, value: str) -> str:
        if not value:
            return ""
        return validate_wiki_slug(value)


class WikiIndexProposal(BaseModel):
    """A proposed wiki page write returned to callers explicitly."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True, validate_default=True)

    slug: str
    title: str
    kind: WikiPageKind
    reason: str
    evidence: list[str] = Field(default_factory=list)
    outbound_refs: list[WikiReference] = Field(default_factory=list)

    @field_validator("slug")
    @classmethod
    def _validate_slug(cls, value: str) -> str:
        return validate_wiki_slug(value)


class WikiIndexResult(BaseModel):
    """Result of pure proposal generation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool
    dry_run: bool
    mutated: bool = False
    proposals: list[WikiIndexProposal] = Field(default_factory=list)


def propose_wiki_pages(
    candidates: Sequence[WikiIndexCandidate],
    *,
    enabled: bool = False,
    dry_run: bool = True,
    existing_slugs: Iterable[str] | None = None,
    max_proposals: int = 20,
) -> WikiIndexResult:
    """Return deterministic page proposals without mutating storage."""

    if not enabled:
        return WikiIndexResult(enabled=False, dry_run=dry_run, mutated=False, proposals=[])

    existing = {validate_wiki_slug(slug) for slug in (existing_slugs or [])}
    proposals: list[WikiIndexProposal] = []
    proposed_slugs: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: (str(item.kind), item.title.lower())):
        if len(proposals) >= max(0, max_proposals):
            break
        slug = _candidate_slug(candidate)
        if slug in existing or slug in proposed_slugs:
            continue
        outbound_refs = []
        if candidate.source_slug:
            outbound_refs.append(
                WikiReference(target_slug=candidate.source_slug, ref_type="source", bidirectional=False)
            )
        proposals.append(
            WikiIndexProposal(
                slug=slug,
                title=candidate.title,
                kind=candidate.kind,
                reason=candidate.reason,
                evidence=candidate.evidence,
                outbound_refs=outbound_refs,
            )
        )
        proposed_slugs.add(slug)
    return WikiIndexResult(enabled=True, dry_run=dry_run, mutated=False, proposals=proposals)


def _candidate_slug(candidate: WikiIndexCandidate) -> str:
    prefix = candidate.kind.value if isinstance(candidate.kind, WikiPageKind) else str(candidate.kind)
    normalized = _NON_SLUG_CHARS.sub("-", candidate.title.strip().lower())
    normalized = _DUPLICATE_DASHES.sub("-", normalized).strip("-")
    if not normalized:
        normalized = "untitled"
    return validate_wiki_slug(f"{prefix}/{normalized[:80].strip('-')}")
