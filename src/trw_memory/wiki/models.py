"""Typed wiki page models for trw-memory.

The wiki module is intentionally pure: it validates page/reference metadata and
provides storage-compatible metadata helpers without touching memory, graph, or
filesystem state.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trw_memory.models.memory import Confidence

__all__ = [
    "WikiPage",
    "WikiPageKind",
    "WikiProvenance",
    "WikiReference",
    "validate_wiki_path",
    "validate_wiki_slug",
]

_SLUG_SEGMENT_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,78}[a-z0-9])?$|^[a-z0-9]$")
_SHELL_METACHARS = frozenset(";|&`$<>\\")
_METADATA_PAYLOAD_KEY = "wiki.page"
_METADATA_SLUG_KEY = "wiki.slug"
_METADATA_KIND_KEY = "wiki.kind"


class WikiPageKind(str, Enum):
    """Supported first-class wiki page kinds."""

    PROJECT = "project"
    TOPIC = "topic"
    ENTITY = "entity"
    ANALYSIS = "analysis"
    SOURCE = "source"


class WikiProvenance(BaseModel):
    """Origin information for a wiki page assertion."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True, strict=True, validate_default=True)

    source: Literal["human", "agent", "tool", "consolidated", "team_sync"] = "agent"
    source_id: str = Field(default="", description="Curator, agent, document, or tool identifier")
    detail: str = Field(default="", description="Optional human-readable provenance detail")

    @field_validator("source_id", "detail")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


class WikiReference(BaseModel):
    """A typed outbound wiki reference."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True, validate_default=True)

    target_slug: str = Field(description="Target wiki page slug")
    ref_type: str = Field(default="related", min_length=1, max_length=64)
    label: str = Field(default="", max_length=160)
    bidirectional: bool = Field(default=True, description="Whether lint should require a backlink")

    @field_validator("target_slug")
    @classmethod
    def _validate_target_slug(cls, value: str) -> str:
        return validate_wiki_slug(value)

    @field_validator("ref_type", "label")
    @classmethod
    def _strip_ref_text(cls, value: str) -> str:
        stripped = value.strip()
        if any(char in stripped for char in _SHELL_METACHARS):
            raise ValueError("wiki reference text must not contain shell metacharacters")
        return stripped


class WikiPage(BaseModel):
    """First-class wiki metadata carried by memory/graph integration layers."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True, populate_by_name=True, validate_default=True)

    kind: WikiPageKind
    slug: str
    title: str = Field(min_length=1, max_length=200)
    provenance: list[WikiProvenance] = Field(default_factory=list)
    confidence: Confidence = Confidence.UNVERIFIED
    evidence: list[str] = Field(default_factory=list)
    outbound_refs: list[WikiReference] = Field(default_factory=list)
    path: str = Field(default="", description="Optional relative wiki source path")

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

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: object) -> Confidence:
        if isinstance(value, Confidence):
            return value
        if isinstance(value, str):
            try:
                return Confidence(value)
            except ValueError as err:
                valid = ", ".join(confidence.value for confidence in Confidence)
                raise ValueError(f"confidence must be one of {valid}") from err
        raise ValueError(f"confidence must be a string or Confidence enum, got {type(value).__name__}")

    @field_validator("slug")
    @classmethod
    def _validate_slug(cls, value: str) -> str:
        return validate_wiki_slug(value)

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("title must not be blank")
        if any(char in stripped for char in _SHELL_METACHARS):
            raise ValueError("title must not contain shell metacharacters")
        return stripped

    @field_validator("evidence")
    @classmethod
    def _validate_evidence(cls, value: list[str]) -> list[str]:
        return [_validate_storage_text(item, field_name="evidence") for item in value]

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        if not value:
            return ""
        return validate_wiki_path(value)

    @model_validator(mode="after")
    def _dedupe_refs(self) -> WikiPage:
        refs_by_key = {(ref.target_slug, ref.ref_type): ref for ref in self.outbound_refs}
        sorted_refs = [refs_by_key[key] for key in sorted(refs_by_key)]
        if sorted_refs != self.outbound_refs:
            self.outbound_refs = sorted_refs
        return self

    def to_memory_metadata(self) -> dict[str, str]:
        """Serialize wiki metadata into ``MemoryEntry.metadata`` compatible strings."""

        payload = self.model_dump(mode="json")
        return {
            _METADATA_PAYLOAD_KEY: json.dumps(payload, sort_keys=True, separators=(",", ":")),
            _METADATA_SLUG_KEY: self.slug,
            _METADATA_KIND_KEY: self.kind.value if isinstance(self.kind, WikiPageKind) else str(self.kind),
        }

    @classmethod
    def from_memory_metadata(cls, metadata: dict[str, str]) -> WikiPage | None:
        """Restore a wiki page from memory metadata, returning ``None`` for non-wiki memories."""

        payload = metadata.get(_METADATA_PAYLOAD_KEY)
        if not payload:
            return None
        return cls.model_validate_json(payload)


def validate_wiki_slug(value: str) -> str:
    """Validate a stable lowercase wiki slug with no traversal or shell syntax."""

    if not isinstance(value, str):
        raise TypeError("wiki slug must be a string")
    slug = value.strip()
    if not slug:
        raise ValueError("wiki slug must not be empty")
    if slug.startswith("/") or slug.endswith("/"):
        raise ValueError("wiki slug must be relative and must not end with /")
    if any(char in slug for char in _SHELL_METACHARS):
        raise ValueError("wiki slug must not contain shell metacharacters")
    segments = slug.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("wiki slug must not contain empty or traversal segments")
    if any(not _SLUG_SEGMENT_RE.fullmatch(segment) for segment in segments):
        raise ValueError("wiki slug segments must be lowercase alphanumeric with single hyphen separators")
    return slug


def validate_wiki_path(value: str) -> str:
    """Validate an optional relative POSIX path without traversal or shell syntax."""

    if not isinstance(value, str):
        raise TypeError("wiki path must be a string")
    path = value.strip()
    if not path:
        raise ValueError("wiki path must not be empty")
    if any(char in path for char in _SHELL_METACHARS):
        raise ValueError("wiki path must not contain shell metacharacters")
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute():
        raise ValueError("wiki path must be relative")
    if any(part in {"", ".", ".."} for part in pure_path.parts):
        raise ValueError("wiki path must not contain traversal segments")
    return pure_path.as_posix()


def _validate_storage_text(value: str, *, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} entries must not be blank")
    if any(char in stripped for char in _SHELL_METACHARS):
        raise ValueError(f"{field_name} entries must not contain shell metacharacters")
    return stripped
