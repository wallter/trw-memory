"""Generation identity for stored vectors; no model loads or retrospective stamps.

This records a producer's measured identity, not an attestation against arbitrary
code with database write access. Consumers must compare it with their own input
and provider descriptor before treating a stored vector as compatible evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Sequence
from dataclasses import asdict, dataclass

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def vector_digest(embedding: Sequence[float]) -> str:
    """Hash the exact float32 storage representation, rejecting invalid vectors."""
    if not embedding or not all(math.isfinite(value) for value in embedding):
        raise ValueError("embedding must be nonempty and finite")
    try:
        packed = struct.pack(f"{len(embedding)}f", *embedding)
    except (OverflowError, struct.error) as exc:
        raise ValueError("embedding is not representable as float32") from exc
    if not all(math.isfinite(value) for value in struct.unpack(f"{len(embedding)}f", packed)):
        raise ValueError("embedding is not finite in float32")
    return hashlib.sha256(packed).hexdigest()


def input_digest(text: str) -> str:
    """Bind the exact string passed to encoding, without lossy normalization."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingSpace:
    """Measured encoder state plus encoding/preprocessing contract and output shape."""

    artifact_sha256: str
    encoding: str
    dimensions: int

    def __post_init__(self) -> None:
        if not _digest(self.artifact_sha256):
            raise ValueError("artifact_sha256 must identify immutable artifact bytes")
        if not isinstance(self.encoding, str) or not self.encoding.strip():
            raise ValueError("encoding contract is required")
        if type(self.dimensions) is not int or self.dimensions < 1:
            raise ValueError("dimensions must be a positive integer")


@dataclass(frozen=True)
class VectorProvenance:
    """Versioned producer record binding space, exact input role/text and bytes."""

    space: EmbeddingSpace
    input_sha256: str
    vector_sha256: str
    input_role: str = "document"
    version: int = 1
    parent_input_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported vector provenance version")
        if not isinstance(self.space, EmbeddingSpace):
            raise TypeError("embedding space is required")
        if not _digest(self.input_sha256) or not _digest(self.vector_sha256):
            raise ValueError("input and vector SHA256 identities are required")
        if self.parent_input_sha256 is not None and not _digest(self.parent_input_sha256):
            raise ValueError("parent input identity must be SHA256")
        if self.input_role == "generated-question" and self.parent_input_sha256 is None:
            raise ValueError("generated-question evidence requires parent input identity")
        if not isinstance(self.input_role, str) or not self.input_role.strip():
            raise ValueError("input role is required")

    @classmethod
    def for_vector(
        cls,
        space: EmbeddingSpace,
        text: str,
        embedding: Sequence[float],
        *,
        input_role: str = "document",
        parent_text: str | None = None,
    ) -> VectorProvenance:
        if len(embedding) != space.dimensions:
            raise ValueError("vector dimension differs from embedding space")
        return cls(
            space,
            input_digest(text),
            vector_digest(embedding),
            input_role,
            parent_input_sha256=input_digest(parent_text) if parent_text is not None else None,
        )

    def matches_vector(self, embedding: Sequence[float]) -> bool:
        try:
            return len(embedding) == self.space.dimensions and vector_digest(embedding) == self.vector_sha256
        except (ValueError, TypeError):
            # trw-fail-silent-allow: a vector that cannot be digested does not MATCH this provenance; False is the correct answer to the question asked, and it fails closed toward unqualified
            return False

    def matches(
        self,
        space: EmbeddingSpace,
        text: str,
        embedding: Sequence[float],
        *,
        input_role: str = "document",
        parent_text: str | None = None,
    ) -> bool:
        return (
            self.space == space
            and self.input_role == input_role
            and self.input_sha256 == input_digest(text)
            and self.parent_input_sha256 == (input_digest(parent_text) if parent_text is not None else None)
            and self.matches_vector(embedding)
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: object) -> VectorProvenance | None:
        """Unknown/malformed records are not repaired or inferred from config."""
        if not isinstance(raw, str):
            return None
        try:
            value = json.loads(raw)
            required = {"space", "input_sha256", "vector_sha256", "input_role", "version"}
            if not isinstance(value, dict) or not required <= set(value) <= required | {"parent_input_sha256"}:
                return None
            space = value.pop("space")
            if not isinstance(space, dict):
                return None
            return cls(space=EmbeddingSpace(**space), **value)
        except (TypeError, ValueError, RecursionError):
            # trw-fail-silent-allow: None IS the typed unqualified signal here -- the docstring's contract is that malformed records are never repaired or inferred, and unqualified vectors keep keyword fallback
            return None


@dataclass(frozen=True)
class StoredVector:
    """One namespace-qualified read snapshot; absent proof means unknown."""

    embedding: tuple[float, ...]
    provenance: VectorProvenance | None


def provider_embedding_space(provider: object) -> EmbeddingSpace | None:
    """Read an optional already-loaded descriptor, never initialize a provider.

    ``embedding_space()`` is an opt-in extension, not an additional required
    method on the existing runtime-checkable EmbeddingProvider protocol.
    """
    descriptor = getattr(provider, "embedding_space", None)
    if not callable(descriptor):
        return None
    try:
        space = descriptor()
    except (OSError, ValueError, RuntimeError, TypeError):
        # trw-fail-silent-allow: embedding_space() is an opt-in extension; a provider that cannot describe its space is unqualified, which is the same outcome as not implementing it
        return None
    return space if isinstance(space, EmbeddingSpace) else None


def generation_provenance_kwargs(
    provider: object,
    text: str,
    embedding: Sequence[float],
    *,
    input_role: str = "document",
    parent_text: str | None = None,
) -> dict[str, VectorProvenance]:
    """Bind an actual encode result; unknown providers keep legacy call shape.

    Called immediately after successful encoding with its exact input string.
    Unknown identity never receives a retrospective descriptor from configuration.
    """
    space = provider_embedding_space(provider)
    if space is None:
        return {}
    try:
        return {
            "provenance": VectorProvenance.for_vector(
                space, text, embedding, input_role=input_role, parent_text=parent_text
            )
        }
    except (ValueError, TypeError):
        # trw-fail-silent-allow: {} means no provenance was bound, so the vector stays unqualified and keeps keyword fallback -- inventing a descriptor here is the failure this gate exists to prevent
        return {}
