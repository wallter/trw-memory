"""The typed contract for the ``trw_memory.tools`` public surface (PRD-CORE-251 FR01).

``trw_memory.tools`` is the only layer another package is meant to reach a
memory *concern* through: storing, recalling, searching, forgetting,
consolidating, and the three reporting surfaces.  Until this module existed the
package exported those callables as a flat list with no declared contract, so a
signature could change under a downstream caller and nothing would say so until
the call raised at runtime.

``MemoryToolSurface`` makes the surface a checkable object.  Each member is a
callback Protocol declaring the *full* call shape of one ``memory_*_impl``
function; ``MemoryToolSurface`` then declares the module-level surface those
callables form.  ``trw_memory/tools/__init__.py`` binds every impl to its member
inside a ``TYPE_CHECKING`` block, so ``mypy --strict`` rejects any drift between
an implementation and the contract it publishes.

Adding a new *defaulted* keyword argument to an impl stays compatible on
purpose — that is a backward-compatible extension.  Removing or renaming a
parameter, changing its kind, or changing a return type is what this contract
exists to fail on.

**``memory_update_impl`` is deliberately absent.**  PRD-CORE-251 FR07 adds it
in Phase 5, and the operator decision of 2026-09-03 fixed its shape: it takes a
TYPED update model rather than a free-form field dict, so the permission check
and the audit event can name what changed.  It is not declared here yet because
FR01's own acceptance criterion requires every member of this Protocol to
resolve to a public callable exported by ``trw_memory.tools``, and the PRD's
phasing (section 9, Phase 1) never authorises a declared-but-unimplemented
member.  A ``NotImplementedError`` stub would satisfy the letter of the
Protocol while making the contract test vacuous for that member.  FR07 adds
both the implementation and its ``UpdateImpl`` member in the same change.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from trw_memory.models.config import MemoryConfig
    from trw_memory.models.memory import Assertion
    from trw_memory.storage.interface import StorageBackend


class StoreImpl(Protocol):
    """Contract for :func:`trw_memory.tools.store.memory_store_impl`."""

    def __call__(
        self,
        content: str,
        namespace: str,
        *,
        backend: StorageBackend,
        tags: list[str] | None = None,
        importance: float = 0.5,
        detail: str = "",
        metadata: dict[str, str] | None = None,
        config: MemoryConfig | None = None,
        source: Literal["human", "agent", "tool", "consolidated"] = "tool",
        source_identity: str = "",
        session_id: str | None = None,
        entry_id: str | None = None,
        evidence: list[str] | None = None,
        expires: str = "",
        assertions: list[Assertion] | None = None,
        raise_security_errors: bool = False,
    ) -> dict[str, object]: ...


class RecallImpl(Protocol):
    """Contract for :func:`trw_memory.tools.recall.memory_recall_impl`."""

    def __call__(
        self,
        query: str,
        namespace: str,
        *,
        backend: StorageBackend,
        namespace_backend_factory: Callable[[str], StorageBackend] | None = None,
        limit: int = 25,
        min_score: float = 0.0,
        tags: list[str] | None = None,
        include_namespaces: list[str] | None = None,
        include_org_memories: bool = True,
        graph_depth: int = 0,
        conn: sqlite3.Connection | None = None,
        token_budget: int | None = None,
        config: MemoryConfig | None = None,
        include_distilled: bool = True,
        distilled_weight: float | None = None,
        include_source_kinds: list[str] | None = None,
        exclude_source_kinds: list[str] | None = None,
        source_weights: dict[str, float] | None = None,
        exclude_expired: bool = True,
    ) -> dict[str, object]: ...


class SearchImpl(Protocol):
    """Contract for :func:`trw_memory.tools.search.memory_search_impl`."""

    def __call__(
        self,
        namespace: str,
        *,
        backend: StorageBackend,
        config: MemoryConfig | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
        sort_by: str = "updated_at",
        offset: int = 0,
        limit: int = 50,
        actor: str | None = None,
    ) -> dict[str, object]: ...


class ForgetImpl(Protocol):
    """Contract for :func:`trw_memory.tools.forget.memory_forget_impl`."""

    def __call__(
        self,
        memory_id: str | None,
        query: str | None,
        namespace: str,
        *,
        backend: StorageBackend,
        config: MemoryConfig | None = None,
        actor: str | None = None,
    ) -> dict[str, object]: ...


class ConsolidateImpl(Protocol):
    """Contract for :func:`trw_memory.tools.consolidate.memory_consolidate_impl`."""

    def __call__(
        self,
        namespace: str,
        *,
        backend: StorageBackend,
        dry_run: bool = False,
        config: MemoryConfig | None = None,
        namespace_backend_factory: Callable[[str], StorageBackend] | None = None,
    ) -> dict[str, object]: ...


class StatusImpl(Protocol):
    """Contract for :func:`trw_memory.tools.status.memory_status_impl`."""

    def __call__(
        self,
        namespace: str | None,
        *,
        backend: StorageBackend,
        config: MemoryConfig | None = None,
    ) -> dict[str, object]: ...


class ReviewImpl(Protocol):
    """Contract for :func:`trw_memory.tools.review.memory_review_impl`.

    Returns ``dict[str, str]``, not ``dict[str, object]``: the review verdict is
    a flat string record and the narrower type is part of the published shape.
    """

    def __call__(
        self,
        learning_id: str,
        *,
        decision: Literal["approve", "reject"],
        reviewer_id: str,
        namespace: str = "default",
        config: MemoryConfig | None = None,
    ) -> dict[str, str]: ...


class AuditImpl(Protocol):
    """Contract for :func:`trw_memory.tools.audit.memory_audit_impl`."""

    def __call__(
        self,
        learning_id: str,
        *,
        namespace: str = "default",
        config: MemoryConfig | None = None,
    ) -> dict[str, object]: ...


@runtime_checkable
class MemoryToolSurface(Protocol):
    """The ``trw_memory.tools`` surface a downstream package may depend on.

    The ``trw_memory.tools`` module itself is the implementation: every member
    below is a module-level function it exports.  ``isinstance(module, ...)``
    therefore answers "is this surface complete", and the per-member callback
    Protocols answer "is each signature still the published one".

    A downstream package that reaches a memory concern any other way — by
    importing a private helper, or by reimplementing the concern — is outside
    this contract.  ``scripts/check_memory_boundary.py`` is what makes that
    statement enforceable rather than aspirational.
    """

    memory_store_impl: StoreImpl
    memory_recall_impl: RecallImpl
    memory_search_impl: SearchImpl
    memory_forget_impl: ForgetImpl
    memory_consolidate_impl: ConsolidateImpl
    memory_status_impl: StatusImpl
    memory_review_impl: ReviewImpl
    memory_audit_impl: AuditImpl


#: The member names ``MemoryToolSurface`` declares, in declaration order.
#: Derived from the Protocol rather than restated, so a member added to the
#: Protocol is automatically covered by the contract test.
SURFACE_MEMBERS: tuple[str, ...] = tuple(MemoryToolSurface.__annotations__)
