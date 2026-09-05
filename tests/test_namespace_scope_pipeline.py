"""PRD-CORE-245 FR04/FR05 + NFR03 — the retrieval boundary carries the scope.

The property under test is narrow and deliberate: the ORDINARY construction path
fails closed. Private or reflective construction (``object.__new__`` plus
``object.__setattr__``, ``dataclasses.replace``, monkeypatching the sentinel)
defeats any in-process guard in Python, and this suite does not pretend
otherwise — see ``test_the_guard_is_against_omission_not_against_reflection``.
"""

from __future__ import annotations

import dataclasses

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.pipeline import hybrid_search
from trw_memory.security.namespace_scope import (
    NamespaceScope,
    NamespaceScopeError,
    authorize_namespaces,
)
from trw_memory.security.rbac import Permission

pytestmark = pytest.mark.unit


def _scope(*namespaces: str, config: MemoryConfig | None = None) -> NamespaceScope:
    return authorize_namespaces(config or MemoryConfig(), namespaces, Permission.READ, "recall")


def _entry(entry_id: str, namespace: str) -> MemoryEntry:
    return MemoryEntry(id=entry_id, content="retrieval boundary probe", namespace=namespace)


def test_out_of_scope_entry_fails_closed() -> None:
    """FR04: a candidate outside the scope raises and yields nothing.

    It raises rather than filtering. A caller that assembled a list spanning
    namespaces it was not cleared for has a bug, and truncating the list here
    would hide it.
    """
    entries = [_entry("M-in", "default"), _entry("M-out", "project:elsewhere")]
    with pytest.raises(NamespaceScopeError, match="outside the authorized scope"):
        hybrid_search("retrieval boundary", entries, scope=_scope("default"))


def test_missing_scope_fails_closed() -> None:
    """NFR03: no scope argument is a TypeError before any retrieval step runs."""
    with pytest.raises(TypeError):
        hybrid_search("retrieval boundary", [_entry("M-in", "default")])  # type: ignore[call-arg]


def test_empty_scope_admits_nothing() -> None:
    """NFR03: an unestablishable scope returns no entries rather than degrading to unfiltered."""
    empty = authorize_namespaces(MemoryConfig(), [], Permission.READ, "recall")
    assert not empty
    with pytest.raises(NamespaceScopeError):
        hybrid_search("retrieval boundary", [_entry("M-in", "default")], scope=empty)


def test_in_scope_entries_rank_normally() -> None:
    """Control: the assertion must not turn a legitimate multi-namespace recall into an error."""
    entries = [_entry("M-a", "default"), _entry("M-b", "project:other")]
    ranked = hybrid_search("retrieval boundary probe", entries, scope=_scope("default", "project:other"))
    assert {entry.id for entry in ranked} == {"M-a", "M-b"}


def test_scope_is_minted_only_by_the_authorizer() -> None:
    """FR05: the ordinary constructor refuses, and RBAC denials are absent from the scope."""
    with pytest.raises(NamespaceScopeError, match="cannot be constructed directly"):
        NamespaceScope(frozenset({"default"}))

    # RBAC OFF (the default): every valid namespace is granted, so containment
    # does not depend on the toggle.
    assert _scope("default", "project:other").namespaces == {"default", "project:other"}

    # RBAC ON with a role that cannot read one namespace: it is simply absent.
    denied = MemoryConfig(rbac_enabled=True, default_role="reader", namespace_roles={"project:secret": "none"})
    scope = _scope("default", "project:secret", config=denied)
    assert scope.namespaces == {"default"}
    assert "project:secret" not in scope


def test_an_invalid_namespace_never_enters_the_scope() -> None:
    """``validate_namespace`` is the one unconditional check; a bad name is dropped, not raised."""
    scope = _scope("default", "not a valid namespace")
    assert scope.namespaces == {"default"}


def test_narrowing_a_scope_needs_no_reauthorization() -> None:
    """``without`` can only remove access, which is why it does not re-run the authorizer."""
    scope = _scope("default", "team:sprint-1")
    narrowed = scope.without("team:sprint-1")
    assert narrowed.namespaces == {"default"}
    assert scope.namespaces == {"default", "team:sprint-1"}, "the original scope is frozen"


def test_the_guard_is_against_omission_not_against_reflection() -> None:
    """Document the boundary honestly: reflective construction defeats it, by design.

    PRD-CORE-245 NFR03 accepts this. The claim being made is that an unscoped
    ranking stops being reachable by OMISSION and becomes an explicit act a
    reviewer or a grep can see — not that it is impossible.
    """
    forged = dataclasses.replace(_scope("default"), namespaces=frozenset({"anything"}))
    assert "anything" in forged, "reflection is expected to work; the guard is not a sandbox"
