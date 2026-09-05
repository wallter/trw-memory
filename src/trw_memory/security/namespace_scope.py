"""The set of namespaces a retrieval call is cleared to rank (PRD-CORE-245 FR04/FR05).

``hybrid_search`` used to take a pre-selected entry list and no principal, so
isolation was entirely a property of how carefully each caller assembled that
list. This type makes it a property of the interface instead: the scope is a
required argument with no default, and the only ordinary way to obtain one is
:func:`authorize_namespaces`.

**What this guarantees, and what it does not.** Python has no capability tokens.
A frozen dataclass is bypassable by ``object.__new__`` plus
``object.__setattr__``, by ``dataclasses.replace``, and by monkeypatching the
sentinel below, so nothing here defends against a deliberately adversarial
in-process caller. The claim is narrower and is the one worth making: the
*ordinary* construction path fails closed, so obtaining an unscoped ranking
stops being something a caller can do by omission and becomes an explicit act of
private or reflective construction that a reviewer or a grep can see. This is an
anti-accidental-misuse boundary, not a security boundary.

Containment does NOT depend on the RBAC toggle. ``rbac_enabled`` defaults to
false and stays false; :func:`authorize_namespaces` still validates every name
and still returns a scope, so the pipeline's membership assertion holds either
way. With RBAC on, a namespace the role cannot read is simply absent from the
scope -- and therefore never read from disk at all, rather than read and filtered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import AuthorizationError, ConfigError
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.security.rbac import Permission, require_namespace_permission

if TYPE_CHECKING:
    from collections.abc import Iterable

    from trw_memory.models.config import MemoryConfig

logger = structlog.get_logger(__name__)

__all__ = ["NamespaceScope", "NamespaceScopeError", "authorize_namespaces"]

#: Handed to the constructor by :func:`authorize_namespaces` and by nothing else.
#: Module-private on purpose: importing it to forge a scope is exactly the
#: explicit, greppable act this design wants a bypass to be.
_MINT_TOKEN = object()


class NamespaceScopeError(RuntimeError):
    """A retrieval call ran against entries outside the scope it was given.

    Raised rather than filtering: a caller that assembled a candidate list
    spanning namespaces it was not cleared for has a bug, and silently
    truncating the list would hide it.
    """


@dataclass(frozen=True)
class NamespaceScope:
    """The namespaces a retrieval call may rank, as minted by the authorizer.

    Construct through :func:`authorize_namespaces`. Calling this constructor
    directly raises :class:`NamespaceScopeError`.
    """

    namespaces: frozenset[str]
    _mint: object = None
    #: How many of the requested namespaces the authorizer refused. Kept on the
    #: scope because the count is otherwise destroyed at the authorizer's return
    #: statement, which made "you asked for one namespace" and "you asked for
    #: four and were cleared for one" the same scope, and therefore the same
    #: apparently-complete result to whoever ranked over it. A count only --
    #: never the names (NFR03).
    denied: int = 0

    def __post_init__(self) -> None:
        if self._mint is not _MINT_TOKEN:
            raise NamespaceScopeError(
                "NamespaceScope cannot be constructed directly; obtain one from "
                "trw_memory.security.namespace_scope.authorize_namespaces(), which is "
                "the only path that runs the namespace validation and permission checks "
                "a scope asserts have happened"
            )

    def __contains__(self, namespace: object) -> bool:
        return namespace in self.namespaces

    def __bool__(self) -> bool:
        return bool(self.namespaces)

    def without(self, namespace: str) -> NamespaceScope:
        """Return a scope with *namespace* removed.

        Narrowing an already-authorized scope needs no re-authorization -- it can
        only ever remove access. Used by the recall tool to drop an expired team
        namespace so it cannot satisfy the pipeline's membership assertion.
        """
        return NamespaceScope(self.namespaces - {namespace}, _MINT_TOKEN, self.denied)


def authorize_namespaces(
    config: MemoryConfig,
    namespaces: Iterable[str],
    permission: Permission,
    operation: str,
) -> NamespaceScope:
    """Return the scope containing exactly those *namespaces* that passed.

    Each name is run through ``validate_namespace`` (unconditional) and
    ``require_namespace_permission`` (a no-op while ``rbac_enabled`` is false).
    A name that fails either check is absent from the returned scope; it is not
    an error, because a caller listing several namespaces should get the ones it
    is entitled to rather than nothing. It IS, however, counted:
    :attr:`NamespaceScope.denied` carries how many were refused so a caller can
    report a narrowed result as narrowed instead of as complete.

    An empty result is a legitimate outcome and yields an empty scope, which the
    pipeline then treats as "no candidate may be ranked" -- fail-closed.
    """
    granted: set[str] = set()
    denied = 0
    for namespace in namespaces:
        try:
            validated = validate_namespace(namespace)
            require_namespace_permission(config, validated, permission, operation)
        except (AuthorizationError, ConfigError):
            denied += 1
            continue
        granted.add(validated)
    if denied:
        # Count only: a namespace label is operator-chosen but still user data
        # (PRD-CORE-245 NFR03 keeps names out of log lines).
        logger.debug("namespace_scope_denied", operation=operation, denied=denied, granted=len(granted))
    return NamespaceScope(frozenset(granted), _MINT_TOKEN, denied)
