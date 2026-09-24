"""Role-Based Access Control (RBAC) for memory operations."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import AuthorizationError
from trw_memory.namespaces.validation import validate_namespace

if TYPE_CHECKING:
    from trw_memory.models.config import MemoryConfig

logger = structlog.get_logger(__name__)


class Role(str, Enum):
    """User roles for RBAC."""

    READER = "reader"
    WRITER = "writer"
    ADMIN = "admin"
    NONE = "none"

    @classmethod
    def from_string(cls, value: str) -> Role:
        """Parse a role name using the public string values."""
        return cls(value.strip().lower())


class Permission(str, Enum):
    """Granular permissions for memory operations."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.READER: {Permission.READ},
    Role.WRITER: {Permission.WRITE},
    Role.ADMIN: {
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
        Permission.ADMIN,
    },
    Role.NONE: set(),
}


def check_permission(role: Role, permission: Permission) -> bool:
    """Check whether *role* has the given *permission*.

    Args:
        role: The user's role.
        permission: The permission to check.

    Returns:
        ``True`` if the role grants the permission, ``False`` otherwise.
    """
    allowed = ROLE_PERMISSIONS.get(role, set())
    return permission in allowed


def transport_grant() -> frozenset[str] | None:
    """The namespaces the current daemon request's token was granted; ``None`` off the transport."""
    from fastmcp.server.dependencies import get_access_token

    token = get_access_token()
    if token is None:
        return None
    return frozenset(scope.removeprefix("ns:") for scope in token.scopes if scope.startswith("ns:"))


def transport_root() -> tuple[bool, str | None]:
    """``(on_transport, root)``: the checkout the request's token was minted for, if it names one."""
    from fastmcp.server.dependencies import get_access_token

    token = get_access_token()
    if token is None:
        return False, None
    root = token.claims.get("root")
    return True, root if isinstance(root, str) else None


def within_grant(namespace: str) -> bool:
    """Whether a whole-store reader may include *namespace* (always, off the transport)."""
    granted = transport_grant()
    return granted is None or namespace in granted


def require_namespace_permission(
    config: MemoryConfig,
    namespace: str,
    permission: Permission,
    operation: str,
) -> None:
    """Enforce a namespace-scoped permission: validation, the transport grant, then RBAC.

    The grant step precedes the ``rbac_enabled`` switch (PRD-CORE-298 FR02): a
    daemon request whose token was not granted *namespace* is refused even with
    RBAC off. Without a transport token -- the in-process SDK -- it is a no-op.
    """
    namespace = validate_namespace(namespace)
    granted = transport_grant()
    if granted is not None and namespace not in granted:
        logger.warning("authorization_denied", op="grant", operation=operation, namespace=namespace)
        raise AuthorizationError(f"This token is not granted namespace '{namespace}' ({operation}).")
    if not config.rbac_enabled:
        return
    role_name = config.namespace_roles.get(namespace, config.default_role)
    role = Role.from_string(role_name)
    if check_permission(role, permission):
        return
    raise AuthorizationError(f"Role '{role.value}' does not have {operation} permission on namespace '{namespace}'.")
