"""Role-Based Access Control (RBAC) for memory operations."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import structlog

from trw_memory.exceptions import AuthorizationError, ConfigError

if TYPE_CHECKING:
    from trw_memory.models.config import MemoryConfig

logger = structlog.get_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


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


def require_namespace_permission(
    config: "MemoryConfig",
    namespace: str,
    permission: Permission,
    operation: str,
) -> None:
    """Enforce a namespace-scoped permission using the current MemoryConfig."""
    if not config.rbac_enabled:
        return
    role_name = config.namespace_roles.get(namespace, config.default_role)
    role = Role.from_string(role_name)
    if check_permission(role, permission):
        return
    raise AuthorizationError(f"Role '{role.value}' does not have {operation} permission on namespace '{namespace}'.")


def require_permission(
    permission: Permission,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator that enforces a permission check before function execution.

    The decorated function **must** accept a keyword argument ``role``
    (of type :class:`Role`).  If the role lacks the required permission,
    a :class:`ConfigError` is raised.

    Args:
        permission: The required permission.

    Returns:
        A decorator that wraps the target function with a permission gate.

    Example::

        @require_permission(Permission.WRITE)
        def store_memory(entry: MemoryEntry, *, role: Role) -> None:
            ...
    """

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            role: object = kwargs.get("role")
            if role is None:
                raise ConfigError(f"Missing 'role' kwarg required by @require_permission({permission.value!r})")
            if not isinstance(role, Role):
                role = Role.from_string(str(role))
            if not check_permission(role, permission):
                logger.warning("authorization_denied", op="rbac", role=role.value, permission=permission.value)
                raise AuthorizationError(f"Role '{role.value}' does not have {permission.value} permission.")
            return func(*args, **kwargs)

        return wrapper

    return decorator
