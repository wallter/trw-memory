"""Role-Based Access Control (RBAC) for memory operations.

Defines a simple role/permission model with three roles:
- **reader**: can read memories
- **writer**: can read and write memories
- **admin**: full access including delete and admin operations
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

from trw_memory.exceptions import ConfigError

F = TypeVar("F", bound=Callable[..., Any])


class Role(str, Enum):
    """User roles for RBAC."""

    READER = "reader"
    WRITER = "writer"
    ADMIN = "admin"


class Permission(str, Enum):
    """Granular permissions for memory operations."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.READER: {Permission.READ},
    Role.WRITER: {Permission.READ, Permission.WRITE},
    Role.ADMIN: {
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
        Permission.ADMIN,
    },
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


def require_permission(permission: Permission) -> Callable[[F], F]:
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

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            role: Role | None = kwargs.get("role")
            if role is None:
                raise ConfigError(
                    f"Missing 'role' kwarg required by "
                    f"@require_permission({permission.value!r})"
                )
            if not isinstance(role, Role):
                role = Role(role)
            if not check_permission(role, permission):
                raise ConfigError(
                    f"Permission denied: role {role.value!r} "
                    f"lacks {permission.value!r} permission"
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
