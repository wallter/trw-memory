"""RBAC namespace enforcement tests."""

from __future__ import annotations

import time
from typing import Literal, cast

import pytest

from trw_memory.exceptions import AuthorizationError
from trw_memory.models.config import MemoryConfig
from trw_memory.security import Permission, require_namespace_permission


class TestRequireNamespacePermission:
    @pytest.mark.parametrize(
        "operation,permission,role,allowed",
        [
            ("memory_store", Permission.WRITE, "admin", True),
            ("memory_store", Permission.WRITE, "writer", True),
            ("memory_store", Permission.WRITE, "reader", False),
            ("memory_store", Permission.WRITE, "none", False),
            ("memory_recall", Permission.READ, "admin", True),
            ("memory_recall", Permission.READ, "writer", False),
            ("memory_recall", Permission.READ, "reader", True),
            ("memory_recall", Permission.READ, "none", False),
            ("memory_forget", Permission.DELETE, "admin", True),
            ("memory_forget", Permission.DELETE, "writer", False),
            ("memory_forget", Permission.DELETE, "reader", False),
            ("memory_forget", Permission.DELETE, "none", False),
            ("memory_search", Permission.READ, "admin", True),
            ("memory_search", Permission.READ, "writer", False),
            ("memory_search", Permission.READ, "reader", True),
            ("memory_search", Permission.READ, "none", False),
            ("memory_rotate_key", Permission.ADMIN, "admin", True),
            ("memory_rotate_key", Permission.ADMIN, "writer", False),
            ("memory_rotate_key", Permission.ADMIN, "reader", False),
            ("memory_rotate_key", Permission.ADMIN, "none", False),
        ],
    )
    def test_permission_matrix(
        self,
        operation: str,
        permission: Permission,
        role: str,
        allowed: bool,
    ) -> None:
        config = MemoryConfig(
            rbac_enabled=True,
            default_role=cast("Literal['admin', 'reader', 'writer', 'none']", role),
        )

        if allowed:
            require_namespace_permission(config, "project:default", permission, operation)
            return

        with pytest.raises(
            AuthorizationError,
            match=f"Role '{role}' does not have {operation} permission on namespace 'project:default'",
        ):
            require_namespace_permission(config, "project:default", permission, operation)


def test_rbac_check_overhead_p99() -> None:
    config = MemoryConfig(
        rbac_enabled=True,
        namespace_roles={f"project:{index}": "admin" for index in range(10)},
        default_role="reader",
    )
    durations_ns: list[int] = []
    for _ in range(10_000):
        start = time.perf_counter_ns()
        require_namespace_permission(config, "project:5", Permission.ADMIN, "memory_rotate_key")
        durations_ns.append(time.perf_counter_ns() - start)

    durations_ns.sort()
    p99_ns = durations_ns[int(len(durations_ns) * 0.99)]
    assert p99_ns < 1_000_000
