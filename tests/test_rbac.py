"""Tests for trw_memory.security.rbac — Role-Based Access Control.

Coverage:
- Role and Permission enum membership and string values
- ROLE_PERMISSIONS matrix: all role+permission combinations
- check_permission(): all valid and invalid combinations
- require_permission() decorator: allow, deny, missing kwarg, string role
- Edge cases: unknown role string, async function decoration
"""

from __future__ import annotations

import time

import pytest

from trw_memory.exceptions import AuthorizationError, ConfigError
from trw_memory.models.config import MemoryConfig
from trw_memory.security import (
    ROLE_PERMISSIONS,
    Permission,
    Role,
    check_permission,
    require_namespace_permission,
    require_permission,
)

# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------


class TestRoleEnum:
    def test_reader_member_exists(self) -> None:
        assert hasattr(Role, "READER")

    def test_writer_member_exists(self) -> None:
        assert hasattr(Role, "WRITER")

    def test_admin_member_exists(self) -> None:
        assert hasattr(Role, "ADMIN")

    def test_none_member_exists(self) -> None:
        assert hasattr(Role, "NONE")

    def test_reader_value(self) -> None:
        assert Role.READER.value == "reader"

    def test_writer_value(self) -> None:
        assert Role.WRITER.value == "writer"

    def test_admin_value(self) -> None:
        assert Role.ADMIN.value == "admin"

    def test_none_value(self) -> None:
        assert Role.NONE.value == "none"

    def test_role_is_str_subclass(self) -> None:
        assert isinstance(Role.READER, str)
        assert isinstance(Role.WRITER, str)
        assert isinstance(Role.ADMIN, str)

    def test_role_from_string_reader(self) -> None:
        assert Role("reader") is Role.READER

    def test_role_from_string_writer(self) -> None:
        assert Role("writer") is Role.WRITER

    def test_role_from_string_admin(self) -> None:
        assert Role("admin") is Role.ADMIN

    def test_role_from_string_none(self) -> None:
        assert Role("none") is Role.NONE

    def test_role_helper_from_string_admin(self) -> None:
        assert Role.from_string("ADMIN") is Role.ADMIN

    def test_role_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError):
            Role("superuser")

    def test_role_enum_has_exactly_four_members(self) -> None:
        assert len(list(Role)) == 4

    @pytest.mark.parametrize(
        "role,expected",
        [
            (Role.READER, "reader"),
            (Role.WRITER, "writer"),
            (Role.ADMIN, "admin"),
            (Role.NONE, "none"),
        ],
    )
    def test_parametrized_role_string_values(self, role: Role, expected: str) -> None:
        assert role.value == expected
        assert role == expected  # str enum equality


# ---------------------------------------------------------------------------
# Permission enum
# ---------------------------------------------------------------------------


class TestPermissionEnum:
    def test_read_member_exists(self) -> None:
        assert hasattr(Permission, "READ")

    def test_write_member_exists(self) -> None:
        assert hasattr(Permission, "WRITE")

    def test_delete_member_exists(self) -> None:
        assert hasattr(Permission, "DELETE")

    def test_admin_member_exists(self) -> None:
        assert hasattr(Permission, "ADMIN")

    def test_read_value(self) -> None:
        assert Permission.READ.value == "read"

    def test_write_value(self) -> None:
        assert Permission.WRITE.value == "write"

    def test_delete_value(self) -> None:
        assert Permission.DELETE.value == "delete"

    def test_admin_value(self) -> None:
        assert Permission.ADMIN.value == "admin"

    def test_permission_is_str_subclass(self) -> None:
        assert isinstance(Permission.READ, str)

    def test_permission_enum_has_exactly_four_members(self) -> None:
        assert len(list(Permission)) == 4

    @pytest.mark.parametrize(
        "perm,expected",
        [
            (Permission.READ, "read"),
            (Permission.WRITE, "write"),
            (Permission.DELETE, "delete"),
            (Permission.ADMIN, "admin"),
        ],
    )
    def test_parametrized_permission_string_values(self, perm: Permission, expected: str) -> None:
        assert perm.value == expected


# ---------------------------------------------------------------------------
# ROLE_PERMISSIONS matrix
# ---------------------------------------------------------------------------


class TestRolePermissionsMatrix:
    def test_reader_has_read(self) -> None:
        assert Permission.READ in ROLE_PERMISSIONS[Role.READER]

    def test_reader_lacks_write(self) -> None:
        assert Permission.WRITE not in ROLE_PERMISSIONS[Role.READER]

    def test_reader_lacks_delete(self) -> None:
        assert Permission.DELETE not in ROLE_PERMISSIONS[Role.READER]

    def test_reader_lacks_admin(self) -> None:
        assert Permission.ADMIN not in ROLE_PERMISSIONS[Role.READER]

    def test_writer_lacks_read(self) -> None:
        assert Permission.READ not in ROLE_PERMISSIONS[Role.WRITER]

    def test_writer_has_write(self) -> None:
        assert Permission.WRITE in ROLE_PERMISSIONS[Role.WRITER]

    def test_writer_lacks_delete(self) -> None:
        assert Permission.DELETE not in ROLE_PERMISSIONS[Role.WRITER]

    def test_writer_lacks_admin(self) -> None:
        assert Permission.ADMIN not in ROLE_PERMISSIONS[Role.WRITER]

    def test_admin_has_read(self) -> None:
        assert Permission.READ in ROLE_PERMISSIONS[Role.ADMIN]

    def test_admin_has_write(self) -> None:
        assert Permission.WRITE in ROLE_PERMISSIONS[Role.ADMIN]

    def test_admin_has_delete(self) -> None:
        assert Permission.DELETE in ROLE_PERMISSIONS[Role.ADMIN]

    def test_admin_has_admin(self) -> None:
        assert Permission.ADMIN in ROLE_PERMISSIONS[Role.ADMIN]

    def test_reader_has_exactly_one_permission(self) -> None:
        assert len(ROLE_PERMISSIONS[Role.READER]) == 1

    def test_writer_has_exactly_one_permission(self) -> None:
        assert len(ROLE_PERMISSIONS[Role.WRITER]) == 1

    def test_admin_has_all_four_permissions(self) -> None:
        assert len(ROLE_PERMISSIONS[Role.ADMIN]) == 4

    def test_none_has_no_permissions(self) -> None:
        assert len(ROLE_PERMISSIONS[Role.NONE]) == 0

    def test_all_roles_present_in_matrix(self) -> None:
        for role in Role:
            assert role in ROLE_PERMISSIONS

    def test_matrix_values_are_sets(self) -> None:
        for role in Role:
            assert isinstance(ROLE_PERMISSIONS[role], set)


# ---------------------------------------------------------------------------
# check_permission()
# ---------------------------------------------------------------------------


class TestCheckPermission:
    # Reader permissions
    def test_reader_read_allowed(self) -> None:
        assert check_permission(Role.READER, Permission.READ) is True

    def test_reader_write_denied(self) -> None:
        assert check_permission(Role.READER, Permission.WRITE) is False

    def test_reader_delete_denied(self) -> None:
        assert check_permission(Role.READER, Permission.DELETE) is False

    def test_reader_admin_denied(self) -> None:
        assert check_permission(Role.READER, Permission.ADMIN) is False

    # Writer permissions
    def test_writer_read_denied(self) -> None:
        assert check_permission(Role.WRITER, Permission.READ) is False

    def test_writer_write_allowed(self) -> None:
        assert check_permission(Role.WRITER, Permission.WRITE) is True

    def test_writer_delete_denied(self) -> None:
        assert check_permission(Role.WRITER, Permission.DELETE) is False

    def test_writer_admin_denied(self) -> None:
        assert check_permission(Role.WRITER, Permission.ADMIN) is False

    # Admin permissions — all allowed
    def test_admin_read_allowed(self) -> None:
        assert check_permission(Role.ADMIN, Permission.READ) is True

    def test_admin_write_allowed(self) -> None:
        assert check_permission(Role.ADMIN, Permission.WRITE) is True

    def test_admin_delete_allowed(self) -> None:
        assert check_permission(Role.ADMIN, Permission.DELETE) is True

    def test_admin_admin_allowed(self) -> None:
        assert check_permission(Role.ADMIN, Permission.ADMIN) is True

    def test_none_read_denied(self) -> None:
        assert check_permission(Role.NONE, Permission.READ) is False

    def test_returns_bool_type(self) -> None:
        result = check_permission(Role.READER, Permission.READ)
        assert type(result) is bool

    @pytest.mark.parametrize(
        "role,perm,expected",
        [
            (Role.READER, Permission.READ, True),
            (Role.READER, Permission.WRITE, False),
            (Role.READER, Permission.DELETE, False),
            (Role.READER, Permission.ADMIN, False),
            (Role.WRITER, Permission.READ, False),
            (Role.WRITER, Permission.WRITE, True),
            (Role.WRITER, Permission.DELETE, False),
            (Role.WRITER, Permission.ADMIN, False),
            (Role.ADMIN, Permission.READ, True),
            (Role.ADMIN, Permission.WRITE, True),
            (Role.ADMIN, Permission.DELETE, True),
            (Role.ADMIN, Permission.ADMIN, True),
            (Role.NONE, Permission.READ, False),
            (Role.NONE, Permission.WRITE, False),
            (Role.NONE, Permission.DELETE, False),
            (Role.NONE, Permission.ADMIN, False),
        ],
    )
    def test_parametrized_all_combinations(self, role: Role, perm: Permission, expected: bool) -> None:
        assert check_permission(role, perm) is expected


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
        config = MemoryConfig(rbac_enabled=True, default_role=role)  # type: ignore[arg-type]

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


# ---------------------------------------------------------------------------
# require_permission() decorator — allow cases
# ---------------------------------------------------------------------------


class TestRequirePermissionAllow:
    def test_reader_can_call_read_function(self) -> None:
        @require_permission(Permission.READ)
        def read_data(*, role: Role) -> str:
            return "data"

        result = read_data(role=Role.READER)
        assert result == "data"

    def test_writer_can_call_write_function(self) -> None:
        @require_permission(Permission.WRITE)
        def write_data(*, role: Role) -> str:
            return "written"

        assert write_data(role=Role.WRITER) == "written"

    def test_admin_can_call_delete_function(self) -> None:
        @require_permission(Permission.DELETE)
        def delete_data(*, role: Role) -> str:
            return "deleted"

        assert delete_data(role=Role.ADMIN) == "deleted"

    def test_admin_can_call_admin_function(self) -> None:
        @require_permission(Permission.ADMIN)
        def admin_op(*, role: Role) -> str:
            return "admin-done"

        assert admin_op(role=Role.ADMIN) == "admin-done"

    def test_positional_args_passed_through(self) -> None:
        @require_permission(Permission.WRITE)
        def add(a: int, b: int, *, role: Role) -> int:
            return a + b

        result = add(3, 4, role=Role.WRITER)
        assert result == 7

    def test_function_name_preserved_by_wraps(self) -> None:
        @require_permission(Permission.READ)
        def my_function(*, role: Role) -> None:
            pass

        assert my_function.__name__ == "my_function"

    def test_function_docstring_preserved(self) -> None:
        @require_permission(Permission.READ)
        def documented_fn(*, role: Role) -> None:
            """This is the docstring."""

        assert documented_fn.__doc__ == "This is the docstring."


# ---------------------------------------------------------------------------
# require_permission() decorator — deny cases
# ---------------------------------------------------------------------------


class TestRequirePermissionDeny:
    def test_reader_cannot_call_write_function(self) -> None:
        @require_permission(Permission.WRITE)
        def write_data(*, role: Role) -> str:
            return "written"

        with pytest.raises(AuthorizationError, match="permission"):
            write_data(role=Role.READER)

    def test_reader_cannot_call_delete_function(self) -> None:
        @require_permission(Permission.DELETE)
        def delete_data(*, role: Role) -> str:
            return "deleted"

        with pytest.raises(AuthorizationError, match="permission"):
            delete_data(role=Role.READER)

    def test_reader_cannot_call_admin_function(self) -> None:
        @require_permission(Permission.ADMIN)
        def admin_op(*, role: Role) -> str:
            return "done"

        with pytest.raises(AuthorizationError, match="permission"):
            admin_op(role=Role.READER)

    def test_writer_cannot_call_delete_function(self) -> None:
        @require_permission(Permission.DELETE)
        def delete_data(*, role: Role) -> str:
            return "deleted"

        with pytest.raises(AuthorizationError, match="permission"):
            delete_data(role=Role.WRITER)

    def test_writer_cannot_call_admin_function(self) -> None:
        @require_permission(Permission.ADMIN)
        def admin_op(*, role: Role) -> str:
            return "done"

        with pytest.raises(AuthorizationError, match="permission"):
            admin_op(role=Role.WRITER)

    def test_writer_cannot_call_read_function(self) -> None:
        @require_permission(Permission.READ)
        def read_data(*, role: Role) -> str:
            return "data"

        with pytest.raises(AuthorizationError, match="read"):
            read_data(role=Role.WRITER)

    def test_error_message_contains_role_name(self) -> None:
        @require_permission(Permission.DELETE)
        def delete_fn(*, role: Role) -> None:
            pass

        with pytest.raises(AuthorizationError, match="reader"):
            delete_fn(role=Role.READER)

    def test_error_message_contains_permission_name(self) -> None:
        @require_permission(Permission.DELETE)
        def delete_fn(*, role: Role) -> None:
            pass

        with pytest.raises(AuthorizationError, match="delete"):
            delete_fn(role=Role.READER)

    def test_decorated_function_not_called_on_deny(self) -> None:
        called: list[bool] = []

        @require_permission(Permission.ADMIN)
        def side_effect_fn(*, role: Role) -> None:
            called.append(True)

        with pytest.raises(AuthorizationError):
            side_effect_fn(role=Role.READER)

        assert called == []


# ---------------------------------------------------------------------------
# require_permission() decorator — missing role kwarg
# ---------------------------------------------------------------------------


class TestRequirePermissionMissingRole:
    def test_raises_config_error_when_role_missing(self) -> None:
        @require_permission(Permission.READ)
        def read_fn(**kwargs: object) -> None:
            pass

        with pytest.raises(ConfigError, match="Missing 'role' kwarg"):
            read_fn()  # no role kwarg

    def test_error_message_contains_permission(self) -> None:
        @require_permission(Permission.WRITE)
        def write_fn(**kwargs: object) -> None:
            pass

        with pytest.raises(ConfigError, match="write"):
            write_fn()

    def test_none_role_raises_config_error(self) -> None:
        @require_permission(Permission.READ)
        def read_fn(*, role: Role | None = None) -> None:
            pass

        with pytest.raises(ConfigError, match="Missing 'role' kwarg"):
            read_fn(role=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# require_permission() decorator — string role coercion
# ---------------------------------------------------------------------------


class TestRequirePermissionStringRole:
    def test_string_reader_accepted(self) -> None:
        @require_permission(Permission.READ)
        def read_fn(*, role: object) -> str:
            return "ok"

        result = read_fn(role="reader")  # type: ignore[arg-type]
        assert result == "ok"

    def test_string_writer_accepted(self) -> None:
        @require_permission(Permission.WRITE)
        def write_fn(*, role: object) -> str:
            return "written"

        result = write_fn(role="writer")  # type: ignore[arg-type]
        assert result == "written"

    def test_string_admin_accepted(self) -> None:
        @require_permission(Permission.ADMIN)
        def admin_fn(*, role: object) -> str:
            return "admin"

        result = admin_fn(role="admin")  # type: ignore[arg-type]
        assert result == "admin"

    def test_invalid_string_role_raises(self) -> None:
        @require_permission(Permission.READ)
        def read_fn(*, role: object) -> None:
            pass

        with pytest.raises(ValueError):
            read_fn(role="superuser")  # type: ignore[arg-type]

    def test_string_reader_denied_write(self) -> None:
        @require_permission(Permission.WRITE)
        def write_fn(*, role: object) -> None:
            pass

        with pytest.raises(AuthorizationError, match="permission"):
            write_fn(role="reader")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# require_permission() decorator — async function support
# ---------------------------------------------------------------------------


class TestRequirePermissionAsync:
    async def test_async_function_allowed(self) -> None:
        @require_permission(Permission.READ)
        async def async_read(*, role: Role) -> str:
            return "async-data"

        result = await async_read(role=Role.READER)
        assert result == "async-data"

    async def test_async_function_denied(self) -> None:
        @require_permission(Permission.WRITE)
        async def async_write(*, role: Role) -> str:
            return "async-written"

        with pytest.raises(AuthorizationError, match="permission"):
            await async_write(role=Role.READER)

    async def test_async_function_name_preserved(self) -> None:
        @require_permission(Permission.READ)
        async def my_async_fn(*, role: Role) -> None:
            pass

        assert my_async_fn.__name__ == "my_async_fn"


# ---------------------------------------------------------------------------
# Parametrized deny matrix — all combinations that should be denied
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role,perm",
    [
        (Role.READER, Permission.WRITE),
        (Role.READER, Permission.DELETE),
        (Role.READER, Permission.ADMIN),
        (Role.WRITER, Permission.READ),
        (Role.WRITER, Permission.DELETE),
        (Role.WRITER, Permission.ADMIN),
        (Role.NONE, Permission.READ),
        (Role.NONE, Permission.WRITE),
        (Role.NONE, Permission.DELETE),
        (Role.NONE, Permission.ADMIN),
    ],
)
def test_deny_matrix_via_decorator(role: Role, perm: Permission) -> None:
    @require_permission(perm)
    def guarded(*, role: Role) -> None:
        pass

    with pytest.raises(AuthorizationError, match="permission"):
        guarded(role=role)


# ---------------------------------------------------------------------------
# Parametrized allow matrix — all combinations that should be allowed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role,perm",
    [
        (Role.READER, Permission.READ),
        (Role.WRITER, Permission.WRITE),
        (Role.ADMIN, Permission.READ),
        (Role.ADMIN, Permission.WRITE),
        (Role.ADMIN, Permission.DELETE),
        (Role.ADMIN, Permission.ADMIN),
    ],
)
def test_allow_matrix_via_decorator(role: Role, perm: Permission) -> None:
    @require_permission(perm)
    def guarded(*, role: Role) -> str:
        return "allowed"

    result = guarded(role=role)
    assert result == "allowed"
