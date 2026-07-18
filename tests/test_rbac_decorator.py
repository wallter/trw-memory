"""RBAC decorator coverage."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from trw_memory.exceptions import AuthorizationError, ConfigError
from trw_memory.security import Permission, Role, require_permission


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


class TestRequirePermissionMissingRole:
    def test_raises_config_error_when_role_missing(self) -> None:
        @require_permission(Permission.READ)
        def read_fn(**kwargs: object) -> None:
            pass

        with pytest.raises(ConfigError, match="Missing 'role' kwarg"):
            read_fn()

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
            read_fn(role=None)


class TestRequirePermissionStringRole:
    def test_string_reader_accepted(self) -> None:
        @require_permission(Permission.READ)
        def read_fn(*, role: object) -> str:
            return "ok"

        result = read_fn(role="reader")
        assert result == "ok"

    def test_string_role_is_normalized_before_protected_function(self) -> None:
        @require_permission(Permission.READ)
        def read_fn(*, role: object) -> object:
            return role

        assert read_fn(role="reader") is Role.READER

    def test_string_writer_accepted(self) -> None:
        @require_permission(Permission.WRITE)
        def write_fn(*, role: object) -> str:
            return "written"

        result = write_fn(role="writer")
        assert result == "written"

    def test_string_admin_accepted(self) -> None:
        @require_permission(Permission.ADMIN)
        def admin_fn(*, role: object) -> str:
            return "admin"

        result = admin_fn(role="admin")
        assert result == "admin"

    def test_invalid_string_role_raises(self) -> None:
        @require_permission(Permission.READ)
        def read_fn(*, role: object) -> None:
            pass

        with pytest.raises(ConfigError, match="Invalid 'role' value"):
            read_fn(role="superuser")

    def test_arbitrary_object_cannot_self_promote_through_string_conversion(self) -> None:
        called = False

        class SpoofedAdmin:
            def __str__(self) -> str:
                return "admin"

        @require_permission(Permission.ADMIN)
        def admin_fn(*, role: object) -> None:
            nonlocal called
            called = True

        with pytest.raises(ConfigError, match="Invalid 'role' type"):
            admin_fn(role=SpoofedAdmin())
        assert called is False

    @pytest.mark.parametrize("role", [b"admin", ["admin"], {"role": "admin"}])
    def test_non_string_role_values_are_rejected(self, role: object) -> None:
        @require_permission(Permission.ADMIN)
        def admin_fn(*, role: object) -> None:
            pass

        with pytest.raises(ConfigError, match="Invalid 'role' type"):
            admin_fn(role=role)

    def test_string_reader_denied_write(self) -> None:
        @require_permission(Permission.WRITE)
        def write_fn(*, role: object) -> None:
            pass

        with pytest.raises(AuthorizationError, match="permission"):
            write_fn(role="reader")


class TestRequirePermissionAsync:
    def test_async_function_preserves_coroutine_identity(self) -> None:
        @require_permission(Permission.READ)
        async def async_read(*, role: Role) -> str:
            return "async-data"

        assert inspect.iscoroutinefunction(async_read)
        assert asyncio.iscoroutinefunction(async_read)

    async def test_async_function_allowed(self) -> None:
        @require_permission(Permission.READ)
        async def async_read(*, role: Role) -> str:
            return "async-data"

        result = await async_read(role=Role.READER)
        assert result == "async-data"

    async def test_async_string_role_is_normalized_before_protected_function(self) -> None:
        @require_permission(Permission.READ)
        async def async_read(*, role: object) -> object:
            return role

        assert await async_read(role="reader") is Role.READER

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
