"""RBAC permission matrix and direct checks."""

from __future__ import annotations

import pytest

from trw_memory.security import ROLE_PERMISSIONS, Permission, Role, check_permission


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


class TestCheckPermission:
    def test_reader_read_allowed(self) -> None:
        assert check_permission(Role.READER, Permission.READ) is True

    def test_reader_write_denied(self) -> None:
        assert check_permission(Role.READER, Permission.WRITE) is False

    def test_reader_delete_denied(self) -> None:
        assert check_permission(Role.READER, Permission.DELETE) is False

    def test_reader_admin_denied(self) -> None:
        assert check_permission(Role.READER, Permission.ADMIN) is False

    def test_writer_read_denied(self) -> None:
        assert check_permission(Role.WRITER, Permission.READ) is False

    def test_writer_write_allowed(self) -> None:
        assert check_permission(Role.WRITER, Permission.WRITE) is True

    def test_writer_delete_denied(self) -> None:
        assert check_permission(Role.WRITER, Permission.DELETE) is False

    def test_writer_admin_denied(self) -> None:
        assert check_permission(Role.WRITER, Permission.ADMIN) is False

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
