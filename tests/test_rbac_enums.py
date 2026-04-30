"""RBAC enum coverage."""

from __future__ import annotations

import pytest

from trw_memory.security import Permission, Role


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
        assert role == expected


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
