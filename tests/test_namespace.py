"""Tests for namespace validation and path mapping (FR07)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from trw_memory.exceptions import ConfigError
from trw_memory.namespaces.path_mapping import namespace_to_path
from trw_memory.namespaces.validation import validate_namespace


class TestValidateNamespace:
    """validate_namespace accepts valid patterns, rejects invalid ones."""

    @pytest.mark.parametrize(
        "ns",
        [
            "project:repo-a",
            "project:my_project",
            "project:A123",
            "global",
            "default",
            "team:research",
            "team:back-end",
            "org:acme-corp",
            "org:MyOrg_2",
        ],
    )
    def test_valid_namespaces(self, ns: str) -> None:
        assert validate_namespace(ns) == ns

    @pytest.mark.parametrize(
        "ns",
        [
            "",
            "   ",
            "bad",
            "project:",
            "project:foo/bar",
            "project:foo bar",
            "project:foo.bar",
            "PROJECT:foo",
            "global:extra",
            "team:",
            "org:",
            "unknown:scope",
            "project:foo:bar",
        ],
    )
    def test_invalid_namespaces(self, ns: str) -> None:
        with pytest.raises(ConfigError):
            validate_namespace(ns)

    def test_too_long_namespace(self) -> None:
        long_ns = "project:" + "a" * 200
        with pytest.raises(ConfigError, match="too long"):
            validate_namespace(long_ns)

    @pytest.mark.parametrize("ns", [b"global", ["global"], {"namespace": "global"}])
    def test_non_string_namespaces_are_rejected(self, ns: object) -> None:
        with pytest.raises(ConfigError, match="namespace must be a string"):
            validate_namespace(cast("str", ns))

    def test_object_cannot_spoof_namespace_through_strip(self) -> None:
        class SpoofedNamespace:
            def strip(self) -> str:
                return "global"

        with pytest.raises(ConfigError, match="namespace must be a string"):
            validate_namespace(cast("str", SpoofedNamespace()))


class TestNamespaceToPath:
    """namespace_to_path maps namespace strings to relative paths."""

    def test_project_namespace(self) -> None:
        assert namespace_to_path("project:repo-a") == Path("project/repo-a")

    def test_global_namespace(self) -> None:
        assert namespace_to_path("global") == Path("global")

    def test_default_namespace(self) -> None:
        assert namespace_to_path("default") == Path("default")

    def test_team_namespace(self) -> None:
        assert namespace_to_path("team:research") == Path("team/research")

    def test_org_namespace(self) -> None:
        assert namespace_to_path("org:acme-corp") == Path("org/acme-corp")

    def test_invalid_raises(self) -> None:
        with pytest.raises(ConfigError):
            namespace_to_path("invalid")
