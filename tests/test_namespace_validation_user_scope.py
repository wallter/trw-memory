"""PRD-CORE-185 FR03: ``user:<id>`` namespace scope acceptance.

The validator must accept ``user:<name>`` (mirroring ``team:``/``org:``)
while preserving every existing accepted form and rejecting malformed input.
"""

from __future__ import annotations

import pytest

from trw_memory.exceptions import ConfigError
from trw_memory.namespaces.validation import validate_namespace


def test_user_scope_accepted() -> None:
    """``user:local`` (and an id variant) validate and return unchanged."""
    assert validate_namespace("user:local") == "user:local"
    assert validate_namespace("user:host-abc_123") == "user:host-abc_123"


def test_existing_scopes_still_pass() -> None:
    """FR03 must not regress the previously accepted namespace forms."""
    for ns in ("project:foo", "global", "default", "team:x", "org:y"):
        assert validate_namespace(ns) == ns


@pytest.mark.parametrize(
    "bad",
    [
        "user:",  # empty id
        "user:bad name",  # space in id
        "user:has.dot",  # dot not allowed
        "user:has/slash",  # slash not allowed
        "user",  # bare scope without id
        "users:local",  # wrong scope keyword
    ],
)
def test_malformed_user_scope_rejected(bad: str) -> None:
    """Malformed ``user:`` namespaces raise ConfigError."""
    with pytest.raises(ConfigError):
        validate_namespace(bad)


def test_error_message_names_user_scope() -> None:
    """The validation error message lists ``user:<name>`` as accepted."""
    with pytest.raises(ConfigError) as excinfo:
        validate_namespace("totally-invalid:scope")
    assert "user:" in str(excinfo.value)
