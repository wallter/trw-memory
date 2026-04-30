"""Tests for the trw_memory exception hierarchy."""

from __future__ import annotations

from trw_memory.exceptions import ConfigError, MemoryError, StorageError


def test_memory_error_message_and_path() -> None:
    err = MemoryError("test msg", path="/some/path")
    assert str(err) == "test msg"
    assert err.path == "/some/path"


def test_memory_error_default_path_empty() -> None:
    err = MemoryError("test")
    assert err.path == ""


def test_storage_error_inherits_memory_error() -> None:
    err = StorageError("storage fail", path="/db")
    assert isinstance(err, MemoryError)
    assert err.path == "/db"


def test_config_error_inherits_memory_error() -> None:
    err = ConfigError("config fail")
    assert isinstance(err, MemoryError)
