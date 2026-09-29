"""TRW_REQUIRE_SQLITE_VEC=1 turns a missing or broken sqlite-vec into a session refusal, not skips."""

from __future__ import annotations

import sys

import pytest

from ._optional_extras import refuse_missing_sqlite_vec_when_required


def test_no_refusal_without_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRW_REQUIRE_SQLITE_VEC", raising=False)
    monkeypatch.setitem(sys.modules, "sqlite_vec", None)  # would fail to import
    assert refuse_missing_sqlite_vec_when_required() is None


def test_refusal_names_the_import_failure_under_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRW_REQUIRE_SQLITE_VEC", "1")
    monkeypatch.setitem(sys.modules, "sqlite_vec", None)  # `import sqlite_vec` now raises ImportError
    reason = refuse_missing_sqlite_vec_when_required()
    assert reason is not None
    assert "sqlite-vec does not import" in reason


def test_no_refusal_under_the_flag_when_it_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sqlite_vec")
    monkeypatch.setenv("TRW_REQUIRE_SQLITE_VEC", "1")
    assert refuse_missing_sqlite_vec_when_required() is None
