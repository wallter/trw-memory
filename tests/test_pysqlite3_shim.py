"""The optional pysqlite3 engine path, isolated (PRD-INFRA-185 FR02).

``_pysqlite3_shim`` is the ONLY module that knows about the second engine, so it
is also the only place the mechanism can be tested: import, ``sys.modules``
installation, and eviction of the provisional swap ``trw_memory/__init__.py``
performs before any policy can run.

Everything here drives a synthetic ``pysqlite3``. It has to: on this repository's
own interpreter (CPython 3.14, stdlib SQLite 3.53.4) no published wheel is newer,
and on an old interpreter none is older, so neither direction of the policy is
reachable by installing something.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from trw_memory.storage import _pysqlite3_shim as shim


@pytest.fixture
def sqlite_modules_restored() -> object:
    """Snapshot and restore every sys.modules entry the shim may touch."""
    names = ("sqlite3", "sqlite3.dbapi2", "pysqlite3")
    saved = {name: sys.modules.get(name) for name in names}
    yield None
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def fake_pysqlite3(version: str) -> ModuleType:
    """A stand-in with the attributes the shim and the policy read."""
    module = ModuleType("pysqlite3")
    module.sqlite_version = version  # type: ignore[attr-defined]
    module.threadsafety = 1  # type: ignore[attr-defined]
    module.dbapi2 = ModuleType("pysqlite3.dbapi2")  # type: ignore[attr-defined]
    return module


class TestCandidateVersion:
    def test_absent_wheel_reports_no_candidate(self, sqlite_modules_restored: object) -> None:
        sys.modules["pysqlite3"] = None  # type: ignore[assignment]  # blocks the import
        assert shim.candidate_version() == ""

    def test_present_wheel_reports_its_bundled_version(self, sqlite_modules_restored: object) -> None:
        sys.modules["pysqlite3"] = fake_pysqlite3("3.51.1")
        assert shim.candidate_version() == "3.51.1"

    def test_asking_does_not_install(self, sqlite_modules_restored: object) -> None:
        """Reading the candidate must not change the engine — that is the policy's call."""
        import sqlite3

        sys.modules["pysqlite3"] = fake_pysqlite3("9.9.9")
        shim.candidate_version()
        assert sys.modules["sqlite3"] is sqlite3


class TestInstall:
    def test_install_makes_import_sqlite3_resolve_to_the_wheel(self, sqlite_modules_restored: object) -> None:
        candidate = fake_pysqlite3("9.9.9")
        sys.modules["pysqlite3"] = candidate

        assert shim.install() is True

        assert sys.modules["sqlite3"] is candidate
        assert sys.modules["sqlite3.dbapi2"] is candidate.dbapi2
        assert getattr(candidate, shim.ACTIVE_FLAG) is True

    def test_install_without_the_wheel_is_a_no_op(self, sqlite_modules_restored: object) -> None:
        import sqlite3

        sys.modules["pysqlite3"] = None  # type: ignore[assignment]
        assert shim.install() is False
        assert sys.modules["sqlite3"] is sqlite3


class TestEvictActiveSwap:
    def test_a_swap_this_process_made_is_undone(self, sqlite_modules_restored: object) -> None:
        candidate = fake_pysqlite3("3.51.1")
        sys.modules["pysqlite3"] = candidate
        shim.install()

        assert shim.evict_active_swap() is True

        restored = sys.modules["sqlite3"]
        assert restored is not candidate, "the stdlib must be back under its own name"
        assert restored.sqlite_version != "3.51.1"
        assert sys.modules["sqlite3.dbapi2"] is restored.dbapi2
        assert getattr(candidate, shim.ACTIVE_FLAG) is False

    def test_an_unswapped_stdlib_is_left_alone(self, sqlite_modules_restored: object) -> None:
        """Eviction must never remove a module the shim did not install."""
        import sqlite3

        assert shim.evict_active_swap() is False
        assert sys.modules["sqlite3"] is sqlite3

    def test_eviction_is_idempotent(self, sqlite_modules_restored: object) -> None:
        sys.modules["pysqlite3"] = fake_pysqlite3("3.51.1")
        shim.install()
        assert shim.evict_active_swap() is True
        assert shim.evict_active_swap() is False


def test_only_the_shim_imports_pysqlite3_and_only_the_policy_imports_the_shim() -> None:
    """The removal recipe in the shim's docstring depends on both halves.

    Keyed on IMPORTS, not on text: a comment or docstring discussing the wheel is
    documentation and costs nothing to keep, and so is a log-format string that
    happens to contain the word. What must stay confined is the dependency —
    otherwise deleting the shim stops being one mechanical commit and the
    docstring becomes a false promise.

    ``__init__.py`` is on the first list because it performs the PROVISIONAL swap
    that must run before any submodule loads; the removal recipe names it.
    """
    import ast
    from pathlib import Path

    package_root = Path(shim.__file__).resolve().parents[1]
    imports_wheel: list[str] = []
    imports_shim: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(package_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(alias.name for alias in node.names)]
            if any(name == "pysqlite3" or name.startswith("pysqlite3.") for name in names):
                imports_wheel.append(rel)
            if any("_pysqlite3_shim" in name for name in names):
                imports_shim.append(rel)

    assert sorted(set(imports_wheel)) == ["__init__.py", "storage/_pysqlite3_shim.py"], (
        "only the shim and the provisional swap in the package __init__ may import pysqlite3; "
        f"found {sorted(set(imports_wheel))}"
    )
    assert sorted(set(imports_shim)) == ["storage/_dbapi.py"], (
        "only the engine policy may import the shim, or its removal touches more files than "
        f"the docstring promises; found {sorted(set(imports_shim))}"
    )


def test_the_shim_documents_how_to_remove_itself() -> None:
    """A deliberately temporary module must say what retiring it costs."""
    assert shim.__doc__ is not None
    assert "How to remove this shim" in shim.__doc__
    assert "[sqlite-fix]" in shim.__doc__
