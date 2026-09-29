"""B71-111: trw-memory's own version comes from its source when run from one, not a stale dist-info.

An editable install whose dist-info said 3.0.0 under a 4.0 source made the client identify as 3.x to the
daemon's major-version gate, which paired it with a 3.x daemon and failed on the first tool call (canon
TB-22). Wheels carry no pyproject.toml, so they keep reading their exact metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory import _version
from trw_memory.daemon import client as client_module


def _source_tree(root: Path, name: str, version: str) -> Path:
    module = root / "src" / "trw_memory" / "_version.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "{version}"\n', encoding="utf-8")
    return module


def test_a_source_tree_reports_its_pyproject_version_over_stale_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_version, "__file__", str(_source_tree(tmp_path, "trw-memory", "9.1.0")))
    monkeypatch.setattr(_version, "version", lambda _dist: "3.0.0")  # the stale editable dist-info

    assert _version._resolve() == "9.1.0"


@pytest.mark.parametrize("name", ["some-other-package", None])
def test_without_its_own_pyproject_it_reads_the_installed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str | None
) -> None:
    """A wheel install (no pyproject) or a pyproject naming another project: the metadata is the answer."""
    module = _source_tree(tmp_path, name or "x", "9.1.0")
    if name is None:
        (tmp_path / "pyproject.toml").unlink()
    monkeypatch.setattr(_version, "__file__", str(module))
    monkeypatch.setattr(_version, "version", lambda _dist: "4.0.1")

    assert _version._resolve() == "4.0.1"


def test_the_daemon_version_gate_identifies_by_the_package_version_not_the_dist_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata

    import trw_memory._version as version_module

    monkeypatch.setattr(importlib.metadata, "version", lambda _dist: "3.0.0")
    monkeypatch.setattr(version_module, "__version__", "4.0.1")

    assert client_module._package_version() == "4.0.1"
