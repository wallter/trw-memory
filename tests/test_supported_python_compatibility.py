"""Compatibility guards for every supported Python runtime."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "trw_memory"


def test_runtime_typed_dicts_use_pydantic_compatible_backport() -> None:
    """Python 3.10/3.11 require typing_extensions for Pydantic TypedDict schemas."""
    offenders: list[str] = []
    for path in _PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "typing":
                continue
            if any(alias.name == "TypedDict" for alias in node.names):
                offenders.append(str(path.relative_to(_PACKAGE_ROOT)))
    assert offenders == [], f"runtime modules import typing.TypedDict: {sorted(offenders)}"


def test_clean_library_import_does_not_emit_structlog_debug_output(tmp_path: Path) -> None:
    """An unconfigured embedding process stays silent until it owns logging."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import structlog, trw_memory; structlog.get_logger('trw_memory.smoke').debug('must-not-print')",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "must-not-print" not in proc.stdout
    assert "must-not-print" not in proc.stderr
