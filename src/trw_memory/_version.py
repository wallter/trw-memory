"""Package version: the source tree's own ``pyproject.toml`` when running from source, else the installed
metadata (B71-111).

An editable install's dist-info records the version it was installed at, and a later bump in the source
leaves it stale (a 4.0 source reporting 3.0). The daemon's major-version gate then paired a 4.x client with
a 3.x daemon, which failed on the first tool call. Wheels ship no ``pyproject.toml`` and read their
metadata, which is exact for them. trw-mcp resolves its own version the same way (``trw_mcp.__version__``).
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _resolve() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        found = re.search(r'^version = "([^"\n]+)"', text, re.MULTILINE)
        if found and re.search(r'^name = "trw-memory"$', text, re.MULTILINE):
            return found.group(1)
    try:
        return version("trw-memory")
    except PackageNotFoundError:  # pragma: no cover - a source tree with neither
        return "unknown"


__version__: str = _resolve()
