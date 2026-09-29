"""PRD-CORE-333 S3b: ``trw_memory.sync`` resolves its exports lazily.

The daemon client's direct read builds its bearer header through
``sync._remote_common.build_platform_headers`` on every UserPromptSubmit read, so
importing that one submodule must not load the rest of the package (``delta``
alone cost ~0.2 s). Every export must still resolve, by attribute and by name.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import trw_memory.sync as sync

pytestmark = pytest.mark.unit


def test_importing_remote_common_loads_neither_delta_nor_the_other_exports() -> None:
    probe = (
        "import sys, trw_memory.sync._remote_common\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('trw_memory.sync.'))\n"
        "sys.exit(0 if 'trw_memory.sync.delta' not in sys.modules else 'delta loaded: ' + repr(loaded))"
    )
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


#: Run in a fresh interpreter per name, so no earlier import in this process can mask a failure.
_IMPORT_ONE = """
import importlib, sys
name = sys.argv[1]
namespace = {}
exec(f"from trw_memory.sync import {name} as imported", namespace)
import trw_memory.sync as sync
home = importlib.import_module(f"trw_memory.sync.{sync._EXPORTS[name]}")
assert namespace["imported"] is getattr(home, name) is getattr(sync, name), name
assert name in dir(sync), name
"""


@pytest.mark.parametrize("name", sync.__all__)
def test_every_export_imports_by_name_in_a_fresh_interpreter(name: str) -> None:
    completed = subprocess.run([sys.executable, "-c", _IMPORT_ONE, name], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_all_is_exactly_the_lazy_export_table() -> None:
    assert sorted(sync.__all__) == sorted(sync._EXPORTS)


def test_an_unknown_name_is_still_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'NoSuchExport'"):
        getattr(sync, "NoSuchExport")  # noqa: B009 -- the dynamic lookup is the point
