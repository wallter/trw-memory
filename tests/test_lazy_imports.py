"""Guard tests: importing trw_memory must not pay the torch/sentence_transformers tax.

Root cause (production feedback sub_psVs_nUWnLJGvOs3): ``trw_memory.retrieval``
eagerly re-exported ``cross_encode_rerank`` from ``reranker.py``, which imported
``sentence_transformers`` (and therefore ``torch``, ~2.5-5.6s) at module load
time.  Because the ``trw_mcp.server`` boot path imports ``trw_memory``
transitively (storage -> sync -> retrieval.dense -> retrieval.__init__), MCP
boot took ~9s and blew past Claude Code's 30s connect timeout under contention.

These tests run in a FRESH subprocess: the pytest process itself may already
have imported torch/sentence_transformers via other tests, so an in-process
``sys.modules`` check would false-positive.  A clean interpreter is the only
faithful reproduction of the client boot path.
"""

from __future__ import annotations

import subprocess
import sys


def _assert_torch_free_import(module: str) -> str:
    """Import *module* in a clean interpreter; assert no torch/ST leaked in."""
    code = (
        f"import {module}\n"
        "import sys\n"
        "forbidden = {'sentence_transformers', 'torch'}\n"
        "leaked = forbidden & set(sys.modules)\n"
        "assert not leaked, leaked\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
    return proc.stdout.strip()


def test_import_retrieval_does_not_import_torch() -> None:
    """``import trw_memory.retrieval`` stays free of sentence_transformers/torch."""
    assert _assert_torch_free_import("trw_memory.retrieval") == "OK"


def test_import_reranker_module_directly_does_not_import_torch() -> None:
    """Importing ``trw_memory.retrieval.reranker`` DIRECTLY stays torch-free.

    The package-level guards above prove ``retrieval.__init__`` defers the
    re-export; this one pins the reranker module's OWN lazy-import layer. A
    regression that moved ``import sentence_transformers`` back to module top
    would slip past the ``__init__`` guards (they never touch the submodule)
    but is caught here, because a direct submodule import bypasses the PEP 562
    ``__getattr__`` on the package.
    """
    assert _assert_torch_free_import("trw_memory.retrieval.reranker") == "OK"


def test_import_trw_memory_package_does_not_import_torch() -> None:
    """Importing the full ``trw_memory`` package stays torch-free.

    This is the exact chain the trw_mcp.server boot path traverses.
    """
    assert _assert_torch_free_import("trw_memory") == "OK"


def test_lazy_re_export_still_resolves() -> None:
    """The PEP 562 lazy re-export keeps the public import contract intact."""
    from trw_memory.retrieval import cross_encode_rerank

    assert callable(cross_encode_rerank)
    # Empty input is a no-op that never touches the model — proves the re-export
    # resolves to the real function, not a placeholder.
    assert cross_encode_rerank("q", []) == []


def test_reranker_module_getattr_exposes_availability_flag() -> None:
    """``reranker._CROSS_ENCODER_AVAILABLE`` is still readable (lazily) and boolean."""
    from trw_memory.retrieval import reranker

    assert isinstance(reranker._CROSS_ENCODER_AVAILABLE, bool)


def test_reranker_getattr_rejects_unknown_name() -> None:
    """The PEP 562 hook raises AttributeError for unknown attributes."""
    import pytest

    from trw_memory.retrieval import reranker

    with pytest.raises(AttributeError):
        _ = reranker.does_not_exist  # type: ignore[attr-defined]


def test_retrieval_getattr_rejects_unknown_name() -> None:
    """The package-level PEP 562 hook raises AttributeError for unknown names."""
    import pytest

    import trw_memory.retrieval as retrieval

    with pytest.raises(AttributeError):
        _ = retrieval.does_not_exist  # type: ignore[attr-defined]
