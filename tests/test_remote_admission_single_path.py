"""PRD-CORE-245 FR06 — one remote-ingest path, one admission gate.

The defect: content that never lands in the store is also content no later audit
or quarantine sweep can reach. Two of the three paths injected peer text
straight into the recall RESPONSE — the agent's context — without passing any
admission check.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.sync._remote_admission import SHARED_NAMESPACE, admit_remote_results

pytestmark = pytest.mark.unit

_PACKAGE = Path(__file__).resolve().parents[1]
# Monorepo checkout: the package sits at <repo>/trw-memory and trw-mcp is a sibling.
# Exported package tree (the public repo, or the release check's CI replay): the
# package root IS the repo root, and parents[2] would point above it.
_REPO = _PACKAGE.parent if (_PACKAGE.name == "trw-memory" and (_PACKAGE.parent / "trw-mcp").is_dir()) else _PACKAGE

_REMOTE_ITEM: dict[str, object] = {
    "source_learning_id": "R-remote-1",
    "summary": "a peer's learning",
    "detail": "arrived from the platform",
    "tags": ["shared"],
}


def _backend(tmp_path: Path) -> SQLiteBackend:
    return SQLiteBackend(tmp_path / "admission.db")


def test_refused_remote_item_never_reaches_the_response(tmp_path: Path) -> None:
    """A gate-refused item is absent from the returned list and present in quarantine."""
    backend = _backend(tmp_path)
    cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
    try:
        refusal = MagicMock()
        refusal.quarantined = True
        refusal.entry = MagicMock()
        with (
            patch("trw_memory.security.runtime.prepare_entry_for_store", return_value=refusal),
            patch("trw_memory.security.runtime.store_quarantined_entry") as quarantine,
        ):
            outcome = admit_remote_results([dict(_REMOTE_ITEM)], config=cfg, backend=backend)

        # W13: the refusal is a COUNT the caller receives, not only a log line —
        # "the peer sent nothing" and "the gate refused everything" were the
        # same empty list before.
        assert (outcome.admitted, outcome.refused, outcome.gate_errors) == ([], 1, 0)
        quarantine.assert_called_once()
    finally:
        backend.close()


def test_a_gate_error_is_a_refusal_not_a_pass(tmp_path: Path) -> None:
    """NFR03: an item the gate could not evaluate is dropped, never returned."""
    backend = _backend(tmp_path)
    cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
    try:
        with patch(
            "trw_memory.security.runtime.prepare_entry_for_store",
            side_effect=RuntimeError("gate exploded"),
        ):
            outcome = admit_remote_results([dict(_REMOTE_ITEM)], config=cfg, backend=backend)
            # A gate that could not be applied is counted apart from a gate that
            # said no, though both fail closed.
            assert (outcome.admitted, outcome.refused, outcome.gate_errors) == ([], 1, 1)
    finally:
        backend.close()


def test_an_admitted_item_is_returned_unchanged(tmp_path: Path) -> None:
    """Control: the gate must not turn ordinary shared results into an empty response."""
    backend = _backend(tmp_path)
    cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
    try:
        passed = MagicMock()
        passed.quarantined = False
        with patch("trw_memory.security.runtime.prepare_entry_for_store", return_value=passed):
            outcome = admit_remote_results([dict(_REMOTE_ITEM)], config=cfg, backend=backend)
        assert (outcome.admitted, outcome.refused) == ([dict(_REMOTE_ITEM)], 0)
    finally:
        backend.close()


def test_shared_content_lands_in_a_real_namespace_not_a_carve_out() -> None:
    """FR06: ``shared`` is a namespace a caller must hold in its scope, like any other."""
    from trw_memory.security.namespace_scope import authorize_namespaces
    from trw_memory.security.rbac import Permission

    scope = authorize_namespaces(MemoryConfig(), [SHARED_NAMESPACE], Permission.READ, "recall")
    assert SHARED_NAMESPACE in scope


def test_fetch_requires_a_backend_so_it_cannot_run_ungated() -> None:
    """The gate needs a backend, so a fetch that cannot be gated does not happen."""
    import inspect

    from trw_memory.sync import fetch_shared_memories

    parameter = inspect.signature(fetch_shared_memories).parameters["backend"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_telemetry_package_has_no_dangling_reference() -> None:
    """FR06 census: importing ``trw_mcp.telemetry`` succeeds after the module deletion."""
    telemetry = pytest.importorskip("trw_mcp.telemetry")
    assert "fetch_shared_learnings" not in telemetry.__all__
    assert not hasattr(telemetry, "fetch_shared_learnings")
    assert not (_REPO / "trw-mcp" / "src" / "trw_mcp" / "telemetry" / "remote_recall.py").exists()
    assert not (_REPO / "trw-mcp" / "tests" / "test_telemetry_remote_recall.py").exists()


def test_exactly_one_client_posts_to_the_learnings_search_endpoint() -> None:
    """The whole point of FR06: one path to the platform, therefore one gate."""
    # Monorepo layout scans both packages; the public trw-memory repository is the
    # package alone, where the same single client lives at src/. A filesystem scan
    # rather than `git grep`, so an exported non-git copy of the tree (the release
    # check's replay of the public CI) answers the same as a checkout.
    monorepo = (_REPO / "trw-memory").is_dir()
    scan_roots = [_REPO / "trw-memory" / "src", _REPO / "trw-mcp" / "src"] if monorepo else [_REPO / "src"]
    expected = [
        "trw-memory/src/trw_memory/sync/_remote_fetch.py" if monorepo else "src/trw_memory/sync/_remote_fetch.py"
    ]
    hits = sorted(
        path.relative_to(_REPO).as_posix()
        for root in scan_roots
        if root.is_dir()
        for path in root.rglob("*.py")
        if "v1/learnings/search" in path.read_text(encoding="utf-8", errors="ignore")
    )
    assert hits == expected, f"expected exactly one client for the platform learning-search endpoint, found {hits}"
