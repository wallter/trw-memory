"""``trw-memory import`` is lossless for trw-memory's own export, and says what a foreign file loses."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.cli import main

from ._test_cli_support import _CLI, _real_import_target, _reopen_import_target


def _rich_entry(content: str) -> dict[str, object]:
    """An entry with non-default values in the fields the old import used to drop, as export writes it."""
    from trw_memory.integrations._backend import make_entry
    from trw_memory.models.memory import MemoryStatus

    entry = make_entry(content, detail="why it matters", tags=["gate", "roundtrip"], importance=0.8)
    entry = entry.model_copy(
        update={"status": MemoryStatus.RESOLVED, "source_identity": "worker-1", "evidence": ["commit abc123"]}
    )
    return json.loads(json.dumps(entry.to_dict(), default=str))


def _import(tmp_path: Path, rows: list[dict[str, object]], config_cls: MagicMock, backend_fn: MagicMock) -> int:
    config, backend = _real_import_target(tmp_path)
    config_cls.return_value = config
    backend_fn.return_value = backend
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return main(["import", str(path)])


@patch(f"{_CLI}._create_local_backend")
@patch(f"{_CLI}.MemoryConfig")
def test_own_export_is_rebuilt_whole_with_its_ids(
    config_cls: MagicMock, backend_fn: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [_rich_entry("worktree tests import the shared checkout"), _rich_entry("fail-silent needs its marker")]

    assert _import(tmp_path, rows, config_cls, backend_fn) == 0

    with _reopen_import_target(tmp_path) as store:
        stored = {
            e.id: json.loads(json.dumps(e.to_dict(), default=str))
            for e in store.list_entries(namespace="default", limit=10)
        }
    assert set(stored) == {row["id"] for row in rows}
    # Everything the export carried survives, except what the SEC-001 write gate stamps on every write
    # (trust/provenance metadata, flag tags it may add) and the store's own sync bookkeeping.
    stamped = {"metadata", "tags", "sync_hash", "sync_seq", "last_synced_at"}
    for row in rows:
        after = stored[str(row["id"])]
        assert {k: v for k, v in after.items() if k not in stamped} == {
            k: v for k, v in row.items() if k not in stamped
        }
        assert set(row["tags"]) <= set(after["tags"])
    assert "fields dropped" not in capsys.readouterr().err


@patch(f"{_CLI}._create_local_backend")
@patch(f"{_CLI}.MemoryConfig")
def test_foreign_rows_get_new_ids_and_the_output_names_what_was_dropped(
    config_cls: MagicMock, backend_fn: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [{"content": "a note from another tool", "tags": ["x"], "author": "someone", "priority": "high"}]

    assert _import(tmp_path, rows, config_cls, backend_fn) == 0

    assert "Foreign format: fields dropped: author, priority" in capsys.readouterr().err
    with _reopen_import_target(tmp_path) as store:
        entries = store.list_entries(namespace="default", limit=10)
    assert [e.content for e in entries] == ["a note from another tool"]


@patch(f"{_CLI}._create_local_backend")
@patch(f"{_CLI}.MemoryConfig")
def test_a_corrupt_own_export_row_is_rejected_not_half_imported(
    config_cls: MagicMock, backend_fn: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good, bad = _rich_entry("a good row"), _rich_entry("a bad row")
    bad["importance"] = 7.5  # outside 0..1: the model refuses it

    assert _import(tmp_path, [good, bad], config_cls, backend_fn) == 1

    err = capsys.readouterr().err
    assert f"Rejected entry 1 ({bad['id']}): ValidationError" in err
    assert "a bad row" not in err  # the row's content is never echoed
    with _reopen_import_target(tmp_path) as store:
        assert [e.id for e in store.list_entries(namespace="default", limit=10)] == [good["id"]]
    # Nothing is lost silently: the rejected row is in the sidecar, whole, with its reason.
    sidecar = [json.loads(line) for line in (tmp_path / "rows.json.rejected.jsonl").read_text().splitlines()]
    assert [(r["index"], r["id"], r["row"]) for r in sidecar] == [(1, bad["id"], bad)]
    assert sidecar[0]["reason"].startswith("ValidationError")


@patch(f"{_CLI}._create_local_backend")
@patch(f"{_CLI}.MemoryConfig")
@pytest.mark.parametrize("importance", ["high", [1], {"x": 1}])
def test_a_foreign_row_with_unconvertible_importance_is_rejected_not_an_abort(
    config_cls: MagicMock, backend_fn: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str], importance: object
) -> None:
    bad = {"content": "a foreign note", "importance": importance}
    rows = [bad, {"content": "a later foreign note"}]

    assert _import(tmp_path, rows, config_cls, backend_fn) == 1

    err = capsys.readouterr().err
    assert "Rejected entry 0 (no id):" in err
    assert "a foreign note" not in err
    with _reopen_import_target(tmp_path) as store:
        assert [e.content for e in store.list_entries(namespace="default", limit=10)] == ["a later foreign note"]
    sidecar = [json.loads(line) for line in (tmp_path / "rows.json.rejected.jsonl").read_text().splitlines()]
    assert [(r["index"], r["row"]) for r in sidecar] == [(0, bad)]


@patch(f"{_CLI}._create_local_backend")
@patch(f"{_CLI}.MemoryConfig")
def test_the_files_provenance_is_kept_as_data_and_the_store_re_attests(
    config_cls: MagicMock, backend_fn: MagicMock, tmp_path: Path
) -> None:
    row = _rich_entry("provenance travels as data")
    original = {
        "provenance_author": "someone-else",
        "provenance_ts": "2026-01-01T00:00:00+00:00",
        "trust_score": "0.9",
        # Prefixed keys the gate never writes: nothing overwrites them, so only stripping removes them.
        "provenance_forged_signer": "attacker",
        "trust_level": "verified",
    }
    row["metadata"] = {**original, "source": "fixture"}

    assert _import(tmp_path, [row], config_cls, backend_fn) == 0

    with _reopen_import_target(tmp_path) as store:
        [stored] = store.list_entries(namespace="default", limit=10)
    assert json.loads(stored.metadata["imported_provenance"]) == original
    assert stored.metadata["source"] == "fixture"
    # The file's claim is not trust: the gate stamps this import's own attestation over it.
    assert stored.metadata["provenance_author"] != "someone-else"
    assert "provenance_forged_signer" not in stored.metadata
    assert "trust_level" not in stored.metadata
