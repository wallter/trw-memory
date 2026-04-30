"""Graph importance and cross-validation tests."""

from __future__ import annotations

import multiprocessing
import threading
from pathlib import Path

from trw_memory.graph import (
    IMPORTANCE_BOOST,
    _merge_cross_validated_entry,
    apply_importance_boost,
    apply_importance_decay,
    detect_cross_validation,
)
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig

from ._test_graph_support import _V1, _V3, _make_conn, _make_entry, _merge_cross_validation_in_subprocess


class TestApplyImportanceBoost:
    def test_adds_default_boost_and_records_history(self) -> None:
        entry = _make_entry("e1", importance=0.5)
        result = apply_importance_boost(entry)

        assert abs(result.importance - (0.5 + IMPORTANCE_BOOST)) < 0.001
        assert len(result.outcome_history) == 1
        assert "importance_boost" in result.outcome_history[0]
        assert f"delta=+{IMPORTANCE_BOOST:.2f}" in result.outcome_history[0]

    def test_caps_at_1_0(self) -> None:
        entry = _make_entry("e1", importance=0.99)
        result = apply_importance_boost(entry, delta=0.1)

        assert result.importance == 1.0

    def test_sets_cross_validated_true(self) -> None:
        entry = _make_entry("e1", cross_validated=False)
        result = apply_importance_boost(entry)

        assert result.cross_validated is True

    def test_preserves_existing_outcome_history(self) -> None:
        entry = _make_entry("e1", outcome_history=["previous_event"])
        result = apply_importance_boost(entry)

        assert len(result.outcome_history) == 2
        assert result.outcome_history[0] == "previous_event"
        assert "importance_boost" in result.outcome_history[1]

    def test_concurrent_cross_validation_merges_both_project_boosts(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

        with create_backend_from_config(cfg, "project:default") as storage:
            backend = storage
            backend.store(_make_entry("e1", importance=0.5))

            threads = [
                threading.Thread(
                    target=_merge_cross_validated_entry,
                    args=(backend, "e1", project_id, 0.97),
                )
                for project_id in ("project-a", "project-b")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            updated = backend.get("e1")
            assert updated is not None
            assert updated.importance == 0.6
            assert updated.cross_validated is True
            assert sum("importance_boost" in item for item in updated.outcome_history) == 2
            assert sum("cross_validated:project_id=" in item for item in updated.outcome_history) == 2

    def test_cross_process_cross_validation_merges_both_project_boosts(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

        with create_backend_from_config(cfg, "project:default") as storage:
            storage.store(_make_entry("e1", importance=0.5))

        ctx = multiprocessing.get_context("spawn")
        processes = [
            ctx.Process(
                target=_merge_cross_validation_in_subprocess,
                args=(str(tmp_path), project_id),
            )
            for project_id in ("project-a", "project-b")
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0

        with create_backend_from_config(cfg, "project:default") as storage:
            updated = storage.get("e1")
            assert updated is not None
            assert updated.importance == 0.6
            assert updated.cross_validated is True
            assert sum("importance_boost" in item for item in updated.outcome_history) == 2
            assert sum("cross_validated:project_id=" in item for item in updated.outcome_history) == 2

    def test_twenty_sequential_boosts_cap_at_one_without_drift(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

        with create_backend_from_config(cfg, "project:default") as storage:
            storage.store(_make_entry("e1", importance=0.0))
            observed: list[float] = []

            for idx in range(20):
                updated, applied = _merge_cross_validated_entry(storage, "e1", f"project-{idx}", 0.97)
                assert applied is True
                assert updated is not None
                observed.append(updated.importance)

            assert observed[-1] == 1.0
            assert max(observed) == 1.0
            assert all(value <= 1.0 for value in observed)


class TestApplyImportanceDecay:
    def test_reduces_by_delta(self) -> None:
        entry = _make_entry("e1", importance=0.5)
        result = apply_importance_decay(entry, delta=0.1)

        assert abs(result.importance - 0.4) < 0.001
        assert len(result.outcome_history) == 1
        assert "importance_decay" in result.outcome_history[0]

    def test_floors_at_0_0(self) -> None:
        entry = _make_entry("e1", importance=0.05)
        result = apply_importance_decay(entry, delta=0.1)

        assert result.importance == 0.0


class TestDetectCrossValidation:
    def test_true_above_threshold(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        remote = [("remote-1", "proj-b", _V1)]

        assert detect_cross_validation(entry, conn, embedding=_V1, remote_entries=remote)

    def test_false_below_threshold(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        remote = [("remote-1", "proj-b", _V3)]

        assert not detect_cross_validation(entry, conn, embedding=_V1, remote_entries=remote)

    def test_false_when_no_embedding(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        remote = [("remote-1", "proj-b", _V1)]

        assert not detect_cross_validation(entry, conn, embedding=None, remote_entries=remote)

    def test_false_when_no_remote_entries(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")

        assert not detect_cross_validation(entry, conn, embedding=_V1, remote_entries=None)
