"""Concurrency tests for the warm-tier JSONL sidecar (memory-lifecycle-6).

The sidecar read-modify-write in ``WarmTierStore._warm_sidecar_upsert`` /
``purge_sidecar_entry`` is now guarded by an advisory lock so concurrent upserts
(and upsert-vs-purge) cannot lose each other's writes.
"""

from __future__ import annotations

import threading
from pathlib import Path

from trw_memory.lifecycle.tiers._warm import WarmTierStore


def _entry(entry_id: str) -> dict[str, object]:
    return {"id": entry_id, "content": f"content for {entry_id}", "tags": []}


class TestWarmSidecarConcurrency:
    def test_concurrent_upserts_lose_no_writes(self, tmp_path: Path) -> None:
        """Many threads upserting distinct ids concurrently — every row survives.

        Without serialization the read-modify-write update path (triggered once
        an entry already exists) would read a stale snapshot and clobber rows
        written by other threads, dropping entries. With the lock all writes are
        retained.
        """
        store = WarmTierStore(tmp_path)
        # Seed one row so every subsequent first-time write for a NEW id appends,
        # and re-writes of an existing id take the read-modify-write branch.
        store.warm_add("seed", _entry("seed"), None)

        ids = [f"e{i}" for i in range(60)]
        errors: list[Exception] = []

        def worker(entry_id: str) -> None:
            try:
                # First write (append path) then an update (RMW path), both of
                # which contend on the same sidecar file.
                store.warm_add(entry_id, _entry(entry_id), None)
                store.warm_add(entry_id, {"id": entry_id, "content": "updated", "tags": []}, None)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(eid,)) for eid in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"errors during concurrent upserts: {errors}"

        persisted = {str(e.get("id")) for e in store.warm_entries()}
        missing = set(ids) - persisted
        assert not missing, f"concurrent upserts lost rows: {sorted(missing)}"
        assert "seed" in persisted
        store.close()

    def test_concurrent_upsert_and_purge_do_not_corrupt(self, tmp_path: Path) -> None:
        """Interleaved upsert + purge on the shared sidecar stay consistent.

        The surviving rows must always be a subset of the ids we upserted and
        the file must remain parseable (no torn/lost-write corruption).
        """
        store = WarmTierStore(tmp_path)
        ids = [f"e{i}" for i in range(40)]
        for eid in ids:
            store.warm_add(eid, _entry(eid), None)

        errors: list[Exception] = []

        def upserter(eid: str) -> None:
            try:
                for _ in range(5):
                    store.warm_add(eid, {"id": eid, "content": "v", "tags": []}, None)
            except Exception as exc:
                errors.append(exc)

        def purger(eid: str) -> None:
            try:
                for _ in range(5):
                    store.purge_sidecar_entry(eid)
            except Exception as exc:
                errors.append(exc)

        threads: list[threading.Thread] = []
        for eid in ids:
            threads.append(threading.Thread(target=upserter, args=(eid,)))
            threads.append(threading.Thread(target=purger, args=(eid,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"errors during interleaved upsert/purge: {errors}"
        # File must still be parseable and contain only known ids (no garbage).
        persisted = {str(e.get("id")) for e in store.warm_entries()}
        assert persisted <= set(ids), f"unexpected ids after race: {persisted - set(ids)}"
        store.close()
