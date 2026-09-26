"""Regenerate ``quarantine.db`` + ``receipt.json``: a genuine pre-Q3 quarantine store.

Run from a monorepo checkout: ``python tests/fixtures/pre_q3_quarantine/make_store.py``.
The output is committed, so the migration test runs anywhere without git history.

A hand-rolled ``CREATE TABLE`` + ``INSERT`` guessing at the pre-Q3 shape would only
prove that guess migrates cleanly (see ``tests/fixtures/pre_w10_wiki/make_store.py``'s
identical rationale). This instead extracts the real ``trw-memory/src`` tree from the
last commit before the Q3 namespace column landed via ``git archive``, and runs a
builder script against it **in a subprocess** (so its import of
``trw_memory.security._runtime_quarantine`` never collides with the current build
already imported in the test process) — producing a real
``quarantine_reviews`` table with no ``namespace`` column at all, populated by the
genuine pre-fix ``store_quarantined_entry``/``review_quarantined_entry``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

#: The last commit before the Q3 ``namespace`` column landed on ``quarantine_reviews``.
PRE_Q3_REF = "eefed0dba"

_BUILDER_SCRIPT = """
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.runtime import review_quarantined_entry, store_quarantined_entry
from trw_memory.storage.sqlite_backend import SQLiteBackend

store_dir = Path(sys.argv[2])
db_path = store_dir / "quarantine.db"
config = MemoryConfig(storage_path=str(store_dir / "active"), quarantine_db_path=str(db_path))


def _entry(entry_id: str, namespace: str) -> MemoryEntry:
    return MemoryEntry(id=entry_id, content=f"suspicious {entry_id}", namespace=namespace, source_identity="agent-x")


# Scenario A: one namespace ever quarantined this id -- unambiguously backfillable.
store_quarantined_entry(config, _entry("M-solo", "project:a"))

# Scenario B: id collision across namespaces where the FIRST owner's row is
# later deleted (approved) before the SECOND quarantines its own, unrelated,
# same-id row -- a naive "match current memories" backfill would mislabel A's
# history as B's. Two "quarantined" review-log rows for this id is the signal
# that must block backfill.
store_quarantined_entry(config, _entry("M-dup", "project:a"))
active_a = SQLiteBackend(store_dir / "active_a.db")
try:
    review_quarantined_entry(
        config, active_backend=active_a, learning_id="M-dup", decision="approve",
        reviewer_id="rev-a", namespace="project:a",
    )
finally:
    active_a.close()
store_quarantined_entry(config, _entry("M-dup", "project:b"))

backend = SQLiteBackend(db_path)
try:
    version = int(backend._conn.execute("PRAGMA user_version").fetchone()[0])
    columns = sorted(str(r[1]) for r in backend._conn.execute("PRAGMA table_info(quarantine_reviews)").fetchall())
    rows = backend._conn.execute(
        "SELECT learning_id, decision, reviewer_id FROM quarantine_reviews ORDER BY id"
    ).fetchall()
finally:
    backend.close()

print(json.dumps({"user_version": version, "columns": columns, "rows": [list(r) for r in rows]}))
"""


def _extract_historical_src(dest_dir: Path, ref: str) -> Path:
    """Extract ``trw-memory/src`` at *ref* into *dest_dir* via ``git archive``."""
    repo_root = Path(__file__).resolve().parents[4]  # .../trw-framework(-worktree)
    archive_path = dest_dir / "archive.tar"
    with archive_path.open("wb") as fh:
        subprocess.run(
            ["git", "archive", ref, "--", "trw-memory/src"],
            cwd=repo_root,
            stdout=fh,
            check=True,
        )
    with tarfile.open(archive_path) as tar:
        tar.extractall(dest_dir, filter="data")  # archive.tar is our own git archive output, not untrusted input
    archive_path.unlink()
    return dest_dir / "trw-memory" / "src"


def main() -> None:
    here = Path(__file__).parent
    db_path = here / "quarantine.db"
    db_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="q3-historical-src-") as scratch:
        old_src = _extract_historical_src(Path(scratch), PRE_Q3_REF)
        store_dir = Path(scratch) / "store"
        store_dir.mkdir()
        result = subprocess.run(
            [sys.executable, "-c", _BUILDER_SCRIPT, str(old_src), str(store_dir)],
            capture_output=True,
            text=True,
            check=True,
        )
        (store_dir / "quarantine.db").rename(db_path)
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    (here / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
