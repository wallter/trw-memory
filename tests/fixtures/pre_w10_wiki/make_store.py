"""Regenerate ``memory.db`` + ``receipt.json``: a genuine pre-W10 wiki_refs store.

Run from a monorepo checkout (needs the historical commit):
``python tests/fixtures/pre_w10_wiki/make_store.py``. The output is committed, so
the migration tests run anywhere -- the public mirror and a copy without
``.git`` included -- without git history.

A synthetic fixture (hand-rolled ``CREATE TABLE`` + ``INSERT`` statements
approximating the old shape) proves only that *this test's guess* about the
old schema migrates cleanly, not that the real thing does. This module
instead extracts the real ``trw-memory/src`` tree from a pre-W10 commit via
``git archive``, runs a small builder script against it **in a subprocess**
(so its imports of ``trw_memory.storage.sqlite_backend`` and
``trw_memory.wiki.models`` never collide with the current build already
imported in the test process), and lets the OLD ``SQLiteBackend.store()`` and
``WikiPage.to_memory_metadata()`` produce the database bytes: real
``wiki_refs`` rows via the real ``replace_wiki_refs_for_entry`` side effect,
both real indexes from the real bootstrap DDL, and real ``wiki.page`` /
``wiki.slug`` / ``wiki.kind`` metadata keys.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

#: The last commit before W10 deleted trw_memory.wiki and its schema-6 shape
#: (SCHEMA_VERSION == 6 there; schema 5's namespace-boundary rebuild is what
#: first gave wiki_refs its current composite-key shape).
PRE_W10_REF = "d44b3461c1"

_BUILDER_SCRIPT = """
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.wiki.models import WikiPage, WikiReference

db_path = Path(sys.argv[2])
count = int(sys.argv[3])

backend = SQLiteBackend(db_path)
ids = []
try:
    for index in range(count):
        page = WikiPage(
            kind="topic",
            slug=f"topic/legacy-{index}",
            title=f"Legacy {index}",
            outbound_refs=[
                WikiReference(target_slug="topic/legacy-target", ref_type="related"),
                WikiReference(target_slug="topic/legacy-backlink", ref_type="related", bidirectional=False),
            ],
        )
        entry_id = f"M-wiki-legacy-{index:04d}"
        entry = MemoryEntry(
            id=entry_id,
            content=f"legacy entry {index}",
            namespace="project:legacy",
            metadata=page.to_memory_metadata(),
        )
        backend.store(entry)
        ids.append(entry_id)
    version = int(backend._conn.execute("PRAGMA user_version").fetchone()[0])
    wiki_rows = int(backend._conn.execute("SELECT COUNT(*) FROM wiki_refs").fetchone()[0])
    indexes = sorted(
        str(r[0])
        for r in backend._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_wiki_refs_%'"
        ).fetchall()
    )
finally:
    backend.close()

print(json.dumps({"user_version": version, "wiki_rows": wiki_rows, "indexes": indexes, "ids": ids}))
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
    """Build the store with the actual historical ``SQLiteBackend``/``WikiPage`` code, in a subprocess.

    The subprocess keeps the old build's imports from colliding with the current
    one; ``wiki_refs`` (data + both indexes) and each entry's ``wiki.*`` metadata
    come from the real pre-retirement implementation, not an approximation.
    """
    here = Path(__file__).parent
    db_path = here / "memory.db"
    db_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="w10-historical-src-") as scratch:
        old_src = _extract_historical_src(Path(scratch), PRE_W10_REF)
        result = subprocess.run(
            [sys.executable, "-c", _BUILDER_SCRIPT, str(old_src), str(db_path), "3"],
            capture_output=True,
            text=True,
            check=True,
        )
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    (here / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
