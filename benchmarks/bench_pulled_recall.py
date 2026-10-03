"""SHARED-RECALL-LOCAL O7: do learnings pulled from another host rank in local recall?

A store holds learnings pulled by team sync (written through ``memory_sync_apply``,
as a pull writes them) beside this host's own rows. Each pulled target has a
paraphrased probe that shares few words with it, and a local distractor that
shares the probe's surface words but answers a different question. The score is
recall@1 and recall@3 of the pulled target for its probe, through the daemon's
``memory_recall`` path, for two arms:

* ``embedded``: pulled rows are encoded as they land (the SHARED-RECALL-LOCAL fix);
* ``bare``: pulled rows land without a vector (the behaviour before it).

No network: the encoder is this machine's cached model (``HF_HUB_OFFLINE``), or an
injected one. Run: ``python -m benchmarks.bench_pulled_recall`` from ``trw-memory/``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NAMESPACE = "project:bench-0000000a"

#: (pulled target summary, its detail, paraphrased probe, local distractor sharing the probe's words)
TOPICS: tuple[tuple[str, str, str, str], ...] = (
    (
        "Pin the wheel cache key to the lockfile hash",
        "Stale wheels were reused after a dependency bump because the cache key ignored uv.lock.",
        "why did CI install old package builds after we upgraded a dependency",
        "Upgrade the CI runner image when a dependency needs a newer glibc",
    ),
    (
        "SQLite WAL file grows without bound when a reader never closes",
        "A long-lived read transaction blocks checkpointing; close readers or run wal_checkpoint(TRUNCATE).",
        "database journal on disk keeps getting bigger and never shrinks",
        "The database disk keeps a nightly backup for seven days",
    ),
    (
        "Retry S3 uploads with jittered exponential backoff",
        "Fixed-interval retries synchronized across workers and amplified throttling (SlowDown).",
        "object storage writes fail under load with throttling errors",
        "Object storage bucket names must be globally unique",
    ),
    (
        "Set a statement timeout on every Postgres connection pool",
        "One runaway analytical query held locks for 40 minutes and starved the API.",
        "a slow query locked tables and took the API down",
        "Slow API responses on cold start come from lazy imports",
    ),
    (
        "Never log request bodies at INFO; they carry tokens",
        "Bearer tokens appeared in shipped logs because the middleware logged raw bodies.",
        "credentials leaked into our log files",
        "Log files rotate daily and keep fourteen days",
    ),
    (
        "Use monotonic clocks for timeouts, not wall time",
        "An NTP step moved time.time() backwards and a deadline never fired.",
        "deadline never triggered after the system clock was adjusted",
        "The system clock widget shows the operator's time zone",
    ),
    (
        "Close httpx clients or they leak sockets",
        "Creating a client per request without closing it exhausted file descriptors after a day.",
        "too many open files error after the service runs for hours",
        "Open files in the editor are restored after a restart",
    ),
    (
        "Docker layer cache breaks when COPY comes before dependency install",
        "Copying the whole source tree first invalidated the pip layer on every commit.",
        "container image rebuilds reinstall all python packages each time",
        "Container images are scanned for CVEs before release",
    ),
    (
        "Guard async tasks with a reference or they are garbage collected",
        "asyncio.create_task results that nobody held were collected mid-flight and silently vanished.",
        "background coroutine sometimes just disappears without an error",
        "Background colors in the dashboard follow the system theme",
    ),
    (
        "Rate-limit the embedding model load to one per process",
        "Each worker loaded its own copy of the model and the host ran out of memory.",
        "machine ran out of RAM when several workers started at once",
        "Workers start in alphabetical order of their queue names",
    ),
    (
        "A git worktree shares the stash with the main checkout",
        "A bare git stash pop in one worktree applied another session's changes.",
        "my saved uncommitted changes showed up in a different working copy",
        "Uncommitted changes are listed by git status",
    ),
    (
        "Pydantic v2 needs populate_by_name for aliased fields",
        "Constructing the model by field name silently ignored the value when an alias was set.",
        "model ignores the value I pass by its attribute name",
        "Model names in the registry use kebab case",
    ),
    (
        "Feature flags must default off in the shipped config",
        "A flag defaulting on enabled telemetry upload on fresh installs before consent.",
        "fresh install started sending data before the user agreed",
        "Fresh installs print a welcome banner once",
    ),
    (
        "Bound every daemon request page to protect other tenants",
        "One caller asked for an unbounded page and the shared daemon stalled for everyone.",
        "one client's huge request froze the shared service for all users",
        "The shared service lists its users in the admin panel",
    ),
    (
        "Hash the content, not the mtime, to detect a changed file",
        "A checkout restored old mtimes and the cache served stale results.",
        "cache returned outdated output after switching branches",
        "Switching branches requires a clean working tree",
    ),
    (
        "Vector search must refuse embeddings from another model",
        "Mixing vectors from two encoders produced confident nonsense neighbours.",
        "semantic search returned unrelated results after we changed models",
        "Search results are paginated twenty per page",
    ),
    (
        "Use line-anchored matching for file markers",
        "A substring match hit a prose mention of the sentinel and deleted 705 lines.",
        "automated edit removed hundreds of lines from a document by mistake",
        "Documents are reviewed by two people before merge",
    ),
    (
        "Make cursors advance only past judged items",
        "Advancing a pull cursor past items the merge never saw dropped them permanently.",
        "some synced records were silently lost and never re-sent",
        "Synced records show a green check in the UI",
    ),
    (
        "Run tests with xdist at two workers on the laptop",
        "Eight workers starved the release train and forced reboots.",
        "parallel test runs made the computer unresponsive",
        "The computer name appears in the telemetry header",
    ),
    (
        "Disable redirects on authenticated HTTP clients",
        "A redirect forwarded the Authorization header to a third-party host.",
        "auth header was sent to a different domain",
        "Domain names are validated against the public suffix list",
    ),
    (
        "Quote shell arguments built from user input",
        "A filename containing a semicolon executed a second command.",
        "a weird file name ran an extra command on the server",
        "File names are lowercased before upload",
    ),
    (
        "Write config files atomically with rename",
        "A crash mid-write left an empty YAML file and the server refused to start.",
        "service would not boot because its settings file was blank",
        "Settings are grouped by section in the UI",
    ),
    (
        "Prefer monkeypatch over unittest.mock.patch in tests",
        "patch() context managers made the suite slower and leaked across tests.",
        "test doubles bled between test cases and slowed the run",
        "Test cases are named after the behavior they check",
    ),
    (
        "Close the SQLite connection before deleting its file on Windows",
        "Unlinking an open database raised PermissionError only on Windows runners.",
        "removing the db file fails with access denied on one OS",
        "The db file lives under the user's home directory",
    ),
    (
        "Count failed sync attempts toward backoff, not just successes",
        "Recording only successes reset backoff to healthy during an outage.",
        "retries hammered the backend while it was down",
        "The backend is deployed in two regions",
    ),
    (
        "Keep tool responses small; diagnostics go to structlog",
        "A build-check response was 65 percent diagnostic fields paid on every call.",
        "agent context filled up with debugging fields from tool output",
        "Tool output is rendered as markdown in the client",
    ),
    (
        "Escape LIKE wildcards in user search terms",
        "A search for 100% matched every row because % was passed through.",
        "searching for a percent sign returned everything",
        "Searching is case-insensitive by default",
    ),
    (
        "Pin GitHub Actions to commit SHAs, not tags",
        "A retagged action ran unreviewed code in the release workflow.",
        "release pipeline executed code nobody reviewed after an upstream change",
        "Release notes are generated from merged pull requests",
    ),
    (
        "Load the tokenizer once per process, not per call",
        "Re-reading the tokenizer on each request added 300 ms to every recall.",
        "every lookup became noticeably slower after the model upgrade",
        "Lookup tables are rebuilt nightly",
    ),
    (
        "Validate namespace grants before opening a backend",
        "An ungranted request opened another project's database before being refused.",
        "a request touched another project's data before the permission check",
        "Project data is exported as JSON lines",
    ),
)


#: Unrelated local rows written AFTER the pulled ones, so a store past recall's 1,000-row
#: recency window is measured too (on a 60-row store every row reaches the reranker).
_SUBJECTS = (
    "the billing export", "the onboarding wizard", "the nightly report", "the search index",
    "the payments webhook", "the mobile build", "the docs site", "the admin console",
    "the email digest", "the audit trail", "the CSV importer", "the theme picker",
)  # fmt: skip
_ACTIONS = (
    "renders dates in the viewer's locale", "keeps a changelog entry per release",
    "uses the shared button component", "is owned by the platform team",
    "documents its config keys in the README", "has a feature owner on call",
    "shows a spinner after 300 ms", "names files with ISO dates",
    "sorts rows by creation time", "accepts drag and drop",
    "links to the style guide", "groups items by month",
    "exposes a health endpoint", "retries once on 502",
    "keeps translations in one folder", "logs one line per job",
    "caches icons for a week", "validates email addresses on blur",
    "pages results fifty at a time", "uses rounded corners of 4 px",
    "prints a summary at the end", "follows the release calendar",
    "is behind the beta flag", "writes timestamps in UTC",
)  # fmt: skip
_PLACES = ("", " in staging", " for enterprise tenants", " on the EU cluster", " since the redesign")
FILLER: tuple[str, ...] = tuple(
    f"{subject.capitalize()} {action}{place}" for place in _PLACES for subject in _SUBJECTS for action in _ACTIONS
)


@dataclass(frozen=True)
class Arm:
    """One arm's result: the target's rank per probe (``None``: not in the top 10)."""

    embedded: bool
    rerank: bool
    filler: int
    ranks: tuple[int | None, ...]

    def recall_at(self, k: int) -> float:
        return sum(1 for rank in self.ranks if rank is not None and rank <= k) / len(self.ranks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pulled": "embedded" if self.embedded else "bare",
            "rerank": self.rerank,
            "store_rows": 2 * len(TOPICS) + self.filler,
            "n": len(self.ranks),
            "recall@1": round(self.recall_at(1), 3),
            "recall@3": round(self.recall_at(3), 3),
        }


def _pulled_row(index: int, summary: str, detail: str) -> dict[str, object]:
    from trw_memory.models.memory import MemoryEntry

    entry = MemoryEntry(id=f"T-{index:02d}", content=summary, detail=detail, namespace=NAMESPACE)
    return entry.model_copy(
        update={"source": "team_sync", "remote_id": f"R-{index:02d}", "metadata": {"origin_project": "host-a"}}
    ).model_dump(mode="json")


def build_store(store_dir: Path, *, embed_pulled: bool, filler: int) -> None:
    """Pulled targets through ``memory_sync_apply``, then filler and distractors through ``memory_store``."""
    from trw_memory.models.config import MemoryConfig
    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.tools import sync as sync_tools
    from trw_memory.tools.store import memory_store_impl

    config = MemoryConfig(storage_path=str(store_dir))
    backend = SQLiteBackend(store_dir / "memory.db")
    real_resolve = sync_tools.resolve_embedder
    try:
        if not embed_pulled:  # the pull before SHARED-RECALL-LOCAL: no encoder on the apply path
            sync_tools.resolve_embedder = lambda _cfg, *, surface: {"status": "unavailable", "reason": "bench"}  # type: ignore[assignment]
        for index, (summary, detail, _probe, _distractor) in enumerate(TOPICS):
            row = _pulled_row(index, summary, detail)
            answer = sync_tools.memory_sync_apply_impl(NAMESPACE, row, backend=backend, config=config, if_revision=None)
            if answer["status"] != "stored":
                raise RuntimeError(f"pulled row {index} was not stored: {answer}")
            if embed_pulled and not backend.vector_exists(str(row["id"]), namespace=NAMESPACE):
                raise RuntimeError(f"pulled row {index} embedding was not confirmed")
        sync_tools.resolve_embedder = real_resolve
        local = [(f"F-{i:04d}", text) for i, text in enumerate(FILLER[:filler])]
        local += [(f"L-{i:02d}", topic[3]) for i, topic in enumerate(TOPICS)]
        for entry_id, text in local:
            memory_store_impl(text, NAMESPACE, backend=backend, config=config, entry_id=entry_id, source="agent")
    finally:
        sync_tools.resolve_embedder = real_resolve
        backend.close()


def rank_targets(store_dir: Path, *, embedded: bool, rerank: bool, filler: int) -> Arm:
    """Each probe's pulled target rank in the top 10 of ``memory_recall``."""
    from trw_memory.models.config import MemoryConfig
    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.tools.recall import memory_recall_impl

    config = MemoryConfig(storage_path=str(store_dir))
    backend = SQLiteBackend(store_dir / "memory.db")
    try:
        ranks: list[int | None] = []
        for index, (_s, _d, probe, _distractor) in enumerate(TOPICS):
            result = memory_recall_impl(
                probe, NAMESPACE, backend=backend, config=config, limit=10, record_access=False, rerank=rerank
            )
            ids = [str(row["id"]) for row in result["memories"]]  # type: ignore[union-attr]
            target = f"T-{index:02d}"
            ranks.append(ids.index(target) + 1 if target in ids else None)
        return Arm(embedded, rerank, filler, tuple(ranks))
    finally:
        backend.close()


def measure(fillers: tuple[int, ...] = (0, len(FILLER))) -> list[dict[str, Any]]:
    """Every arm: pulled rows embedded or bare, at each store size, reranked or not."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the cached models only: no network
    results = []
    for filler in fillers:
        for embedded in (True, False):
            with tempfile.TemporaryDirectory(prefix="bench-pulled-") as tmp:
                build_store(Path(tmp), embed_pulled=embedded, filler=filler)
                results.extend(
                    rank_targets(Path(tmp), embedded=embedded, rerank=rerank, filler=filler).as_dict()
                    for rerank in (True, False)
                )
    return results


if __name__ == "__main__":
    for row in measure():
        print(json.dumps(row))
