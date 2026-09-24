"""EngMem record format: a time-ordered event stream plus as-of queries.

LOCOMO asks when two people discussed pottery. EngMem asks the questions an
engineering memory is actually for: what should I have known before editing this
file, did I already make this mistake, is the current convention ranked above the
one it replaced.

Everything here is public and dataset-agnostic: the schema, the replay engine and
the scorers. **The extractor that mines TRW's own git history, and every gold
label derived from it, is proprietary and lives in `trw-eval`** -- as do the
team_sync corpora, which are other projects' data and never leave. This module
must stay free of anything that only makes sense for TRW's own history.

Two invariants the whole benchmark rests on:

* **Time order is load-bearing.** Events replay in timestamp order into a fresh
  store and every query carries an ``as_of``. A learning written after the commit
  it would have helped is not evidence, it is leakage, and the replay must make
  that structurally impossible rather than filtered after the fact.
* **Git-derived labels are silver, not gold.** A commit citing a learning id is
  evidence that someone thought it relevant, not proof it was. Label precision
  gets published from a human-verified stratified sample, and every rate derived
  from these labels is reported as a lower bound.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

EventKind = Literal["learning_created", "status_changed", "merged", "commit"]
TaskId = Literal["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened, at a time. Replayed in ``at`` order, never re-sorted
    by anything else -- ties broken by ``seq`` so a same-second create/supersede
    pair cannot silently invert."""

    at: datetime
    seq: int
    kind: EventKind
    learning_id: str = ""
    content: str = ""
    detail: str = ""
    tags: tuple[str, ...] = ()
    learning_type: str = ""
    status: str = ""
    # `merged` events name what this record consolidated, so C6 can score whether a
    # system keeps the survivor and drops the absorbed rows.
    merged_from: tuple[str, ...] = ()
    # `commit` events carry the code side: what changed and which learnings the
    # message cited. Paths are how C1 asks "what should I have known before
    # editing this file".
    commit_sha: str = ""
    subject: str = ""
    paths: tuple[str, ...] = ()
    cites: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["at"] = self.at.isoformat()
        for k in ("tags", "merged_from", "paths", "cites"):
            d[k] = list(d[k])
        return d


@dataclass(frozen=True, slots=True)
class Query:
    """One retrieval the benchmark will grade.

    ``as_of`` is authoritative: the store is replayed only up to this instant, so a
    later learning cannot be retrieved even by accident. ``forbidden_ids`` are rows
    that must NOT surface -- superseded records for C3, foreign-project rows for
    C5 -- and they are what separates a memory system from a search engine, since
    surfacing a retired convention is worse than returning nothing.
    """

    qid: str
    task: TaskId
    as_of: datetime
    text: str
    gold_ids: tuple[str, ...] = ()
    forbidden_ids: tuple[str, ...] = ()
    # Free-form provenance for the label: which commit, which supersession pair.
    # Kept so a human verifying a sample can find the source without re-deriving it.
    context: dict[str, Any] = field(default_factory=dict)
    # Silver by default. A human-verified row says so, and the verified subset is
    # what label-precision is estimated from.
    label_source: Literal["git", "human", "synthetic"] = "git"

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        for k in ("gold_ids", "forbidden_ids"):
            d[k] = list(d[k])
        return d


def _parse_dt(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def read_events(path: Path) -> list[Event]:
    out: list[Event] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        d["at"] = _parse_dt(d["at"])
        for k in ("tags", "merged_from", "paths", "cites"):
            d[k] = tuple(d.get(k) or ())
        out.append(Event(**d))
    # Sort by (at, seq) once, here, so every arm replays the identical order no
    # matter how the extractor emitted it.
    out.sort(key=lambda e: (e.at, e.seq))
    return out


def read_queries(path: Path) -> list[Query]:
    out: list[Query] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        d["as_of"] = _parse_dt(d["as_of"])
        for k in ("gold_ids", "forbidden_ids"):
            d[k] = tuple(d.get(k) or ())
        out.append(Query(**d))
    return out


def write_jsonl(path: Path, rows: list[Event] | list[Query]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_json(), sort_keys=True) + "\n")


def freeze(path: Path) -> str:
    """SHA-256 of a gold file, printed with every result. A benchmark whose gold can
    change without the number changing is not a benchmark."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holdout(qid: str, fraction: float = 0.2) -> bool:
    """Stable 20% holdout keyed on the query id, so tuning on the development
    portion cannot quietly consume the held-out portion as the suite grows.
    Hash-based rather than random: adding queries never reshuffles old ones."""
    h = hashlib.sha256(qid.encode()).digest()
    return (int.from_bytes(h[:4], "big") % 10_000) < fraction * 10_000
