"""EngMem-Synth: a seeded synthetic engineering-memory stream with planted gold.

Exists for three reasons:

1. **It proves the harness runs with no TRW data present**, which is what keeps
   the public half of this benchmark publishable (the git-history extractor and
   its gold are proprietary).
2. **Gold is planted, not inferred**, so unlike the git-derived labels it is
   truly gold and a scorer bug shows up as an impossible number rather than a
   plausible one.
3. It generates the failure modes LOCOMO cannot: conventions that get superseded,
   the same lesson written by several people in slightly different words, and
   rows from a neighbouring project that must not leak.

Deliberately template-driven and LLM-free: generation must stay free and
deterministic under a seed, so a scale curve costs nothing but time.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from .schema import Event, Query

_AREAS = [
    ("auth", "token refresh", "src/auth/session.py"),
    ("db", "connection pooling", "src/db/pool.py"),
    ("cache", "key eviction", "src/cache/lru.py"),
    ("api", "pagination cursors", "src/api/list.py"),
    ("build", "artifact caching", "tools/build.py"),
    ("queue", "visibility timeout", "src/queue/worker.py"),
    ("search", "analyzer config", "src/search/index.py"),
    ("billing", "proration rounding", "src/billing/invoice.py"),
]
_LESSON = "When touching {what} in {path}, {rule} -- we lost a day to this."
_RULES = [
    "always drain in-flight work before swapping the handle",
    "never assume the clock is monotonic across processes",
    "treat a partial write as a failure, not a retry",
    "the cap is per-namespace, not global",
]
_SUPERSEDED = "Older guidance for {what}: {rule}. Replaced after the {year} incident."


def generate(
    *,
    seed: int = 7,
    n_areas: int = 8,
    distractors: int = 200,
    near_dup_rate: float = 0.05,
    foreign_rate: float = 0.1,
) -> tuple[list[Event], list[Query], dict[str, str]]:
    """Return (events, queries, successor_of).

    ``distractors`` sets corpus size, so the same call shape produces the scale
    curve the design asks for (10^3 .. 10^7) by raising one number.
    """
    # S311: reproducible synthetic corpus generation, not a security context --
    # the seed IS the point, so a CSPRNG would defeat it.
    rng = random.Random(seed)  # noqa: S311
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events: list[Event] = []
    queries: list[Query] = []
    successor_of: dict[str, str] = {}
    seq = 0

    def add(ev_kwargs: dict) -> Event:
        nonlocal seq
        seq += 1
        ev = Event(seq=seq, **ev_kwargs)
        events.append(ev)
        return ev

    areas = _AREAS[:n_areas]

    # Distractor pool first, drawn from a DIFFERENT rule/area pairing space than
    # the gold rows, so generation can never accidentally emit a gold duplicate.
    for i in range(distractors):
        area, what, path = rng.choice(areas)
        at = t0 + timedelta(hours=rng.randint(0, 24 * 120))
        foreign = rng.random() < foreign_rate
        add(
            {
                "at": at,
                "kind": "learning_created",
                "learning_id": f"D{i:06d}",
                "content": f"Note on {what}: {rng.choice(_RULES)}.",
                "detail": f"Seen in {path}." if not foreign else "Seen in another project's tree.",
                "tags": (area, "foreign") if foreign else (area,),
                "learning_type": "pattern",
            }
        )

    # Gold: one superseded convention and one current one per area, plus a commit
    # that should have recalled the current one.
    for ai, (area, what, path) in enumerate(areas):
        old_at = t0 + timedelta(days=10 + ai)
        new_at = t0 + timedelta(days=60 + ai)
        commit_at = t0 + timedelta(days=90 + ai)
        old_id, new_id = f"G{ai:03d}old", f"G{ai:03d}new"
        add(
            {
                "at": old_at,
                "kind": "learning_created",
                "learning_id": old_id,
                "content": _SUPERSEDED.format(what=what, rule=rng.choice(_RULES), year=2025),
                "detail": f"Applies to {path}.",
                "tags": (area, "convention"),
                "learning_type": "convention",
            }
        )
        add(
            {
                "at": new_at,
                "kind": "learning_created",
                "learning_id": new_id,
                "content": _LESSON.format(what=what, path=path, rule=rng.choice(_RULES)),
                "detail": f"Supersedes the earlier {what} guidance. Applies to {path}.",
                "tags": (area, "convention"),
                "learning_type": "convention",
            }
        )
        add({"at": new_at, "kind": "status_changed", "learning_id": old_id, "status": "obsolete"})
        successor_of[old_id] = new_id

        add(
            {
                "at": commit_at,
                "kind": "commit",
                "commit_sha": f"c{ai:07d}",
                "subject": f"fix({area}): correct {what} handling",
                "paths": (path,),
                "cites": (new_id,),
            }
        )
        # C1: before editing this file, what should I know? Gold is the current
        # convention; the superseded one is forbidden, which is the whole point.
        queries.append(
            Query(
                qid=f"C1-{area}",
                task="C1",
                as_of=commit_at,
                text=f"about to edit {path}: {what}",
                gold_ids=(new_id,),
                forbidden_ids=(old_id,),
                context={"commit": f"c{ai:07d}", "path": path},
                label_source="synthetic",
            )
        )
        # C3: current above superseded, asked without naming either.
        queries.append(
            Query(
                qid=f"C3-{area}",
                task="C3",
                as_of=commit_at,
                text=f"current guidance for {what}",
                gold_ids=(new_id,),
                forbidden_ids=(old_id,),
                context={"pair": [old_id, new_id]},
                label_source="synthetic",
            )
        )

    # Near-duplicates: the same lesson as several people would write it. Not gold
    # and not forbidden -- they test whether dedup keeps one without burying gold.
    dup_target = int(len(areas) * near_dup_rate * 20)
    for i in range(dup_target):
        area, what, path = rng.choice(areas)
        add(
            {
                "at": t0 + timedelta(days=rng.randint(61, 89)),
                "kind": "learning_created",
                "learning_id": f"N{i:05d}",
                "content": _LESSON.format(what=what, path=path, rule=rng.choice(_RULES)),
                "detail": f"Also applies to {path}.",
                "tags": (area,),
                "learning_type": "pattern",
            }
        )

    events.sort(key=lambda e: (e.at, e.seq))
    return events, queries, successor_of
