"""LLM-free LongMemEval retrieval scorer for trw-memory: turn/session recall@k, hit@k, MRR.

LongMemEval (Wu et al., ICLR 2025, arXiv:2410.10813, MIT) gives every question
its own haystack of chat sessions and marks the evidence two ways: the
``answer_session_ids`` that hold the answer and, inside them, the turns flagged
``has_answer``. So retrieval is scoreable without an answerer or a judge: did
the evidence turns (or any turn of an evidence session) surface in the top-k?

Each question gets a fresh store directory and its own namespace, because the
haystacks share filler sessions and trw-memory's cross-project graph pass
reads and writes sibling project stores in the same storage directory. Every
session is ingested the way the product does it -- ``conversation_requests``,
the shaping behind ``MemoryClient.store_conversation`` (one preceding turn of
context plus the session's month/year in ``detail``) -- with the session's
``haystack_dates`` entry as ``observed_at``. Rows are tagged with the session
id, its position in the haystack and the turn index, so a hit maps back to a
``has_answer`` turn exactly.

Abstention questions (``question_id`` ending ``_abs``) have no evidence to
retrieve and are excluded. The question's ``question_date`` is NOT passed to
recall: ``recall(as_of=...)`` selects rows by their bitemporal validity window,
which starts at ingest time, so a 2023 ``as_of`` would hide every row.

Usage::

    python retrieval_eval.py --store DIR [--limit-questions N]
        [--question-types multi-session,...] [--k 10,50] [--workers 2]
        [--reingest] [--ingest-only] [--label NAME] [--out FILE]

Ingestion is idempotent per question (skipped when the store still holds the
row count recorded at ingest), so ``MEMORY_*`` retrieval knobs can be swept
without re-embedding. ``--out`` writes the per-question JSON in the LOCOMO
``retrieval_eval.py`` shape (``conv`` = question id, ``q`` = 0) so paired
McNemar scripts written for LOCOMO work unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

DEFAULT_DATASET = Path(__file__).resolve().parents[3] / "scratch/longmemeval/longmemeval_s_cleaned.json"
MANIFEST = "lme_ingest.json"


def load_questions(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data: list[dict[str, Any]] = json.load(fh)
    return data


def parse_lme_date(value: str) -> datetime | None:
    """``"2023/05/20 (Sat) 02:21"`` -> aware UTC datetime (the dataset carries no zone)."""
    try:
        return datetime.strptime(value, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc)
    except (
        ValueError,
        TypeError,
    ):  # trw-fail-silent-allow: an unparseable haystack date means the session is stored without observed_at, exactly as the dataset gives no usable date
        return None


def turn_key(session_pos: int | str, turn_index: int | str) -> str:
    return f"{session_pos}:{turn_index}"


def gold(item: dict[str, Any]) -> tuple[set[str], set[str]]:
    """(has_answer turn keys, answer session ids) for one question."""
    turns = {
        turn_key(pos, ti)
        for pos, session in enumerate(item["haystack_sessions"])
        for ti, turn in enumerate(session)
        if turn.get("has_answer")
    }
    return turns, set(item["answer_session_ids"])


def select_questions(data: list[dict[str, Any]], types: set[str], limit: int) -> list[dict[str, Any]]:
    """Non-abstention questions of *types*; with *limit*, a deterministic stratified sample.

    Allocation per question_type is proportional (largest remainder, at least one
    per type when the limit allows); within a type questions are ordered by a hash
    of their id, so the sample is stable across runs yet not file-order biased.
    """
    pool = [q for q in data if not q["question_id"].endswith("_abs") and (not types or q["question_type"] in types)]
    if not limit or limit >= len(pool):
        return pool
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in pool:
        by_type[q["question_type"]].append(q)
    for rows in by_type.values():
        rows.sort(key=lambda q: hashlib.sha1(q["question_id"].encode(), usedforsecurity=False).hexdigest())
    shares = {t: limit * len(rows) / len(pool) for t, rows in by_type.items()}
    alloc = {t: max(1, int(s)) if limit >= len(by_type) else int(s) for t, s in shares.items()}
    for t in sorted(shares, key=lambda t: shares[t] - int(shares[t]), reverse=True):
        if sum(alloc.values()) >= limit:
            break
        alloc[t] += 1
    while sum(alloc.values()) > limit:
        alloc[max(alloc, key=lambda t: alloc[t])] -= 1
    picked = {q["question_id"] for t, rows in by_type.items() for q in rows[: alloc[t]]}
    return [q for q in pool if q["question_id"] in picked]


async def ingest_question(client: Any, item: dict[str, Any], context_turns: int) -> dict[str, Any]:
    """One ``bulk_store`` per session through the product's conversation shaping."""
    from trw_memory._client_conversation import conversation_requests

    status: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    dropped: list[str] = []
    turns = 0
    for pos, (sid, date, session) in enumerate(
        zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"], strict=True)
    ):
        dt = parse_lme_date(date)
        requests = conversation_requests(
            [{"content": t["content"], "role": t["role"]} for t in session],
            context_turns=context_turns,
            observed_at=dt,
            # session id rides in metadata, NOT session_id=: that argument keys the
            # write rate limiter (10 rows/min/session), which silently rejects
            # every turn after the tenth of a session stored in one call.
            metadata={"session_id": sid, "session_pos": str(pos), "session_date": date},
        )
        turns += len(session)
        if not requests:
            continue
        summary = await client.bulk_store(requests)
        for req, res in zip(requests, summary.items, strict=True):
            status[res.status] += 1
            if res.status not in ("stored", "updated"):
                dropped.append(turn_key(pos, req.metadata["turn_index"]))
                reasons[(res.skipped_reason or res.anomaly_dimension or res.status)[:80]] += 1
    return {"turns": turns, "status": dict(status), "dropped": dropped, "reasons": dict(reasons)}


def score(
    ranked: list[tuple[str, str]], gold_turns: set[str], gold_sessions: set[str], ks: list[int]
) -> dict[str, Any]:
    """Turn-level (has_answer) and session-level (answer_session_ids) metrics over *ranked*."""
    rec: dict[str, Any] = {}
    first = next((i for i, (tk, _s) in enumerate(ranked) if tk in gold_turns), None)
    rec["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    sfirst = next((i for i, (_tk, s) in enumerate(ranked) if s in gold_sessions), None)
    rec["session_mrr"] = 1.0 / (sfirst + 1) if sfirst is not None else 0.0
    for k in ks:
        top = ranked[:k]
        turn_hits = len(gold_turns & {tk for tk, _s in top})
        sess_hits = len(gold_sessions & {s for _tk, s in top})
        rec[f"hit@{k}"] = 1.0 if turn_hits else 0.0
        rec[f"recall@{k}"] = turn_hits / len(gold_turns) if gold_turns else 0.0
        rec[f"session_hit@{k}"] = 1.0 if sess_hits else 0.0
        rec[f"session_recall@{k}"] = sess_hits / len(gold_sessions) if gold_sessions else 0.0
    return rec


async def _close(client: Any) -> None:
    close = getattr(client, "close", None)
    if close is not None:
        res = close()
        if asyncio.iscoroutine(res):
            await res


async def evaluate_one(item: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    qid = item["question_id"]
    # <root>/<qid>/memory: trw-memory keeps security state (write-rate and size-
    # anomaly baselines, audit log) beside storage_path, so the parent must be
    # per-question too or one question's ingest shifts the next one's quarantine.
    store = Path(opts["store"]) / qid / "memory"
    store.mkdir(parents=True, exist_ok=True)
    os.environ["MEMORY_STORAGE_PATH"] = str(store)
    from trw_memory.client import MemoryClient

    ns = f"project:lme-{qid}"
    client = MemoryClient(ns, mode="local")
    manifest_path = store / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    backend = client._get_backend()
    count = backend.count(namespace=ns)
    ingest_s = 0.0
    reused = not opts["reingest"] and manifest is not None and manifest.get("count") == count
    if not reused:
        if count:  # stale or partial store: start the question from an empty directory
            await _close(client)
            shutil.rmtree(store.parent)
            store.mkdir(parents=True)
            client = MemoryClient(ns, mode="local")
            backend = client._get_backend()
        t = time.monotonic()
        manifest = await ingest_question(client, item, opts["context"])
        ingest_s = time.monotonic() - t
        manifest["count"] = backend.count(namespace=ns)
        manifest["ingest_s"] = ingest_s
        manifest_path.write_text(json.dumps(manifest))
    if manifest is None:  # unreachable: reuse requires a manifest; narrows the type
        raise RuntimeError(f"{qid}: no ingest manifest")
    rec: dict[str, Any] = {"conv": qid, "q": 0, "category": item["question_type"], "reused": reused}
    rec["ingest_s"] = round(ingest_s, 2)
    rec["rows"] = manifest["count"]
    if opts["ingest_only"]:
        await _close(client)
        return rec
    gold_turns, gold_sessions = gold(item)
    t = time.monotonic()
    rows = await client.recall(item["question"], limit=opts["kmax"], include_org_memories=False)
    rec["query_s"] = round(time.monotonic() - t, 3)
    await _close(client)
    ranked = [
        (turn_key(m.get("session_pos", ""), m.get("turn_index", "")), m.get("session_id", ""))
        for m in ((r.get("metadata") or {}) for r in rows)
    ]
    rec["n_evidence"] = len(gold_turns)
    rec["n_answer_sessions"] = len(gold_sessions)
    rec["n_evidence_dropped"] = len(gold_turns & set(manifest["dropped"]))
    rec["n_returned"] = len(ranked)
    rec.update(score(ranked, gold_turns, gold_sessions, opts["ks"]))
    return rec


def _worker(item: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(evaluate_one(item, opts))


def report(per_q: list[dict[str, Any]], ks: list[int], label: str, store: str) -> None:
    def agg(rows: list[dict[str, Any]], key: str) -> float:
        return statistics.mean(r[key] for r in rows) * 100 if rows else 0.0

    blocks = [
        ("turn (has_answer)", [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks] + ["mrr"]),
        ("session (answer_session_ids)", [f"session_recall@{k}" for k in ks] + ["session_mrr"]),
    ]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in per_q:
        by_type[r["category"]].append(r)
    print(f"\n== {label or 'trw-memory'} | {len(per_q)} questions | store={store}")
    for title, metrics in blocks:
        print(f"\n-- {title}")
        head = [m.replace("session_", "s_") for m in metrics]
        print(f"{'question_type':26s}" + "".join(f"{m:>11s}" for m in head) + "     n")
        for qt in sorted(by_type):
            rows = by_type[qt]
            print(f"{qt:26s}" + "".join(f"{agg(rows, m):10.1f}%" for m in metrics) + f"  {len(rows):4d}")
        print(f"{'ALL':26s}" + "".join(f"{agg(per_q, m):10.1f}%" for m in metrics) + f"  {len(per_q):4d}")
    dropped = sum(r["n_evidence_dropped"] for r in per_q)
    if dropped:
        print(f"\nWARNING: {dropped} has_answer turns were not stored (quarantined/rejected) and cannot be hit")


def run(args: argparse.Namespace) -> None:
    ks = sorted({int(k) for k in args.k.split(",")})
    types = {t for t in args.question_types.split(",") if t}
    data = load_questions(args.dataset)
    n_abs = sum(1 for q in data if q["question_id"].endswith("_abs"))
    items = select_questions(data, types, args.limit_questions)
    print(
        f"{len(items)} questions selected ({n_abs} abstention excluded) by type: "
        f"{dict(Counter(q['question_type'] for q in items))}",
        file=sys.stderr,
    )
    opts = {
        "store": str(Path(args.store).resolve()),
        "context": args.context,
        "reingest": args.reingest,
        "ingest_only": args.ingest_only,
        "ks": ks,
        "kmax": max(ks),
    }
    per_q: list[dict[str, Any]] = []
    t0 = time.monotonic()

    def done(rec: dict[str, Any]) -> None:
        per_q.append(rec)
        how = "reused" if rec["reused"] else f"ingested in {rec['ingest_s']:.1f}s"
        hit = "" if args.ingest_only else f" hit@{ks[0]}={rec[f'hit@{ks[0]}']:.0f}"
        elapsed = time.monotonic() - t0
        print(
            f"[{len(per_q)}/{len(items)} {elapsed:.0f}s] {rec['conv']} {rec['category']} rows={rec['rows']} {how}{hit}",
            file=sys.stderr,
        )

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for fut in as_completed([pool.submit(_worker, item, opts) for item in items]):
                done(fut.result())
    else:
        for item in items:
            done(_worker(item, opts))
    wall = time.monotonic() - t0
    order = {q["question_id"]: i for i, q in enumerate(items)}
    per_q.sort(key=lambda r: order[r["conv"]])
    timing = {
        "wall_s": round(wall, 1),
        "workers": args.workers,
        "ingest_s_sum": round(sum(r["ingest_s"] for r in per_q), 1),
        "query_s_sum": round(sum(r.get("query_s", 0.0) for r in per_q), 2),
        "questions_ingested": sum(1 for r in per_q if not r["reused"]),
        "rows": sum(r["rows"] for r in per_q),
    }
    if per_q and not args.ingest_only:
        timing["query_s_median"] = statistics.median(r["query_s"] for r in per_q)
        report(per_q, ks, args.label, args.store)
    print(f"\ntiming: {json.dumps(timing)}")
    if args.out:
        env = {k: v for k, v in os.environ.items() if k.startswith("MEMORY_")}
        payload = {"label": args.label, "k": ks, "dataset": args.dataset, "timing": timing, "env": env}
        payload["per_question"] = per_q
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--store", required=True, help="root dir; one sub-store per question")
    p.add_argument("--k", default="10,50")
    p.add_argument("--limit-questions", type=int, default=0, help="stratified deterministic sample of N questions")
    p.add_argument("--question-types", default="", help="comma list of question_type values to keep")
    p.add_argument("--workers", type=int, default=1, help="parallel processes (questions use separate stores)")
    p.add_argument("--context", type=int, default=1, help="context_turns for the product shaping")
    p.add_argument("--reingest", action="store_true")
    p.add_argument("--ingest-only", action="store_true", help="populate the stores, skip scoring")
    p.add_argument("--label", default="")
    p.add_argument("--out", default="")
    run(p.parse_args())


if __name__ == "__main__":
    main()
