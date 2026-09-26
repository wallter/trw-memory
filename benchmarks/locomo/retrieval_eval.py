"""LLM-free LOCOMO retrieval scorer for trw-memory: evidence recall@k / hit@k / MRR.

Every LOCOMO question (categories 1-4) lists the dialogue turns (``dia_id``)
that contain its answer. If the memory system stores turns, retrieval quality
is directly measurable without an answerer or a judge: did the evidence turns
surface in the top-k? This runs in seconds and is deterministic, which makes
it the inner loop for iterating on trw-memory retrieval. The LLM-judged
end-to-end run (``run.sh``) is the outer confirmation.

Usage::

    python retrieval_eval.py --store DIR [--conversations 0,1,...] [--k 10,50]
                             [--reingest] [--label NAME]

Ingestion is idempotent per conversation (skipped when the namespace already
holds the expected number of turns) so retrieval knobs can be swept via
``MEMORY_*`` environment variables without re-embedding.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def resolve_path(path: str) -> str:
    return str(Path(path).resolve())


def write_json(path: str, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1))


def parse_locomo_date(date_str: str) -> datetime | None:
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def turn_text(turn: dict[str, Any]) -> str:
    """Mirror upstream ``session_to_chunks`` formatting (speaker prefix + photo tag)."""
    text = turn.get("text", "")
    blip = turn.get("blip_caption", "")
    query = turn.get("query", "")
    if query and blip:
        tag = f"[Sharing image - query: {query}. The image shows: {blip}]"
    elif query:
        tag = f"[Sharing image - query for: {query}]"
    elif blip:
        tag = f"[Sharing image that shows: {blip}]"
    else:
        tag = ""
    if tag:
        text = f"{text} {tag}" if text else tag
    if not text:
        return ""
    return f"{turn.get('speaker', '')}: {text}"


def iter_turns(conversation: dict[str, Any]) -> Iterator[tuple[str, str, datetime | None, str, str]]:
    keys = [k for k in conversation if re.match(r"^session_\d+$", k)]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    for key in keys:
        date = conversation.get(f"{key}_date_time", "")
        dt = parse_locomo_date(date)
        for turn in conversation[key]:
            text = turn_text(turn)
            if text:
                yield key, date, dt, turn.get("dia_id", ""), text


async def ingest(client: Any, conversation: dict[str, Any], context_turns: int = 0) -> int:
    """Store every turn; with ``context_turns`` > 0 the preceding turns of the
    same session are carried in ``detail`` so BM25 and the embedder see the
    conversational context a bare reply like "Cool, what did it look like?"
    lacks. ``content`` stays the verbatim turn so evidence ids map 1:1."""
    n = 0
    prev: list[str] = []
    prev_key = None
    for key, date, dt, dia_id, text in iter_turns(conversation):
        if key != prev_key:
            prev, prev_key = [], key
        meta = {"dia_id": dia_id, "observed_at": dt.isoformat() if dt else "", "session_date": date}
        detail = " | ".join(prev[-context_turns:]) if context_turns else ""
        await client.store(text, detail=detail, metadata=meta, source="human")
        prev.append(text)
        n += 1
    return n


async def ingest_product(client: Any, conversation: dict[str, Any], context_turns: int = 1) -> int:
    """Ingest exactly as the product does: ``conversation_requests`` (the shaping
    behind ``MemoryClient.store_conversation``) per session, so context turns and
    session date words match the REST shim; only ``dia_id`` is added per row."""
    from trw_memory._client_conversation import conversation_requests

    sessions: dict[str, list[tuple[datetime | None, str, str]]] = defaultdict(list)
    for key, _date, dt, dia_id, text in iter_turns(conversation):
        sessions[key].append((dt, dia_id, text))
    n = 0
    for turns in sessions.values():
        observed = turns[0][0].isoformat() if turns[0][0] else None
        requests = conversation_requests(
            [{"content": text} for _dt, _d, text in turns], context_turns=context_turns, observed_at=observed
        )
        requests = [
            dataclasses.replace(req, metadata={**(req.metadata or {}), "dia_id": dia_id})
            for req, (_dt, dia_id, _text) in zip(requests, turns, strict=True)
        ]
        await client.bulk_store(requests)
        n += len(requests)
    return n


async def run(args: argparse.Namespace) -> None:
    os.environ["MEMORY_STORAGE_PATH"] = resolve_path(args.store)
    from trw_memory.client import MemoryClient

    data = load_json(args.dataset)
    conv_idx = [int(c) for c in args.conversations.split(",")]
    ks = [int(k) for k in args.k.split(",")]
    kmax = max(ks)

    per_q: list[dict[str, Any]] = []
    for ci in conv_idx:
        entry = data[ci]
        conv = entry["conversation"]
        ns = f"project:locomo-{ci}"
        client = MemoryClient(ns, mode="local")
        expected = sum(1 for _ in iter_turns(conv))
        have = len(await client.recall("a", limit=1)) if not args.reingest else 0
        backend = client._get_backend()
        count = backend.count(namespace=ns) if hasattr(backend, "count") else None
        if args.reingest or count != expected:
            if count:
                await client.clear() if hasattr(client, "clear") else None
            t = time.monotonic()
            if args.product:
                n = await ingest_product(client, conv, context_turns=args.context or 1)
            else:
                n = await ingest(client, conv, context_turns=args.context)
            print(f"conv {ci}: ingested {n} turns in {time.monotonic() - t:.1f}s", file=sys.stderr)
        else:
            print(f"conv {ci}: reusing {count} stored turns", file=sys.stderr)
        del have
        if args.ingest_only:
            continue

        t = time.monotonic()
        for qi, qa in enumerate(entry["qa"]):
            cat = qa.get("category")
            if cat not in (1, 2, 3, 4):
                continue
            evidence = set(qa.get("evidence", []))
            tq = time.monotonic()
            rows = await client.recall(qa["question"], limit=kmax, include_org_memories=False)
            query_s = round(time.monotonic() - tq, 3)
            ranked = [(r.get("metadata") or {}).get("dia_id", "") for r in rows]
            rec: dict[str, Any] = {"conv": ci, "q": qi, "category": cat, "n_evidence": len(evidence)}
            rec["n_returned"] = len(rows)
            rec["query_s"] = query_s
            first = next((i for i, d in enumerate(ranked) if d in evidence), None)
            rec["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
            # Rank (1-based) of every required turn, in the dataset's evidence order;
            # None when that turn never surfaced within kmax. Everything below is
            # derived from this, so a saved run can be re-scored without retrieval.
            pos: dict[str, int] = {}
            for i, d in enumerate(ranked):
                pos.setdefault(d, i + 1)
            ev_ranks = [pos.get(d) for d in sorted(evidence)]
            rec["evidence_ranks"] = ev_ranks
            # Cost of context: characters returned, so coverage can be read per token budget.
            rec["chars"] = [len(r.get("content") or "") + len(r.get("detail") or "") for r in rows]
            found = [r for r in ev_ranks if r is not None]
            # Rank of the LAST required turn: the depth a reader must see to answer
            # fully. None when any required turn is missing entirely.
            rec["last_required_rank"] = max(found) if evidence and len(found) == len(ev_ranks) else None
            for k in ks:
                top = ranked[:k]
                hits = sum(1 for d in evidence if d in top)
                rec[f"hit@{k}"] = 1.0 if hits else 0.0
                rec[f"recall@{k}"] = hits / len(evidence) if evidence else 0.0
                # complete@k: EVERY required turn is in the top k. hit@k and MRR
                # both score a single hit as success, which is what hid the
                # multi-hop bottleneck.
                rec[f"complete@{k}"] = 1.0 if evidence and hits == len(evidence) else 0.0
                rec[f"missing@{k}"] = (len(evidence) - hits) / len(evidence) if evidence else 0.0
            # Selection loss: complete at the deepest k, incomplete at the shallowest.
            # These items need no better acquisition, only a better choice of what to keep.
            rec["selection_loss"] = 1.0 if rec[f"complete@{kmax}"] and not rec[f"complete@{min(ks)}"] else 0.0
            per_q.append(rec)
        print(
            f"conv {ci}: {sum(1 for r in per_q if r['conv'] == ci)} questions in {time.monotonic() - t:.1f}s",
            file=sys.stderr,
        )
        close = getattr(client, "close", None)
        if close is not None:
            res = close()
            if asyncio.iscoroutine(res):
                await res

    def agg(rows: list[dict[str, Any]], key: str) -> float:
        return statistics.mean(r[key] for r in rows) * 100 if rows else 0.0

    metrics = [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks] + ["mrr"]
    print(f"\n== {args.label or 'trw-memory'} | {len(per_q)} questions | store={args.store}")
    print(f"{'category':12s}" + "".join(f"{m:>11s}" for m in metrics) + "     n")
    by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in per_q:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        print(f"{CATEGORY_NAMES[cat]:12s}" + "".join(f"{agg(rows, m):10.1f}%" for m in metrics) + f"  {len(rows):4d}")
    print(f"{'ALL':12s}" + "".join(f"{agg(per_q, m):10.1f}%" for m in metrics) + f"  {len(per_q):4d}")

    # Completeness view: a question is only answerable when EVERY required turn is
    # visible, so complete@k is the metric a reader actually lives under. The gap
    # between complete@kmin and complete@kmax is pure selection loss -- the evidence
    # was acquired and then discarded.
    comp = [f"complete@{k}" for k in ks] + ["selection_loss"]

    def med_last(rows: list[dict[str, Any]]) -> str:
        vals = [r["last_required_rank"] for r in rows if r["last_required_rank"] is not None]
        return f"{statistics.median(vals):.0f}" if vals else "-"

    print(f"\n-- completeness ({args.label or 'trw-memory'})")
    print(f"{'category':12s}" + "".join(f"{m:>15s}" for m in comp) + "   med_last_rank")
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        print(f"{CATEGORY_NAMES[cat]:12s}" + "".join(f"{agg(rows, m):14.1f}%" for m in comp) + f"{med_last(rows):>15s}")
    print(f"{'ALL':12s}" + "".join(f"{agg(per_q, m):14.1f}%" for m in comp) + f"{med_last(per_q):>15s}")
    if args.out:
        env = {k: v for k, v in os.environ.items() if k.startswith("MEMORY_")}
        write_json(args.out, {"label": args.label, "k": ks, "per_question": per_q, "env": env})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parents[3] / "scratch/memory-benchmarks/datasets/locomo/locomo10.json"),
    )
    p.add_argument("--store", required=True)
    p.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--k", default="10,50")
    p.add_argument("--reingest", action="store_true")
    p.add_argument("--ingest-only", action="store_true", help="populate the store, skip scoring")
    p.add_argument("--context", type=int, default=0, help="preceding turns carried in detail at ingest")
    p.add_argument(
        "--product", action="store_true", help="ingest through store_conversation's shaping (context + date words)"
    )
    p.add_argument("--label", default="")
    p.add_argument("--out", default="")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
