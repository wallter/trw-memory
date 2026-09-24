"""EngMem runner: replay an event stream, score every arm, compare them paired.

    python -m benchmarks.engmem.run --synth --store /tmp/engmem
    python -m benchmarks.engmem.run --events e.jsonl --queries q.jsonl --store DIR

Prints one table per arm plus the paired comparison that matters most right now:
``trw-hybrid`` (what LOCOMO measures) against ``trw-framework`` (what
``trw_recall`` and ``trw_session_start`` actually execute). The gold SHA and the
leakage audit print with every run -- a benchmark whose gold can change silently,
or whose queries can be answered from the future, is not measuring anything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.engmem import synth
from benchmarks.engmem.arms import Bm25Arm, GrepArm, RecencyArm, TrwFrameworkArm, TrwFtsFirstArm, TrwHybridArm
from benchmarks.engmem.replay import ReplayState, leakage_check, replay_and_score
from benchmarks.engmem.schema import freeze, read_events, read_queries, write_jsonl
from benchmarks.engmem.score import aggregate, latency, paired_mcnemar, pairwise_order_accuracy, render

NAMESPACE = "project:engmem"


async def main_async(args: argparse.Namespace) -> int:
    if args.synth:
        events, queries, successor_of = synth.generate(seed=args.seed, distractors=args.distractors)
        if args.dump:
            write_jsonl(Path(args.dump) / "events.jsonl", events)
            write_jsonl(Path(args.dump) / "queries.jsonl", queries)
            print(f"gold sha256 = {freeze(Path(args.dump) / 'queries.jsonl')}")
    else:
        events = read_events(Path(args.events))
        queries = read_queries(Path(args.queries))
        successor_of = {}
        print(f"gold sha256 = {freeze(Path(args.queries))}")

    leaks = leakage_check(events, queries)
    if leaks:
        print(f"LEAKAGE: {len(leaks)} queries answerable only from the future", file=sys.stderr)
        for line in leaks[:10]:
            print(f"  {line}", file=sys.stderr)
        if not args.allow_leakage:
            raise SystemExit("refusing to score a leaking suite; fix the extractor or pass --allow-leakage")

    # ASYNC240: a one-shot CLI, not a server -- blocking filesystem calls here
    # cost nothing and an async filesystem dependency would be pure ceremony.
    store = Path(args.store).expanduser()  # noqa: ASYNC240
    if store.exists() and args.fresh:
        shutil.rmtree(store)
    os.environ["MEMORY_STORAGE_PATH"] = str(store.resolve())

    # The plain-list arms need the corpus as it grows, which the replay maintains.
    # They share one ReplayState with the trw arms so every arm sees one corpus.
    mirror = ReplayState(rows=[], successor_of=dict(successor_of), retired=set())

    client = None
    arms: list = [RecencyArm(mirror.rows), GrepArm(mirror.rows)]
    if not args.no_trw:
        from trw_memory.client import MemoryClient

        client = MemoryClient(NAMESPACE, mode="local")
        arms += [
            TrwHybridArm(client, NAMESPACE),
            TrwFrameworkArm(client, NAMESPACE),
            TrwFtsFirstArm(client, NAMESPACE),
            TrwFtsFirstArm(client, NAMESPACE, rerank=True),
            Bm25Arm(client, NAMESPACE),
        ]

    try:
        results = await replay_and_score(events, queries, arms, client=client, state=mirror, limit=args.limit)
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                res = close()
                if asyncio.iscoroutine(res):
                    await res

    for name, scored in results.items():
        lat = latency(scored)
        print(
            render(
                aggregate(scored),
                f"{name} | {len(scored)} queries | limit={args.limit} | "
                f"p50 {lat['p50']:.0f}ms p95 {lat['p95']:.0f}ms max {lat['max']:.0f}ms",
            )
        )
        if mirror.successor_of:
            acc, n = pairwise_order_accuracy(scored, mirror.successor_of)
            print(f"  C3 pairwise order accuracy: {100 * acc:5.1f}%  (n={n} pairs where both rows surfaced)")

    if "trw-hybrid" in results and "trw-framework" in results:
        print("\n== what the framework path costs (paired, same queries, same store) ==")
        for metric in ("hit@5", "complete@5", "hit@10"):
            d = paired_mcnemar(results["trw-hybrid"], results["trw-framework"], metric)
            print(
                f"  {metric:12s} framework - hybrid = {d['delta_pp']:+5.1f} pp   "
                f"(hybrid-only {d['a_only']}, framework-only {d['b_only']}, p={d['p']:.3f}, n={d['n']})"
            )

    if args.out:
        Path(args.out).write_text(  # noqa: ASYNC240
            json.dumps({name: aggregate(s) for name, s in results.items()}, indent=1, sort_keys=True)
        )
        print(f"\nwrote {args.out}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synth", action="store_true", help="generate EngMem-Synth instead of reading files")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--distractors", type=int, default=200, help="corpus size knob for the scale curve")
    p.add_argument("--events", default="")
    p.add_argument("--queries", default="")
    p.add_argument("--store", default="~/.cache/trw-bench/engmem-store")
    p.add_argument("--dump", default="", help="write the generated suite here")
    p.add_argument("--out", default="")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--fresh", action="store_true", help="delete the store first (replay must start empty)")
    p.add_argument("--no-trw", action="store_true", help="baselines only; no embedding model needed")
    p.add_argument("--allow-leakage", action="store_true", help="score anyway (never for a published number)")
    args = p.parse_args()
    if not args.synth and not (args.events and args.queries):
        p.error("pass --synth, or both --events and --queries")
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
