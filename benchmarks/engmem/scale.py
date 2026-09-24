"""Scale curve: quality AND cost as the store grows.

    python -m benchmarks.engmem.scale --sizes 1000,10000,100000

trw-memory's stated ambition is 100-10,000 engineers and 10^7-10^9+ learnings.
Today it is per-project SQLite, and the recall path has three structures whose
cost is a function of store size rather than of the answer:

* ``_client_recall_hybrid`` auto-scales the BM25 and dense candidate caps to the
  namespace size, so both legs grow with the corpus rather than with ``limit``;
* the BM25 model is rebuilt per namespace from an in-memory corpus;
* dense retrieval scans the supplied ids rather than an ANN index.

Any one of those turns a 10^3-row benchmark win into a 10^6-row outage, and no
amount of LOCOMO tuning would show it. This sweep plants the same gold at every
size and grows only the distractors, so a drop in quality is interference and a
rise in latency is the structure -- and the two are read together, because a
policy that holds its recall while going superlinear in time has not scaled.

Run it with ``--no-trw`` for a free structural baseline (grep/recency), or with
the trw arms to get the number that matters. Sizes above ~10^5 will take a while
on a laptop; that is itself a finding worth writing down rather than hiding.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.engmem import synth
from benchmarks.engmem.arms import (
    Bm25Arm,
    GrepArm,
    RecencyArm,
    TrwDaemonToolArm,
    TrwFrameworkArm,
    TrwFtsFirstArm,
    TrwHybridArm,
    TrwQueryPoolArm,
)
from benchmarks.engmem.replay import ReplayState, replay_and_score
from benchmarks.engmem.score import aggregate, latency, metrics_for, paired_mcnemar, wilson

NAMESPACE = "project:engmem-scale"
#: The arm every other arm is compared against per query: what the framework ships.
BASE_ARM = "trw-framework"


async def one_size(size: int, args: argparse.Namespace) -> dict[str, dict[str, float]]:
    import os

    events, queries, successor_of = synth.generate(seed=args.seed, distractors=size)
    # ASYNC240: a one-shot CLI sweep, not a server; blocking filesystem calls here
    # are a rounding error next to the replay they set up.
    store = Path(args.store).expanduser() / f"n{size}"  # noqa: ASYNC240
    if store.exists():
        shutil.rmtree(store)  # every size starts from an empty store, or the curve is a lie
    store.mkdir(parents=True, exist_ok=True)
    os.environ["MEMORY_STORAGE_PATH"] = str(store.resolve())

    mirror = ReplayState(rows=[], successor_of=dict(successor_of), retired=set())
    arms: list = [RecencyArm(mirror.rows), GrepArm(mirror.rows)]
    client = None
    if not args.no_trw:
        from trw_memory.client import MemoryClient

        client = MemoryClient(NAMESPACE, mode="local")
        arms += [
            TrwHybridArm(client, NAMESPACE),
            TrwFrameworkArm(client, NAMESPACE),
            TrwDaemonToolArm(client, NAMESPACE),
            TrwQueryPoolArm(client, NAMESPACE),
            TrwFtsFirstArm(client, NAMESPACE),
            TrwFtsFirstArm(client, NAMESPACE, rerank=True),
            Bm25Arm(client, NAMESPACE),
        ]
    if args.arms:
        wanted = set(args.arms.split(","))
        unknown = wanted - {arm.name for arm in arms}
        if unknown:
            raise SystemExit(f"unknown arms: {sorted(unknown)}")
        arms = [arm for arm in arms if arm.name in wanted]

    t0 = time.perf_counter()
    close_error = ""
    try:
        results = await replay_and_score(events, queries, arms, client=client, state=mirror, limit=args.limit)
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    res = close()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as exc:
                    # A close failure at size N is a SCALE RESULT, not a reason to
                    # lose the sweep: `close_client` drains background graph
                    # updates under a fixed timeout, and that drain is the first
                    # thing observed to break as the store grows. Record it and
                    # carry on, so the curve shows where it starts.
                    close_error = f"{type(exc).__name__}: {exc}"
                    print(f"    CLOSE FAILED at {size:,} rows -- {close_error}", flush=True)
    ingest_s = time.perf_counter() - t0

    out: dict[str, dict[str, float]] = {}
    for name, scored in results.items():
        agg = aggregate(scored)["ALL"]
        lat = latency(scored)
        out[name] = {
            "n": float(len(scored)),
            "hit@10": 100 * agg["hit@10"],
            "complete@10": 100 * agg["complete@10"],
            "forbidden@10": 100 * agg["forbidden@10"],
            "mrr": 100 * agg["mrr"],
            "p50_ms": lat["p50"],
            "p95_ms": lat["p95"],
        }
        # Both rates are binary per query, so a Wilson interval over queries applies.
        for metric in ("complete@10", "forbidden@10"):
            successes = sum(int(metrics_for(s, (10,))[metric]) for s in scored)
            low, high = wilson(successes, len(scored))
            out[name][f"{metric}_ci95"] = [100 * low, 100 * high]  # type: ignore[assignment]
        if BASE_ARM in results and name != BASE_ARM:
            # Paired over the same queries: a per-query comparison, not two means.
            out[name][f"complete@10_mcnemar_vs_{BASE_ARM}"] = paired_mcnemar(  # type: ignore[assignment]
                results[BASE_ARM], scored, "complete@10", ks=(10,)
            )
    # Disk is the other enterprise cost nobody measures until it bites.
    out["_run"] = {
        "rows": float(size),
        "wall_s": ingest_s,
        "store_mb": sum(f.stat().st_size for f in store.rglob("*") if f.is_file()) / 1e6,
        "close_failed": 1.0 if close_error else 0.0,
    }
    if close_error:
        out["_run"]["close_error"] = close_error  # type: ignore[assignment]
    return out


async def main_async(args: argparse.Namespace) -> int:
    sizes = [int(s) for s in args.sizes.split(",")]
    curve: dict[int, dict[str, dict[str, float]]] = {}
    for size in sizes:
        print(f"\n--- store size {size:,} ---", flush=True)
        curve[size] = await one_size(size, args)
        run = curve[size]["_run"]
        print(f"    replay {run['wall_s']:.1f}s, store {run['store_mb']:.1f} MB", flush=True)
        for name, m in curve[size].items():
            if name == "_run":
                continue
            print(
                f"    {name:14s} hit@10 {m['hit@10']:5.1f}%  complete@10 {m['complete@10']:5.1f}%  "
                f"forbidden@10 {m['forbidden@10']:5.1f}%  p50 {m['p50_ms']:7.1f}ms  p95 {m['p95_ms']:7.1f}ms",
                flush=True,
            )

    arms = [k for k in curve[sizes[0]] if k != "_run"]
    print("\n== scale curve: p50 latency (ms) by store size ==")
    print(f"{'arm':16s}" + "".join(f"{s:>12,}" for s in sizes))
    for name in arms:
        print(f"{name:16s}" + "".join(f"{curve[s][name]['p50_ms']:12.1f}" for s in sizes))
    print("\n== scale curve: hit@10 (%) by store size ==")
    print(f"{'arm':16s}" + "".join(f"{s:>12,}" for s in sizes))
    for name in arms:
        print(f"{name:16s}" + "".join(f"{curve[s][name]['hit@10']:12.1f}" for s in sizes))

    if args.out:
        Path(args.out).write_text(  # noqa: ASYNC240
            json.dumps({str(k): v for k, v in curve.items()}, indent=1, sort_keys=True)
        )
        print(f"\nwrote {args.out}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="1000,10000", help="distractor counts to sweep")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--store", default="~/.cache/trw-bench/engmem-scale")
    p.add_argument("--out", default="")
    p.add_argument("--no-trw", action="store_true", help="baselines only; no embedding model needed")
    p.add_argument("--arms", default="", help="comma-separated arm names to run (default: all)")
    raise SystemExit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
