"""PRD-CORE-302 FR08 (W07): does ``rank_bm25`` earn a place as a base dependency?

Replays the synthetic EngMem corpus through the product recall paths twice per
seed: once as shipped, once as an install without the old ``bm25`` extra behaved
(no BM25 lane, no entity-bridge hop). Each (seed, variant) runs in its own
subprocess, so the patched state and the embedding model never leak between
arms, and the per-query results are paired by ``qid`` afterwards.

    MEMORY_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \\
      .venv/bin/python trw-memory/benchmarks/engmem/bm25_decision.py \\
      --seeds 7,11 --distractors 1000 --out <dir>

Prints one markdown table per recall path: complete@10 and forbidden@10 with
Wilson 95% CIs, p50/p95 latency, and the exact McNemar test on complete@10.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#: with = as shipped; nobridge = BM25 lane on, entity-bridge hop off; without = what an
#: install lacking the old ``bm25`` extra ran (no BM25 lane, and so no bridge either).
VARIANTS = ("with", "nobridge", "without")
ARMS = ("trw-framework", "trw-daemon-tool", "trw-hybrid")


async def _one(seed: int, distractors: int, store: Path, limit: int) -> dict[str, list[dict[str, object]]]:
    from benchmarks.engmem import synth
    from benchmarks.engmem.arms import TrwDaemonToolArm, TrwFrameworkArm, TrwHybridArm
    from benchmarks.engmem.replay import ReplayState, replay_and_score
    from benchmarks.engmem.scale import NAMESPACE
    from trw_memory.client import MemoryClient

    events, queries, successor_of = synth.generate(seed=seed, distractors=distractors)
    # ASYNC240: one-shot setup before the replay, as in scale.py.
    if store.exists():  # noqa: ASYNC240
        shutil.rmtree(store)
    store.mkdir(parents=True)  # noqa: ASYNC240
    os.environ["MEMORY_STORAGE_PATH"] = str(store.resolve())  # noqa: ASYNC240
    client = MemoryClient(NAMESPACE, mode="local")
    state = ReplayState(rows=[], successor_of=dict(successor_of), retired=set())
    arms = [TrwFrameworkArm(client, NAMESPACE), TrwDaemonToolArm(client, NAMESPACE), TrwHybridArm(client, NAMESPACE)]
    try:
        results = await replay_and_score(events, queries, arms, client=client, state=state, limit=limit)
    finally:
        await client.close()
    return {
        name: [{f: getattr(s, f) for f in ("qid", "task", "ranked", "gold", "forbidden", "tokens", "ms")} for s in rows]
        for name, rows in results.items()
    }


_PKG = Path(__file__).resolve().parents[2]


def _provenance(args: argparse.Namespace, variant: str, seed: int) -> dict[str, object]:
    """What a dump was measured on: the tree (HEAD plus any uncommitted src diff), this
    driver, and the run parameters. A resumed or paired dump must match on all of it."""

    def git(*argv: str) -> str:
        return subprocess.run(["git", "-C", str(_PKG), *argv], capture_output=True, text=True, check=True).stdout  # noqa: S603, S607

    return {
        "head": git("rev-parse", "HEAD").strip(),
        "src_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--", "src").encode()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "params": {"variant": variant, "seed": seed, "distractors": args.distractors, "limit": args.limit},
    }


def _child(args: argparse.Namespace) -> None:
    from trw_memory.retrieval import bm25, bridge

    if args.variant == "without":
        # Before FR08 made rank_bm25 a base dependency, ``bm25_search`` returned
        # ``[]`` without it. Replayed here so the measurement stays reproducible
        # now that the code has no such branch.
        real = bm25.bm25_search
        for module in list(sys.modules.values()):
            if getattr(module, "bm25_search", None) is real:
                module.bm25_search = lambda *a, **k: []  # type: ignore[attr-defined]
    if args.variant in ("nobridge", "without"):
        # The bridge reads BM25's idf, so it cannot run without rank_bm25 (it
        # raised NameError there, found by this benchmark).
        bridge.extend_with_bridge = lambda query, scored, tail, *a, **k: (scored, tail, False)  # type: ignore[assignment]
    out = asyncio.run(_one(args.seed, args.distractors, Path(args.store), args.limit))
    provenance = _provenance(args, args.variant, args.seed)
    Path(args.dump).write_text(json.dumps({"variant": args.variant, "provenance": provenance, "arms": out}))


def _load(path: Path) -> dict[str, list[object]]:
    from benchmarks.engmem.score import Scored

    raw = json.loads(path.read_text())["arms"]
    return {
        name: [
            Scored(**{**r, "ranked": tuple(r["ranked"]), "gold": tuple(r["gold"]), "forbidden": tuple(r["forbidden"])})
            for r in rows
        ]
        for name, rows in raw.items()
    }


def _pool(dumps: dict[tuple[int, str], Path], arm: str, seeds: list[int], variant: str) -> list[object]:
    """One arm's results for *variant* over *seeds*, qids prefixed by seed so pairing stays per query."""
    return [
        type(s)(**{**{f: getattr(s, f) for f in s.__slots__}, "qid": f"{seed}:{s.qid}"})
        for seed in seeds
        for s in _load(dumps[(seed, variant)])[arm]
    ]


def _rows(label: str, rows: list[object]) -> str:
    from benchmarks.engmem.score import latency, metrics_for, wilson

    n = len(rows)
    cells = []
    for metric in ("complete@10", "forbidden@10"):
        hits = sum(1 for s in rows if metrics_for(s, (10,))[metric] >= 1.0)
        lo, hi = wilson(hits, n)
        cells.append(f"{hits}/{n} = {hits / n:.3f} ({lo:.3f}-{hi:.3f})")
    lat = latency(rows)
    return f"| {label} | {n} | {cells[0]} | {cells[1]} | {lat['p50']:.1f} | {lat['p95']:.1f} |"


def _pair_line(a: str, b: str, rows_a: list[object], rows_b: list[object]) -> str:
    from benchmarks.engmem.score import paired_mcnemar

    m = paired_mcnemar(rows_a, rows_b, "complete@10", (10,))
    f = paired_mcnemar(rows_a, rows_b, "forbidden@10", (10,))
    return (
        f"- {a} vs {b}: complete@10 only-{a}={m['a_only']}, only-{b}={m['b_only']}, "
        f"delta={m['delta_pp']:+.1f} pp, McNemar p={m['p']:.4f}; forbidden@10 "
        f"only-{a}={f['a_only']}, only-{b}={f['b_only']}."
    )


_HEADER = [
    "| variant | N | complete@10 (95% CI) | forbidden@10 (95% CI) | p50 ms | p95 ms |",
    "|---|---|---|---|---|---|",
]


def _table(dumps: dict[tuple[int, str], Path], seeds: list[int], variants: list[str]) -> str:
    from benchmarks.engmem.score import metrics_for

    lines = []
    for arm in ARMS:
        pooled = {v: _pool(dumps, arm, seeds, v) for v in variants}
        lines += [f"### `{arm}` (seeds {', '.join(map(str, seeds))} pooled)", "", *_HEADER]
        lines += [_rows(v, pooled[v]) for v in variants]
        lines.append("")
        for a, b in (("without", "with"), ("without", "nobridge"), ("nobridge", "with")):
            if a in pooled and b in pooled:
                lines.append(_pair_line(a, b, pooled[a], pooled[b]))
        if "with" in pooled and "without" in pooled:
            # Queries share templates across seeds, so the pooled test treats clustered
            # outcomes as independent. Show where the discordance sits.
            a_ok = {s.qid: metrics_for(s, (10,))["complete@10"] >= 1.0 for s in pooled["without"]}
            by: dict[tuple[str, str], int] = {}
            for s in pooled["with"]:
                if metrics_for(s, (10,))["complete@10"] >= 1.0 and not a_ok[s.qid]:
                    key = (s.qid.split(":", 1)[0], s.task)
                    by[key] = by.get(key, 0) + 1
            per_seed = {seed: sum(n for (sd, _), n in by.items() if sd == str(seed)) for seed in seeds}
            lines += [
                "",
                f"- Found only with rank_bm25, by seed: {per_seed}; by (seed, task): "
                + (", ".join(f"{k[0]}/{k[1]}={n}" for k, n in sorted(by.items())) or "none")
                + ".",
            ]
        lines.append("")
    return "\n".join(lines)


def _compare(base: Path, cand: Path, seeds: list[int]) -> str:
    """Pair the ``with`` runs of two trees (baseline vs candidate) by ``seed:qid``.

    Refuses unless every pair was measured with the same driver and parameters on
    different trees, so a stale or mismatched dump cannot pass as a comparison.
    """
    dumps_b = {(seed, "with"): base / f"seed{seed}-with.json" for seed in seeds}
    dumps_c = {(seed, "with"): cand / f"seed{seed}-with.json" for seed in seeds}
    for seed in seeds:
        pb = json.loads(dumps_b[(seed, "with")].read_text())["provenance"]
        pc = json.loads(dumps_c[(seed, "with")].read_text())["provenance"]
        if pb["params"] != pc["params"] or pb["driver_sha256"] != pc["driver_sha256"]:
            raise SystemExit(f"seed {seed}: baseline and candidate differ in driver or parameters")
        if (pb["head"], pb["src_diff_sha256"]) == (pc["head"], pc["src_diff_sha256"]):
            raise SystemExit(f"seed {seed}: baseline and candidate were measured on the same tree")
    lines = [f"Baseline {base} vs candidate {cand}.", ""]
    for arm in ARMS:
        rb, rc = _pool(dumps_b, arm, seeds, "with"), _pool(dumps_c, arm, seeds, "with")
        lines += [f"### `{arm}` (seeds {', '.join(map(str, seeds))} pooled)", "", *_HEADER]
        lines += [_rows("baseline", rb), _rows("candidate", rc), "", _pair_line("baseline", "candidate", rb, rc), ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="7,11")
    ap.add_argument("--distractors", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", required=True, help="directory for stores and per-run dumps")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma list of variants to run")
    ap.add_argument("--baseline", help="another tree's --out dir: pair its `with` runs against this one's")
    ap.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--store", help=argparse.SUPPRESS)
    ap.add_argument("--dump", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.variant:
        _child(args)
        return
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.baseline:
        print(_compare(Path(args.baseline).expanduser(), out, seeds))
        return
    variants = [v for v in VARIANTS if v in args.variants.split(",")]
    dumps: dict[tuple[int, str], Path] = {}
    for seed in seeds:
        for variant in variants:
            dump = out / f"seed{seed}-{variant}.json"
            dumps[(seed, variant)] = dump
            if dump.exists() and json.loads(dump.read_text()).get("provenance") == _provenance(args, variant, seed):
                continue  # resumable: a finished run on this exact tree, driver and parameters
            print(f"seed {seed}, rank_bm25 {variant} ...", flush=True)
            subprocess.run(  # noqa: S603 -- this file, re-invoked with its own arguments
                [
                    sys.executable,
                    __file__,
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                    "--distractors",
                    str(args.distractors),
                    "--limit",
                    str(args.limit),
                    "--out",
                    str(out),
                    "--store",
                    str(out / f"store-{seed}-{variant}"),
                    "--dump",
                    str(dump),
                ],
                check=True,
            )
    print(_table(dumps, seeds, variants))


if __name__ == "__main__":
    main()
