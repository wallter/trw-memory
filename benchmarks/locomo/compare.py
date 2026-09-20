"""Compare two judged LOCOMO runs question-by-question (paired).

Usage::  python compare.py <predicted_dir_A> <predicted_dir_B> [--label-a mem0 --label-b trw]
                          --cutoff top_10 [--expected 1540] [--margin 2.0] [--json out.json]

Reports, per the repo's statistical rule and the hosted-judge plan's pre-registration:
accuracy with Wilson 95% CIs, discordant counts, exact McNemar, the paired difference
(B - A) with a Wald 95% CI and a conversation-cluster bootstrap 95% CI (the 1,540
questions sit in only ten conversations), a TOST-style equivalence verdict against
``--margin`` percentage points (cluster-bootstrap 90% CI inside +/- margin), a
per-conversation table, and context/answer size per system. One cutoff per call.

Decision rule (pre-registered): DIFFERENCE needs McNemar p < 0.05 AND a cluster 95% CI
excluding zero; EQUIVALENT needs the cluster 90% CI inside +/- margin; otherwise INCONCLUSIVE.

Integrity is checked before anything is scored: both sides must contain the same
question ids, the same gold answers and categories, and (with ``--expected``) exactly
that many questions. A judgment of ``ERROR`` (an unparseable judge reply) scores as
wrong, like the stock runner, but is counted and also excluded in a sensitivity row.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(d: str) -> dict[str, dict[str, Any]]:
    out = {}
    for p in Path(d).glob("conv*_q*.json"):
        r = json.loads(p.read_text())
        out[r.get("question_id") or p.stem] = r
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on the discordant pairs (b: A right/B wrong, c: A wrong/B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_diff_wald(pairs: list[tuple[int, int]], z: float = 1.96) -> tuple[float, float, float]:
    """B - A difference in proportions for paired binary outcomes, with a Wald CI."""
    n = len(pairs)
    b = sum(1 for a, bb in pairs if a and not bb)
    c = sum(1 for a, bb in pairs if bb and not a)
    d = (c - b) / n
    se = math.sqrt(max((b + c) / n - d * d, 0.0) / n)
    return d, d - z * se, d + z * se


def bootstrap_diffs(pairs_by_conv: dict[str, list[tuple[int, int]]], reps: int, seed: int) -> list[float]:
    """Sorted B - A differences from resampling whole conversations (one distribution for every CI)."""
    rng = random.Random(seed)
    convs = sorted(pairs_by_conv)
    stats = []
    for _ in range(reps):
        sample = [pairs_by_conv[rng.choice(convs)] for _ in convs]
        n = sum(len(s) for s in sample)
        stats.append(sum(bb - a for s in sample for a, bb in s) / n)
    return sorted(stats)


def percentile_ci(stats: list[float], alpha: float) -> tuple[float, float]:
    reps = len(stats)
    return stats[int(alpha / 2 * reps)], stats[min(reps - 1, int((1 - alpha / 2) * reps))]


VERDICTS = {"CORRECT", "WRONG", "ERROR"}


def gold(r: dict[str, Any]) -> Any:
    return r.get("ground_truth_answer", r.get("answer"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--cutoff", required=True, help="e.g. top_10")
    ap.add_argument("--expected", type=int, default=None, help="exact number of paired questions required")
    ap.add_argument("--margin", type=float, default=2.0, help="equivalence margin, percentage points")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--json", type=Path, default=None, help="also write the numbers here")
    ap.add_argument("--qids", type=Path, default=None, help="compare only these question ids (samples, P2)")
    ap.add_argument("--max-discordant", type=float, default=None,
                    help="agreement gate: exit 3 if more than this fraction of pairs disagree (P2: 0.05)")  # fmt: skip
    args = ap.parse_args(argv)
    cut, la_, lb_ = args.cutoff, args.label_a, args.label_b

    A, B = load(args.a), load(args.b)
    if args.qids:
        keep = {q.strip() for q in args.qids.read_text().splitlines() if q.strip()}
        A, B = {q: r for q, r in A.items() if q in keep}, {q: r for q, r in B.items() if q in keep}
    problems = []
    if set(A) != set(B):
        problems.append(f"question ids differ: only-{la_}={len(set(A) - set(B))} only-{lb_}={len(set(B) - set(A))}")
    ids = sorted(set(A) & set(B))
    if not ids:
        problems.append("no paired questions")
    if args.expected is not None and len(ids) != args.expected:
        problems.append(f"expected {args.expected} paired questions, found {len(ids)}")
    for q in ids:
        for side, r in ((la_, A[q]), (lb_, B[q])):
            if gold(r) is None or "category" not in r or "question" not in r:
                problems.append(f"{q}: {side} lacks gold answer, category or question")
            elif (res := r.get("cutoff_results", {}).get(cut)) is None:
                problems.append(f"{q}: {side} not judged at {cut}")
            elif str(res.get("judgment", "")).upper() not in VERDICTS:
                problems.append(f"{q}: {side} judgment {res.get('judgment')!r} is not one of {sorted(VERDICTS)}")
        if (
            gold(A[q]) != gold(B[q])
            or A[q].get("category") != B[q].get("category")
            or A[q].get("question") != B[q].get("question")
        ):
            problems.append(f"{q}: gold answer, category or question differs between runs")
    if problems:
        print(
            f"INTEGRITY FAILURE ({len(problems)} problems) — not scoring:\n  " + "\n  ".join(problems[:20]),
            file=sys.stderr,
        )
        return 2

    ra = {q: A[q]["cutoff_results"][cut] for q in ids}
    rb = {q: B[q]["cutoff_results"][cut] for q in ids}
    vd = {q: (str(ra[q]["judgment"]).upper(), str(rb[q]["judgment"]).upper()) for q in ids}
    errors = {la_: sum(v[0] == "ERROR" for v in vd.values()), lb_: sum(v[1] == "ERROR" for v in vd.values())}
    pairs = {q: (int(vd[q][0] == "CORRECT"), int(vd[q][1] == "CORRECT")) for q in ids}
    by_conv: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for q, p in pairs.items():
        by_conv[q.split("_")[0]].append(p)

    print(f"paired questions: {len(ids)}   cutoff {cut}   judge errors: {errors}")
    print(f"{'category':12s} {'n':>4s}  {la_:>18s}  {lb_:>18s}   discordant  McNemar p")
    rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for q, p in pairs.items():
        rows["ALL"].append(p)
        rows[A[q].get("category_name", "unknown")].append(p)
    report: dict[str, Any] = {"n": len(ids), "cutoff": cut, "labels": [la_, lb_], "judge_errors": errors, "rows": {}}
    for key in ["ALL", *sorted(k for k in rows if k != "ALL")]:
        ps = rows[key]
        n, ka, kb = len(ps), sum(a for a, _ in ps), sum(b for _, b in ps)
        d_a, d_b = sum(1 for a, b in ps if a and not b), sum(1 for a, b in ps if b and not a)
        (lo_a, hi_a), (lo_b, hi_b), p = wilson(ka, n), wilson(kb, n), mcnemar(d_a, d_b)
        report["rows"][key] = {"n": n, "a": ka, "b": kb, "a_only": d_a, "b_only": d_b, "mcnemar_p": p}
        print(f"{key:12s} {n:4d}  {100 * ka / n:5.1f}% [{100 * lo_a:4.1f},{100 * hi_a:4.1f}]"
              f"  {100 * kb / n:5.1f}% [{100 * lo_b:4.1f},{100 * hi_b:4.1f}]   {d_a:3d}/{d_b:<3d}    {p:.4f}")  # fmt: skip

    d, wlo, whi = paired_diff_wald(list(pairs.values()))
    stats = bootstrap_diffs(by_conv, args.reps, args.seed)
    (blo, bhi), (elo, ehi) = percentile_ci(stats, 0.05), percentile_ci(stats, 0.10)
    m = args.margin / 100
    p_all = report["rows"]["ALL"]["mcnemar_p"]
    # Pre-registered decision rule: a difference needs BOTH the question-level test and the
    # conversation-cluster interval; equivalence needs the cluster 90% interval inside the margin.
    if p_all < 0.05 and (blo > 0 or bhi < 0):
        decision = f"DIFFERENCE ({lb_} {'ahead' if d > 0 else 'behind'})"
    elif -m < elo and ehi < m:
        decision = f"EQUIVALENT within ±{args.margin:g} pp"
    else:
        decision = "INCONCLUSIVE"
    print(f"\n{lb_} - {la_}: {100 * d:+.1f} pp   Wald 95% [{100 * wlo:+.1f}, {100 * whi:+.1f}]"
          f"   cluster-bootstrap 95% [{100 * blo:+.1f}, {100 * bhi:+.1f}]   cluster 90% [{100 * elo:+.1f}, {100 * ehi:+.1f}]")  # fmt: skip
    print(f"decision: {decision}")

    kept = [q for q in ids if "ERROR" not in vd[q]]
    sens = paired_diff_wald([pairs[q] for q in kept])[0] if kept else None
    if len(kept) != len(ids):
        shown = "n/a" if sens is None else f"{100 * sens:+.1f} pp"
        print(f"sensitivity, judge errors excluded (n={len(kept)}): {shown}")

    print(f"\n{'conv':6s} {'n':>4s} {la_:>8s} {lb_:>8s}   diff")
    per_conv = {}
    for conv in sorted(by_conv, key=lambda c: int(c.removeprefix("conv"))):
        ps = by_conv[conv]
        ka, kb = sum(a for a, _ in ps), sum(b for _, b in ps)
        per_conv[conv] = {"n": len(ps), "a": ka, "b": kb}
        print(
            f"{conv:6s} {len(ps):4d} {100 * ka / len(ps):7.1f}% {100 * kb / len(ps):7.1f}%  {100 * (kb - ka) / len(ps):+5.1f}"
        )
    ahead = sum(1 for v in per_conv.values() if v["b"] > v["a"])
    behind = sum(1 for v in per_conv.values() if v["b"] < v["a"])
    print(f"conversations: {lb_} ahead {ahead}, behind {behind}, tied {len(per_conv) - ahead - behind}")

    def sizes(r: dict[str, dict[str, Any]]) -> dict[str, float]:
        mem = [r[q].get("memories_evaluated", 0) for q in ids]
        ans = [len(str(r[q].get("generated_answer", ""))) for q in ids]
        return {"mean_memories": sum(mem) / len(mem), "mean_answer_chars": sum(ans) / len(ans)}

    size = {la_: sizes(ra), lb_: sizes(rb)}
    print("context/answer size:", json.dumps({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in size.items()}))
    report.update({
        "diff": d, "wald95": [wlo, whi], "cluster95": [blo, bhi], "cluster90": [elo, ehi], "margin_pp": args.margin,
        "decision": decision, "sensitivity_diff": sens, "sensitivity_n": len(kept),
        "per_conversation": per_conv, "sizes": size,
    })  # fmt: skip
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    if args.max_discordant is not None:
        row = report["rows"]["ALL"]
        frac = (row["a_only"] + row["b_only"]) / row["n"]
        print(f"agreement gate: {100 * (1 - frac):.1f}% agree (limit {100 * (1 - args.max_discordant):.1f}%)")
        if frac > args.max_discordant:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
