"""Compare two judged LOCOMO runs question-by-question: accuracy, Wilson 95% CI, McNemar.

Usage::  python compare.py <predicted_dir_A> <predicted_dir_B> [--label-a mem0 --label-b trw]

Both directories must come from the same dataset slice; only question ids
present in both are compared (paired). Per CLAUDE.md, a comparative claim
needs N, the point estimates, and either disjoint Wilson intervals or a
paired test -- this prints all of them per cutoff and per category.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load(d: str) -> dict[str, dict]:
    out = {}
    for p in Path(d).glob("conv*_q*.json"):
        r = json.loads(p.read_text())
        if r.get("cutoff_results"):
            out[r["question_id"]] = r
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    ids = sorted(set(A) & set(B))
    print(f"paired questions: {len(ids)}  ({args.label_a}: {len(A)} judged, {args.label_b}: {len(B)} judged)")
    # Only cutoffs judged for EVERY paired question on both sides: a run judged
    # at fewer cutoffs (e.g. top-10 only) must not be compared at a missing one.
    cutoffs = sorted(
        set.intersection(*(set(A[q]["cutoff_results"]) & set(B[q]["cutoff_results"]) for q in ids)) if ids else set()
    )
    for cut in cutoffs:
        rows = defaultdict(lambda: [0, 0, 0, 0, 0])  # n, a_correct, b_correct, b(A only), c(B only)
        for q in ids:
            a = A[q]["cutoff_results"][cut]["judgment"] == "CORRECT"
            b = B[q]["cutoff_results"][cut]["judgment"] == "CORRECT"
            for key in ("ALL", A[q]["category_name"]):
                r = rows[key]
                r[0] += 1
                r[1] += a
                r[2] += b
                r[3] += a and not b
                r[4] += b and not a
        print(f"\n== {cut}")
        print(f"{'category':12s} {'n':>4s}  {args.label_a:>18s}  {args.label_b:>18s}   discordant  McNemar p")
        for key in ["ALL", *sorted(k for k in rows if k != "ALL")]:
            n, ka, kb, d_a, d_b = rows[key]
            la, ha = wilson(ka, n)
            lb, hb = wilson(kb, n)
            print(
                f"{key:12s} {n:4d}  {100 * ka / n:5.1f}% [{100 * la:4.1f},{100 * ha:4.1f}]"
                f"  {100 * kb / n:5.1f}% [{100 * lb:4.1f},{100 * hb:4.1f}]"
                f"   {d_a:3d}/{d_b:<3d}    {mcnemar(d_a, d_b):.3f}"
            )


if __name__ == "__main__":
    main()
