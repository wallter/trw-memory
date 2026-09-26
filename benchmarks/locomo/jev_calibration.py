"""Calibration and discrimination of trw-jev against the harness judge, n=1,540.

A calibrated probability makes a falsifiable promise: of the items it scores 0.9,
about 90% should be correct. That is checkable, and it is what separates a
probability from a confidence-flavoured label. This measures it.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# A jev_judge.py output: {"provenance", "rows"}, or the older flat {qid: row}.
default = Path.home() / ".cache/trw-bench/memory-benchmarks/results/locomo/predicted_trw-v6__gpt4omini/jev.json"
with open(sys.argv[1] if len(sys.argv) > 1 else default) as fh:
    rows = json.load(fh)
rows = rows.get("rows", rows)
vals = [
    (r["p"], r["harness"]) for r in rows.values() if r.get("p") is not None and r.get("harness") in ("CORRECT", "WRONG")
]
n = len(vals)
pos = [p for p, h in vals if h == "CORRECT"]
neg = [p for p, h in vals if h == "WRONG"]
print(f"n={n}   harness CORRECT={len(pos)} ({100 * len(pos) / n:.1f}%)   WRONG={len(neg)}")

# AUC via rank-sum (ties averaged) -- probability a random correct item outranks
# a random wrong one. Threshold-free, so it survives disagreement about where the
# line should sit.
order = sorted(range(n), key=lambda i: vals[i][0])
ranks = [0.0] * n
i = 0
while i < n:
    j = i
    while j + 1 < n and vals[order[j + 1]][0] == vals[order[i]][0]:
        j += 1
    avg = (i + j) / 2 + 1
    for k in range(i, j + 1):
        ranks[order[k]] = avg
    i = j + 1
rank_sum = sum(ranks[i] for i in range(n) if vals[i][1] == "CORRECT")
auc = (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
print(f"AUC = {auc:.4f}")

print("\ncalibration -- of items JEV scored in this band, how many did the harness call correct?")
print(f"{'band':>12s}{'n':>7s}{'% of set':>10s}{'harness correct':>18s}")
bands = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.01)]
for lo, hi in bands:
    sel = [(p, h) for p, h in vals if lo <= p < hi]
    if not sel:
        continue
    acc = sum(1 for _p, h in sel if h == "CORRECT") / len(sel)
    print(
        f"{f'{lo:.1f}-{hi if hi <= 1 else 1.0:.1f}':>12s}{len(sel):>7d}{100 * len(sel) / n:>9.1f}%{100 * acc:>17.1f}%"
    )

mid = [(p, h) for p, h in vals if 0.3 <= p < 0.7]
print(f"\nmid-band (0.3-0.7): {len(mid)} items, {100 * len(mid) / n:.1f}% of the set")
print("  these are the items where the RUBRIC is undecided, not where the system half-worked")

print("\nagreement at thresholds:")
for t in (0.5, 0.53, 0.7, 0.9):
    agree = sum(1 for p, h in vals if (p >= t) == (h == "CORRECT"))
    says_yes = sum(1 for p, _h in vals if p >= t)
    print(
        f"  p>={t:.2f}: JEV says correct for {100 * says_yes / n:5.1f}%   agrees with harness on {100 * agree / n:5.1f}%"
    )

by_cat = defaultdict(list)
for r in rows.values():
    if r.get("p") is not None and r.get("category"):
        by_cat[r["category"]].append((r["p"], r.get("harness")))
print("\nby category:   harness%   jev>=0.5%   mean p")
for c in sorted(by_cat):
    s = by_cat[c]
    hc = sum(1 for _p, h in s if h == "CORRECT") / len(s)
    jc = sum(1 for p, _h in s if p >= 0.5) / len(s)
    mp = sum(p for p, _h in s) / len(s)
    print(f"  {c:12s}{100 * hc:9.1f}%{100 * jc:11.1f}%{mp:9.2f}   n={len(s)}")
