"""Paired significance for two LLM-free retrieval runs over the SAME questions.

Reads two per-question JSON files written by ``locomo/retrieval_eval.py`` or
``longmemeval/retrieval_eval.py`` (``--out``), pairs questions on
``(conv, q)``, and reports:

* a per-category table (LOCOMO categories by name, including multi-hop):
  exact McNemar for every binary metric (``hit@k``, ``complete@k``), Wilcoxon
  signed-rank for ``recall@k`` and ``mrr``;
* overall McNemar with the discordant counts, the two-sided p-value and the
  one-sided p-value in the REGRESSION direction (B worse than A) -- the number a
  "no significant regression" gate needs;
* ``--tost METRIC --margin M``: two one-sided tests that B is within +/-M of A
  on the paired per-question differences (Wilcoxon by default, ``--tost-test t``);
* ``--latency``: paired ``query_s`` -- N, each side's median with a bootstrap 95%
  CI (percentile method, fixed seed), the median paired delta with its CI, and a
  paired Wilcoxon p-value on the deltas;
* ``--rows``: the ``n_returned`` distribution (min / median / p90 / max) per side.

Usage::

    python retrieval_paired_compare.py BEFORE.json AFTER.json [--label-a before --label-b after]
        [--metrics hit@10,hit@50,recall@10,mrr] [--latency] [--rows]
        [--tost recall@10 --margin 0.01] [--json summary.json]

Every comparative claim needs N, the point estimates and a paired test
(CLAUDE.md statistical-significance rule); this prints all three. Requires
scipy (Wilcoxon, t) and numpy (bootstrap).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
Key = tuple[str, str]
Rows = dict[Key, dict[str, Any]]


def load(path: str) -> tuple[dict[str, Any], Rows]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {(str(r["conv"]), str(r.get("q", 0))): r for r in payload["per_question"]}
    return payload, rows


def category(row: dict[str, Any]) -> str:
    cat = row.get("category")
    return LOCOMO_CATEGORIES.get(cat, str(cat)) if isinstance(cat, int) else str(cat)


def mcnemar(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar p-value on the discordant pairs."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


def mcnemar_regression(a_only: int, b_only: int) -> float:
    """One-sided exact p-value that B loses more questions than it gains (H1: B worse)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(a_only, n + 1)) / 2**n


def discordant(keys: list[Key], a: Rows, b: Rows, metric: str) -> tuple[int, int]:
    a_only = sum(1 for i in keys if a[i][metric] and not b[i][metric])
    b_only = sum(1 for i in keys if b[i][metric] and not a[i][metric])
    return a_only, b_only


def wilcoxon_p(diffs: list[float], alternative: str = "two-sided") -> float:
    """Paired Wilcoxon signed-rank p-value; 1.0 when every difference is zero."""
    from scipy.stats import wilcoxon

    if not any(diffs):
        return 1.0
    return float(wilcoxon(diffs, zero_method="wilcox", alternative=alternative).pvalue)


def is_binary(metric: str) -> bool:
    """A 0/1 per-question outcome, so a paired comparison is exact McNemar on the discordant pairs."""
    return metric.startswith(("hit@", "complete@"))


def paired_p(keys: list[Key], a: Rows, b: Rows, metric: str) -> float:
    if is_binary(metric):
        return mcnemar(*discordant(keys, a, b, metric))
    return wilcoxon_p([float(b[i][metric]) - float(a[i][metric]) for i in keys])


def tost(diffs: list[float], margin: float, test: str = "wilcoxon") -> dict[str, float]:
    """Two one-sided tests: B - A lies inside (-margin, +margin). Equivalent when p < alpha."""
    if margin <= 0:
        raise ValueError("TOST margin must be positive")
    if test == "t":
        from scipy.stats import ttest_1samp

        p_lower = float(ttest_1samp(diffs, -margin, alternative="greater").pvalue)
        p_upper = float(ttest_1samp(diffs, margin, alternative="less").pvalue)
    else:
        p_lower = wilcoxon_p([d + margin for d in diffs], alternative="greater")
        p_upper = wilcoxon_p([d - margin for d in diffs], alternative="less")
    return {"mean_diff": statistics.fmean(diffs), "p_lower": p_lower, "p_upper": p_upper, "p": max(p_lower, p_upper)}


def bootstrap_median_ci(values: list[float], *, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile-method bootstrap 95% CI on the median (deterministic for a seed)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    medians = np.median(rng.choice(arr, size=(n_boot, arr.size), replace=True), axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return float(lo), float(hi)


def latency(keys: list[Key], a: Rows, b: Rows, *, n_boot: int, seed: int) -> dict[str, Any] | None:
    keys = [i for i in keys if a[i].get("query_s") is not None and b[i].get("query_s") is not None]
    if not keys:
        return None
    qa = [float(a[i]["query_s"]) for i in keys]
    qb = [float(b[i]["query_s"]) for i in keys]
    delta = [y - x for x, y in zip(qa, qb, strict=True)]
    return {
        "n": len(keys),
        "median_a": statistics.median(qa),
        "median_a_ci": bootstrap_median_ci(qa, n_boot=n_boot, seed=seed),
        "median_b": statistics.median(qb),
        "median_b_ci": bootstrap_median_ci(qb, n_boot=n_boot, seed=seed),
        "median_delta": statistics.median(delta),
        "median_delta_ci": bootstrap_median_ci(delta, n_boot=n_boot, seed=seed),
        "wilcoxon_p": wilcoxon_p(delta),
    }


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]  # nearest-rank
    return {"min": ordered[0], "median": statistics.median(ordered), "p90": p90, "max": ordered[-1]}


def rows_returned(keys: list[Key], a: Rows, b: Rows) -> dict[str, Any] | None:
    keys = [i for i in keys if "n_returned" in a[i] and "n_returned" in b[i]]
    if not keys:
        return None
    return {
        "n": len(keys),
        "a": distribution([a[i]["n_returned"] for i in keys]),
        "b": distribution([b[i]["n_returned"] for i in keys]),
    }


def default_metrics(payload: dict[str, Any], rows: Rows) -> list[str]:
    ks = payload.get("k") or [10, 50]
    sample = next(iter(rows.values()), {})
    names = [f"{m}@{k}" for m in ("hit", "complete", "recall") for k in ks] + ["mrr"]
    return [m for m in names if m in sample]


def compare(
    a: Rows,
    b: Rows,
    metrics: list[str],
    *,
    tost_metric: str | None = None,
    margin: float | None = None,
    tost_test: str = "wilcoxon",
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Every figure the CLI prints, as data (the importable entry point)."""
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("the two runs share no (conv, q) question ids")
    groups: dict[str, list[Key]] = defaultdict(list)
    for i in keys:
        groups[category(a[i])].append(i)
    groups = {**dict(sorted(groups.items())), "ALL": keys}
    table = {
        g: {
            m: {
                "a": statistics.fmean(float(a[i][m]) for i in ks),
                "b": statistics.fmean(float(b[i][m]) for i in ks),
                "p": paired_p(ks, a, b, m),
            }
            for m in metrics
        }
        | {"n": len(ks)}
        for g, ks in groups.items()
    }
    overall = {}
    for m in metrics:
        if is_binary(m):
            a_only, b_only = discordant(keys, a, b, m)
            overall[m] = {
                "a_only": a_only,
                "b_only": b_only,
                "p_two_sided": mcnemar(a_only, b_only),
                "p_regression": mcnemar_regression(a_only, b_only),
            }
    out: dict[str, Any] = {"n": len(keys), "unpaired_a": len(a) - len(keys), "unpaired_b": len(b) - len(keys)}
    out |= {"table": table, "mcnemar": overall}
    out["latency"] = latency(keys, a, b, n_boot=n_boot, seed=seed)
    out["n_returned"] = rows_returned(keys, a, b)
    if tost_metric is not None:
        if margin is None:
            raise ValueError("--tost needs --margin")
        diffs = [float(b[i][tost_metric]) - float(a[i][tost_metric]) for i in keys]
        out["tost"] = {"metric": tost_metric, "margin": margin, "test": tost_test} | tost(diffs, margin, tost_test)
    return out


def render(result: dict[str, Any], metrics: list[str], label_a: str, label_b: str, args: argparse.Namespace) -> None:
    print(f"A={label_a}  B={label_b}  paired n={result['n']}", end="")
    if result["unpaired_a"] or result["unpaired_b"]:
        print(f"  (unpaired dropped: A {result['unpaired_a']}, B {result['unpaired_b']})", end="")
    print("\ncells: A% > B%, p (McNemar for hit@k and complete@k, Wilcoxon otherwise)")
    print(f"{'category':26s} {'n':>4s} " + " ".join(f"{m:>18s}" for m in metrics))
    for group, row in result["table"].items():
        cells = [f"{100 * row[m]['a']:5.1f}>{100 * row[m]['b']:5.1f} p{row[m]['p']:.3f}" for m in metrics]
        print(f"{group:26s} {row['n']:4d} " + " ".join(f"{c:>18s}" for c in cells))
    for m, r in result["mcnemar"].items():
        print(
            f"{m}: A-only={r['a_only']} B-only={r['b_only']} McNemar p={r['p_two_sided']:.4f} "
            f"(one-sided, B worse: p={r['p_regression']:.4f})"
        )
    if "tost" in result:
        t = result["tost"]
        verdict = "equivalent" if t["p"] < args.alpha else "NOT shown equivalent"
        print(
            f"TOST {t['metric']} ({t['test']}, margin +/-{t['margin']}): mean B-A={t['mean_diff']:+.4f} "
            f"p_lower={t['p_lower']:.4f} p_upper={t['p_upper']:.4f} -> {verdict} at alpha={args.alpha}"
        )
    if args.latency:
        lat = result["latency"]
        if lat is None:
            print("latency: no query_s on both sides")
        else:
            ms = [1000 * x for x in (lat["median_a"], *lat["median_a_ci"], lat["median_b"], *lat["median_b_ci"])]
            dl = [1000 * x for x in (lat["median_delta"], *lat["median_delta_ci"])]
            print(
                f"latency n={lat['n']}: median A={ms[0]:.1f}ms [{ms[1]:.1f}, {ms[2]:.1f}] "
                f"B={ms[3]:.1f}ms [{ms[4]:.1f}, {ms[5]:.1f}]; median paired delta={dl[0]:+.1f}ms "
                f"[{dl[1]:+.1f}, {dl[2]:+.1f}] (bootstrap 95% CI); Wilcoxon p={lat['wilcoxon_p']:.4f}"
            )
    if args.rows:
        dist = result["n_returned"]
        if dist is None:
            print("n_returned: not recorded on both sides")
        else:
            for side, label in (("a", label_a), ("b", label_b)):
                d = dist[side]
                print(
                    f"n_returned {label}: min={d['min']} median={d['median']} p90={d['p90']} max={d['max']} "
                    f"(n={dist['n']})"
                )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("a", help="baseline per-question JSON (A)")
    p.add_argument("b", help="candidate per-question JSON (B)")
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument(
        "--metrics", default="", help="comma list; default: hit@k, complete@k, recall@k for the run's k, and mrr"
    )
    p.add_argument("--tost", default=None, metavar="METRIC", help="TOST equivalence on this metric")
    p.add_argument("--margin", type=float, default=None, help="TOST equivalence margin (metric units)")
    p.add_argument("--tost-test", choices=("wilcoxon", "t"), default="wilcoxon")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--latency", action="store_true", help="paired query_s comparison")
    p.add_argument("--rows", action="store_true", help="n_returned distribution per side")
    p.add_argument("--bootstrap", type=int, default=10_000, help="bootstrap resamples for the latency CIs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default="", help="also write the full result here")
    args = p.parse_args(argv)
    pa, a = load(args.a)
    pb, b = load(args.b)
    metrics = [m for m in args.metrics.split(",") if m] or default_metrics(pa, a)
    try:
        result = compare(
            a, b, metrics, tost_metric=args.tost, margin=args.margin, tost_test=args.tost_test,
            n_boot=args.bootstrap, seed=args.seed,
        )  # fmt: skip
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    label_a = args.label_a or pa.get("label") or Path(args.a).stem
    label_b = args.label_b or pb.get("label") or Path(args.b).stem
    render(result, metrics, label_a, label_b, args)
    if args.json:
        Path(args.json).write_text(json.dumps({"a": label_a, "b": label_b, "metrics": metrics} | result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
