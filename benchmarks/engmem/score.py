"""EngMem metrics.

Deliberately different from the LOCOMO scorers in two ways, both because the
product's failure modes are different from a chat benchmark's:

* **Completeness over first-hit.** ``hit@k`` and MRR both score one hit of three
  as a success. On LOCOMO that hid a multi-hop bottleneck for a whole round of
  work; here ``complete@k`` is reported beside them from the start.
* **Surfacing the wrong row is a cost, not a neutral.** ``forbidden_rate`` counts
  retired conventions, superseded facts and foreign-project rows that reached the
  agent. A memory system that answers well and occasionally hands over a
  superseded instruction is worse than one that stays quiet, and no retrieval
  metric in the LOCOMO family can express that.

Everything is computed from a ranked list of ids, so the same scorers grade any
arm -- trw-memory, BM25, grep or recency -- and any future one.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .schema import Query


@dataclass(frozen=True, slots=True)
class Scored:
    qid: str
    task: str
    ranked: tuple[str, ...]
    gold: tuple[str, ...]
    forbidden: tuple[str, ...]
    tokens: int  # what this arm would have injected into the agent's context
    ms: float = 0.0  # wall time for this one retrieval


def latency(scored: list[Scored]) -> dict[str, float]:
    """p50/p95/max in milliseconds. Reported beside quality because at enterprise
    scale a retrieval policy that wins on recall and costs a second per call is
    not a better memory system -- it is an unusable one, and the curve over store
    size is where that shows up."""
    if not scored:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    xs = sorted(s.ms for s in scored)
    return {
        "p50": xs[len(xs) // 2],
        "p95": xs[min(len(xs) - 1, int(0.95 * len(xs)))],
        "max": xs[-1],
    }


def _dcg(rels: Sequence[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at(s: Scored, k: int) -> float:
    gold = set(s.gold)
    if not gold:
        return 0.0
    rels = [1 if rid in gold else 0 for rid in s.ranked[:k]]
    ideal = [1] * min(len(gold), k)
    denom = _dcg(ideal)
    return _dcg(rels) / denom if denom else 0.0


def metrics_for(s: Scored, ks: Sequence[int]) -> dict[str, float]:
    gold, forb = set(s.gold), set(s.forbidden)
    out: dict[str, float] = {}
    first = next((i for i, rid in enumerate(s.ranked) if rid in gold), None)
    out["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    for k in ks:
        top = s.ranked[:k]
        hits = sum(1 for g in gold if g in top)
        out[f"hit@{k}"] = 1.0 if hits else 0.0
        out[f"recall@{k}"] = hits / len(gold) if gold else 0.0
        out[f"complete@{k}"] = 1.0 if gold and hits == len(gold) else 0.0
        # The cost side: a forbidden row inside the window the agent reads.
        out[f"forbidden@{k}"] = 1.0 if forb and any(r in forb for r in top) else 0.0
    out["ndcg@10"] = ndcg_at(s, 10)
    out["tokens"] = float(s.tokens)
    return out


def pairwise_order_accuracy(scored: list[Scored], successor_of: dict[str, str]) -> tuple[float, int]:
    """C3: is the current record ranked above the one it replaced?

    Only pairs where BOTH rows appear in the ranking are counted -- if the
    superseded row never surfaced, there is no ordering error to make, and
    counting it as a win would reward a system for failing to retrieve. That
    case is already penalised by ``forbidden@k``.
    """
    correct = total = 0
    for s in scored:
        pos = {rid: i for i, rid in enumerate(s.ranked)}
        for old, new in successor_of.items():
            if old in pos and new in pos:
                total += 1
                correct += pos[new] < pos[old]
    return (correct / total if total else 0.0), total


def aggregate(scored: list[Scored], ks: Sequence[int] = (1, 5, 10)) -> dict[str, dict[str, float]]:
    """Mean of each metric, overall and per task. Means only -- confidence
    intervals belong with the caller, which knows the clustering (by learning id
    for the inner loop) and must not pretend queries are independent."""
    by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    for s in scored:
        m = metrics_for(s, ks)
        by_task[s.task].append(m)
        by_task["ALL"].append(m)
    out: dict[str, dict[str, float]] = {}
    for task, rows in by_task.items():
        keys = rows[0].keys()
        out[task] = {k: statistics.mean(r[k] for r in rows) for k in keys}
        out[task]["n"] = float(len(rows))
    return out


def render(agg: dict[str, dict[str, float]], label: str, ks: Sequence[int] = (1, 5, 10)) -> str:
    cols = ["mrr", *(f"hit@{k}" for k in ks), *(f"complete@{k}" for k in ks), "ndcg@10", f"forbidden@{max(ks)}"]
    lines = [f"\n== {label}", f"{'task':6s}" + "".join(f"{c:>13s}" for c in cols) + f"{'tokens':>9s}{'n':>6s}"]
    for task in sorted(agg, key=lambda t: (t == "ALL", t)):
        row = agg[task]
        lines.append(
            f"{task:6s}"
            + "".join(f"{100 * row.get(c, 0.0):12.1f}%" for c in cols)
            + f"{row.get('tokens', 0.0):9.0f}{int(row['n']):6d}"
        )
    return "\n".join(lines)


def paired_mcnemar(a: list[Scored], b: list[Scored], metric: str, ks: Sequence[int] = (1, 5, 10)) -> dict[str, Any]:
    """Exact McNemar on a binary metric between two arms over the same queries.
    Binary only -- passing it a rate would silently compare means."""
    from math import comb

    am = {s.qid: metrics_for(s, ks)[metric] for s in a}
    bm = {s.qid: metrics_for(s, ks)[metric] for s in b}
    shared = sorted(set(am) & set(bm))
    a_only = sum(1 for q in shared if am[q] > bm[q])
    b_only = sum(1 for q in shared if bm[q] > am[q])
    n = a_only + b_only
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2.0**n)
    return {
        "n": len(shared),
        "a_only": a_only,
        "b_only": b_only,
        "delta_pp": 100.0 * (b_only - a_only) / len(shared) if shared else 0.0,
        "p": p,
    }


def label_precision(verified: list[Query]) -> dict[str, Any]:
    """Of the human-checked sample, what fraction of silver labels held up?
    Reported as a Wilson lower bound, because every git-derived rate in this
    suite is an estimate of an estimate."""
    n = len(verified)
    if not n:
        return {"n": 0, "precision": None, "wilson_low": None}
    good = sum(1 for q in verified if q.context.get("label_verified") is True)
    return {"n": n, "precision": good / n, "wilson_low": wilson(good, n)[0]}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate; ``(0.0, 1.0)`` when ``n`` is zero."""
    if not n:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
