"""Offline component analysis over an already-ingested LOCOMO store.

Reads the entries + stored embeddings straight from the backend (no recall
pipeline) and scores several rankers on evidence hit@k so fusion strategies
can be compared in one pass without re-embedding:

    bm25          rank_bm25 over content tokens (as the pipeline does)
    dense         cosine over stored all-MiniLM vectors
    rrf@k         reciprocal-rank fusion of both at smoothing k
    combmax       max reciprocal rank
    + any ranker registered in RANKERS below

Usage::  python component_eval.py --store DIR [--conversations 0,1] [--k 10,50]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_eval import CATEGORY_NAMES, load_json, resolve_path

Ranking = list[tuple[str, float]]
_STOP_TEXT = (
    "what when where who whom which why how did do does is are was were the a an of to in on at for and or "
    "with about from by as that this these those it its his her their them they he she you your i me my we our "
    "have has had be been being"
)


def hit_at(ranked_dia: list[str], evidence: set[str], k: int) -> float:
    return 1.0 if any(d in evidence for d in ranked_dia[:k]) else 0.0


def recall_at(ranked_dia: list[str], evidence: set[str], k: int) -> float:
    if not evidence:
        return 0.0
    top = ranked_dia[:k]
    return sum(1 for d in evidence if d in top) / len(evidence)


async def load_conversation(store: str, ci: int) -> tuple[Any, list[Any], dict[str, list[float]], Any]:
    os.environ["MEMORY_STORAGE_PATH"] = resolve_path(store)
    from trw_memory.client import MemoryClient

    ns = f"project:locomo-{ci}"
    client = MemoryClient(ns, mode="local")
    backend = client._get_backend()
    entries = backend.list_entries(namespace=ns, limit=5000)
    embeddings = backend.get_stored_embeddings([e.id for e in entries])
    return client, entries, embeddings, client._get_embedder()


def build_rankers(rrf_ks: list[int]) -> dict[str, Callable[..., Ranking]]:
    from trw_memory.retrieval.bm25 import bm25_search
    from trw_memory.retrieval.dense import dense_search
    from trw_memory.retrieval.fusion import combmax_fuse, rrf_fuse

    def bm25(q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float]) -> Ranking:
        return bm25_search(q, entries, top_k=len(entries))

    def dense(q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float]) -> Ranking:
        return dense_search(q, [e.id for e in entries], query_embedding=qv, stored_embeddings=emb, top_k=len(entries))

    rankers: dict[str, Callable[..., Ranking]] = {"bm25": bm25, "dense": dense}
    for k in rrf_ks:

        def rrf(q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float], _k: int = k) -> Ranking:
            return rrf_fuse([bm25(q, entries, emb, qv), dense(q, entries, emb, qv)], k=_k)

        rankers[f"rrf@{k}"] = rrf

    def combmax(q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float]) -> Ranking:
        return combmax_fuse([bm25(q, entries, emb, qv), dense(q, entries, emb, qv)], k=5)

    rankers["combmax"] = combmax

    _STOP = frozenset(_STOP_TEXT.split())

    def bm25_nostop(q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float]) -> Ranking:
        kept = " ".join(t for t in q.split() if t.lower().strip("?.,!'\"") not in _STOP)
        return bm25_search(kept or q, entries, top_k=len(entries))

    def rrf5_nostop(q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float]) -> Ranking:
        return rrf_fuse([bm25_nostop(q, entries, emb, qv), dense(q, entries, emb, qv)], k=5)

    rankers["bm25-nostop"] = bm25_nostop
    rankers["rrf5-nostop"] = rrf5_nostop

    from trw_memory.retrieval.reranker import cross_encode_rerank

    def rerank50(q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float]) -> Ranking:
        fused = rrf_fuse([bm25(q, entries, emb, qv), dense(q, entries, emb, qv)], k=5)
        by_id = {e.id: e for e in entries}
        head = [by_id[eid] for eid, _ in fused[:50] if eid in by_id]
        tail = fused[50:]
        reranked = cross_encode_rerank(q, head)
        return [(e.id, 1.0 / (i + 1)) for i, e in enumerate(reranked)] + tail

    rankers["rrf5+ce50"] = rerank50

    def _ce_over(fused: Ranking, q: str, entries: list[Any], n: int) -> Ranking:
        by_id = {e.id: e for e in entries}
        head = [by_id[eid] for eid, _ in fused[:n] if eid in by_id]
        reranked = cross_encode_rerank(q, head)
        return [(e.id, 1.0 / (i + 1)) for i, e in enumerate(reranked)] + fused[n:]

    for n in (50, 100):

        def nostop_ce(q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float], _n: int = n) -> Ranking:
            return _ce_over(rrf5_nostop(q, entries, emb, qv), q, entries, _n)

        rankers[f"rrf5-nostop+ce{n}"] = nostop_ce

    # --- crude suffix stemming on both sides (researched ~ research, agencies ~ agency)
    from rank_bm25 import BM25Okapi

    from trw_memory.retrieval.bm25 import _QUERY_STOPWORDS, _normalize_text

    def _stem(tok: str) -> str:
        for suf in ("ing", "edly", "ed", "ies", "es", "ly", "s"):
            if len(tok) > len(suf) + 3 and tok.endswith(suf):
                base = tok[: -len(suf)]
                return base + "y" if suf == "ies" else base
        return tok

    _stem_cache: dict[int, tuple[Any, list[str]]] = {}

    def bm25_stem(q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float]) -> Ranking:
        key = id(entries)
        if key not in _stem_cache:
            corpus = [[_stem(t) for t in _normalize_text(f"{e.content} {e.detail}").split()] for e in entries]
            _stem_cache[key] = (BM25Okapi(corpus), [e.id for e in entries])
        model, ids = _stem_cache[key]
        toks = [_stem(t) for t in _normalize_text(q).split() if t not in _QUERY_STOPWORDS] or _normalize_text(q).split()
        scores = model.get_scores(toks)
        ranked = sorted(zip(ids, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [(eid, float(s)) for eid, s in ranked if s > 0]

    def rrf5_stem(q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float]) -> Ranking:
        return rrf_fuse([bm25_stem(q, entries, emb, qv), dense(q, entries, emb, qv)], k=5)

    def rrf5_stem_ce50(q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float]) -> Ranking:
        return _ce_over(rrf5_stem(q, entries, emb, qv), q, entries, 50)

    rankers["bm25-stem"] = bm25_stem
    rankers["rrf5-stem"] = rrf5_stem
    rankers["rrf5-stem+ce50"] = rrf5_stem_ce50

    # --- session-neighbour expansion: after reranking, pull in the turns
    # immediately before/after each of the top hits (same session), so a
    # multi-hop answer whose second half sits one turn away is in the window.
    def _neighbours(entries: list[Any]) -> dict[str, list[str]]:
        by_dia = {}
        for e in entries:
            d = (e.metadata or {}).get("dia_id", "")
            if d:
                by_dia[d] = e.id
        out: dict[str, list[str]] = {}
        for d, eid in by_dia.items():
            sess, _, idx = d.partition(":")
            if not idx.isdigit():
                continue
            i = int(idx)
            out[eid] = [by_dia[k] for k in (f"{sess}:{i - 1}", f"{sess}:{i + 1}") if k in by_dia]
        return out

    _nbr_cache: dict[int, dict[str, list[str]]] = {}

    def expand(fused: Ranking, entries: list[Any], head: int, width: int) -> Ranking:
        key = id(entries)
        if key not in _nbr_cache:
            _nbr_cache[key] = _neighbours(entries)
        nbr = _nbr_cache[key]
        seen: set[str] = set()
        out: Ranking = []
        for rank, (eid, sc) in enumerate(fused):
            if eid not in seen:
                seen.add(eid)
                out.append((eid, sc))
            if rank < head:
                for n in nbr.get(eid, [])[:width]:
                    if n not in seen:
                        seen.add(n)
                        out.append((n, sc))
        return out

    # --- document-frequency pruning of query tokens: a token in more than half
    # of the corpus (a speaker name in a two-person chat) carries no signal.
    from collections import Counter

    from trw_memory.retrieval.bm25 import _stem_token, _tokenize_entry

    _df_cache: dict[int, tuple[Any, list[str], Counter[str], int]] = {}

    def bm25_dfprune(
        q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float], _frac: float = 0.5
    ) -> Ranking:
        key = id(entries)
        if key not in _df_cache:
            corpus = [_tokenize_entry(e) for e in entries]
            df: Counter[str] = Counter()
            for toks in corpus:
                df.update(set(toks))
            _df_cache[key] = (BM25Okapi(corpus), [e.id for e in entries], df, len(corpus))
        model, ids, df, n = _df_cache[key]
        toks = [_stem_token(t) for t in _normalize_text(q).split() if t not in _QUERY_STOPWORDS] or _normalize_text(
            q
        ).split()
        pruned = [t for t in toks if df[t] <= _frac * n]
        scores = model.get_scores(pruned or toks)
        ranked = sorted(zip(ids, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [(eid, float(sc)) for eid, sc in ranked if sc > 0]

    for frac in (0.5, 0.25):

        def rrf_df(
            q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float], _f: float = frac
        ) -> Ranking:
            return rrf_fuse([bm25_dfprune(q, entries, emb, qv, _f), dense(q, entries, emb, qv)], k=5)

        def rrf_df_ce(
            q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float], _f: float = frac
        ) -> Ranking:
            return _ce_over(
                rrf_fuse([bm25_dfprune(q, entries, emb, qv, _f), dense(q, entries, emb, qv)], k=5), q, entries, 50
            )

        rankers[f"bm25-df{frac}"] = lambda q, e, m, v, _f=frac: bm25_dfprune(q, e, m, v, _f)
        rankers[f"rrf5-df{frac}"] = rrf_df
        rankers[f"rrf5-df{frac}+ce50"] = rrf_df_ce

    for head in (10, 20):

        def ce_nbr(q: str, entries: list[Any], emb: dict[str, list[float]], qv: list[float], _h: int = head) -> Ranking:
            return expand(rrf5_stem_ce50(q, entries, emb, qv), entries, _h, 2)

        rankers[f"rrf5-stem+ce50+nbr{head}"] = ce_nbr
    return rankers


async def run(args: argparse.Namespace) -> None:
    data = load_json(args.dataset)
    ks = [int(k) for k in args.k.split(",")]
    rankers = build_rankers([int(k) for k in args.rrf_k.split(",")])
    scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ci in [int(c) for c in args.conversations.split(",")]:
        client, entries, embeddings, embedder = await load_conversation(args.store, ci)
        dia_of = {e.id: (e.metadata or {}).get("dia_id", "") for e in entries}
        t = time.monotonic()
        for qa in data[ci]["qa"]:
            cat = qa.get("category")
            if cat not in (1, 2, 3, 4):
                continue
            evidence = set(qa.get("evidence", []))
            q = qa["question"]
            qv = embedder.embed(q)
            for name, fn in rankers.items():
                ranked = [dia_of[eid] for eid, _ in fn(q, entries, embeddings, qv)]
                rec = {"category": cat, "mrr": next((1 / (i + 1) for i, d in enumerate(ranked) if d in evidence), 0.0)}
                for k in ks:
                    rec[f"hit@{k}"] = hit_at(ranked, evidence, k)
                    rec[f"recall@{k}"] = recall_at(ranked, evidence, k)
                scores[name].append(rec)
        print(f"conv {ci}: {len(entries)} entries, scored in {time.monotonic() - t:.1f}s", file=sys.stderr)
        close = getattr(client, "close", None)
        if close is not None:
            res = close()
            if asyncio.iscoroutine(res):
                await res

    metrics = [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks] + ["mrr"]
    n = len(next(iter(scores.values())))
    print(f"\n== component eval | {n} questions | store={args.store}")
    print(f"{'ranker':12s}" + "".join(f"{m:>11s}" for m in metrics))
    for name, rows in scores.items():
        print(f"{name:12s}" + "".join(f"{statistics.mean(r[m] for r in rows) * 100:10.1f}%" for m in metrics))
    if args.by_category:
        for name, rows in scores.items():
            print(f"\n-- {name}")
            by: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for r in rows:
                by[r["category"]].append(r)
            for cat in sorted(by):
                print(
                    f"  {CATEGORY_NAMES[cat]:12s}"
                    + "".join(f"{statistics.mean(r[m] for r in by[cat]) * 100:10.1f}%" for m in metrics)
                    + f"  n={len(by[cat])}"
                )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parents[3] / "scratch/memory-benchmarks/datasets/locomo/locomo10.json"),
    )
    p.add_argument("--store", required=True)
    p.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--k", default="10,50")
    p.add_argument("--rrf-k", default="5,60")
    p.add_argument("--by-category", action="store_true")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
