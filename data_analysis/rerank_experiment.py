"""A/B test: does a cross-encoder reranker improve retrieval over hybrid alone?

Runs on the **production** index (cleaned, V1) so it measures the real pipeline.
For each eval query we take the top-RERANK_POOL hybrid candidates, then compare:
  * base    — hybrid (dense+BM25+RRF) order, as the bot currently uses
  * rerank  — those same candidates reordered by the cross-encoder

Eval sets are reused for comparability with the transform experiment: the cached
synthetic user questions (data/exp/eval_queries.json) plus video-title queries.
Metric: video-level recall@k and reciprocal rank; significance via paired
bootstrap 95% CI on the per-query RR delta (rerank − base).

Run:  python -m data_analysis.rerank_experiment
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from DrK_Chat.retrieval import Retriever
from DrK_Chat.rerank import Reranker

EXP_DIR = config.DATA_DIR / "exp"
REPORT = config.ROOT / "data_analysis" / "rerank_experiment.md"
FIG_DIR = config.ROOT / "data_analysis" / "figures"
KS = [1, 3, 5, 10]
POOL = config.RERANK_POOL


def dedup_videos(cands):
    out = []
    for c in cands:
        v = c["metadata"].get("video_id")
        if v and v not in out:
            out.append(v)
    return out


def recip_rank(videos, gold):
    return 1.0 / (videos.index(gold) + 1) if gold in videos else 0.0


def load_queries(retriever):
    # synthetic (cached, same as the transform experiment)
    synth = []
    cache = EXP_DIR / "eval_queries.json"
    if cache.exists():
        synth = [q for q in json.loads(cache.read_text()) if q.get("set") == "synth"]
    # title queries from videos present in the production index
    indexed = {m.get("video_id") for m in (retriever.by_id[i]["metadata"] for i in retriever.ids)}
    titles = {m.get("video_id"): m.get("video_title")
              for m in (retriever.by_id[i]["metadata"] for i in retriever.ids)}
    title_q = [{"query": titles[v], "gold": v, "set": "title"}
               for v in indexed if titles.get(v)]
    return {"synth": synth, "title": title_q}


def bootstrap_ci(diffs, n=5000):
    rng = np.random.default_rng(0)
    d = np.array(diffs)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def decide(lo, hi):
    return "✅ INCLUDE" if lo > 0 else ("❌ EXCLUDE" if hi < 0 else "➖ INCONCLUSIVE")


def evaluate(retriever, reranker, queries):
    base_rr, rr_rr = [], []
    base_hits = {k: 0 for k in KS}
    rr_hits = {k: 0 for k in KS}
    rerank_latency = []
    for q in queries:
        cands = retriever.hybrid_search(q["query"], k=POOL, pool=POOL)
        base_videos = dedup_videos(cands)
        t = time.time()
        reranked = reranker.rerank(q["query"], cands)
        rerank_latency.append(time.time() - t)
        rr_videos = dedup_videos(reranked)
        base_rr.append(recip_rank(base_videos, q["gold"]))
        rr_rr.append(recip_rank(rr_videos, q["gold"]))
        for k in KS:
            base_hits[k] += q["gold"] in base_videos[:k]
            rr_hits[k] += q["gold"] in rr_videos[:k]
    n = len(queries)
    return {
        "n": n,
        "base": {"mrr": float(np.mean(base_rr)), "rr": base_rr,
                 **{f"r@{k}": base_hits[k] / n for k in KS}},
        "rerank": {"mrr": float(np.mean(rr_rr)), "rr": rr_rr,
                   **{f"r@{k}": rr_hits[k] / n for k in KS}},
        "latency_ms": float(np.median(rerank_latency) * 1000),
    }


def run():
    retriever = Retriever()
    print(f"Production index: {retriever.size} chunks")
    reranker = Reranker()
    print(f"Reranker: {reranker.model_name}")
    qs = load_queries(retriever)
    results = {es: evaluate(retriever, reranker, q) for es, q in qs.items() if q}
    write_report(results)
    print(f"Wrote {REPORT}")


def write_report(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = ["# Reranker Experiment — does a cross-encoder help?\n",
         f"_A/B on the **production** (cleaned) index: hybrid vs hybrid+rerank, reranker "
         f"`{config.RERANK_MODEL}` over the top-{POOL} hybrid candidates. Evaluated by "
         "`data_analysis/rerank_experiment.py`. Significance = paired bootstrap 95% CI on the "
         "per-query reciprocal-rank delta (rerank − hybrid)._\n",
         "## Results",
         "| Eval | Stage | R@1 | R@3 | R@5 | R@10 | MRR | ΔMRR | 95% CI | Verdict |",
         "|---|---|--:|--:|--:|--:|--:|--:|--|--|"]
    fig_rows = {}
    for es, r in results.items():
        b, rr = r["base"], r["rerank"]
        diffs = np.array(rr["rr"]) - np.array(b["rr"])
        lo, hi = bootstrap_ci(diffs)
        fig_rows[es] = (b["mrr"], rr["mrr"])
        L.append(f"| {es} (n={r['n']}) | hybrid | {b['r@1']:.3f} | {b['r@3']:.3f} | {b['r@5']:.3f} | "
                 f"{b['r@10']:.3f} | {b['mrr']:.3f} | — | — | — |")
        L.append(f"| {es} | +rerank | {rr['r@1']:.3f} | {rr['r@3']:.3f} | {rr['r@5']:.3f} | "
                 f"{rr['r@10']:.3f} | {rr['mrr']:.3f} | {rr['mrr'] - b['mrr']:+.3f} | "
                 f"[{lo:+.3f}, {hi:+.3f}] | {decide(lo, hi)} |")
    L.append("")
    med_lat = np.median([r["latency_ms"] for r in results.values()])
    L.append(f"**Added latency:** ~{med_lat:.0f} ms/query median to rerank {POOL} candidates "
             f"on `{config.EMBED_DEVICE}`.\n")

    # figure
    fig, ax = plt.subplots(figsize=(7, 4.2))
    es_list = list(fig_rows)
    x = np.arange(len(es_list))
    w = 0.35
    ax.bar(x - w / 2, [fig_rows[e][0] for e in es_list], w, label="hybrid")
    ax.bar(x + w / 2, [fig_rows[e][1] for e in es_list], w, label="hybrid+rerank")
    ax.set(title="MRR: hybrid vs hybrid+rerank", ylabel="MRR")
    ax.set_xticks(x, es_list)
    ax.legend()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rerank_experiment.png", dpi=120)
    plt.close(fig)
    L.append("![Reranker](figures/rerank_experiment.png)\n")

    L.append("## Decision\n")
    syn = results.get("synth")
    if syn:
        diffs = np.array(syn["rerank"]["rr"]) - np.array(syn["base"]["rr"])
        lo, hi = bootstrap_ci(diffs)
        L.append(f"- On realistic (synth) queries the reranker is **{decide(lo, hi)}** "
                 f"(ΔMRR {syn['rerank']['mrr'] - syn['base']['mrr']:+.3f}, CI [{lo:+.3f}, {hi:+.3f}]). "
                 "Weigh any gain against the added per-query latency above.")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
