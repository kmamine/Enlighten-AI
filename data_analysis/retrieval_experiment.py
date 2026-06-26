"""Test-driven A/B evaluation of transcript transforms for RAG retrieval.

Variants (only the *indexed text* differs; chunking params, embedder, retriever,
queries and gold labels are held constant):
  * B0  baseline   — raw transcript chunks (current production behaviour)
  * V1  clean      — deterministic per-segment cleaning (transforms.clean_segments)
  * V2  multirep   — baseline chunks augmented with an LLM query-surrogate in BOTH
                     the dense and lexical representations; the original chunk text
                     stays the stored/cited document.

Two independent eval sets (fixed across variants):
  * title   — each indexed video's title is the query, gold = that video (neutral
              re: question phrasing, but easy).
  * synth   — Gemma-generated natural user questions per video (realistic; the
              question-style surrogate of V2 could be flattered here, hence we also
              report `title` as an independent check).

Metric: video-level recall@k and reciprocal rank over hybrid (dense+BM25+RRF) and
dense-only retrieval. Significance: paired bootstrap 95% CI on the per-query
reciprocal-rank delta vs baseline.

Run:  python -m data_analysis.retrieval_experiment
Caches eval queries and surrogates under data/exp/ so re-runs are cheap.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
from DrK_Chat.chunking import chunk_segments
from DrK_Chat.embeddings import Embedder
from DrK_Chat import transforms

EXP_DIR = config.DATA_DIR / "exp"
EXP_CHROMA = config.DATA_DIR / "chroma_exp"
REPORT = config.ROOT / "data_analysis" / "retrieval_experiment.md"
FIG_DIR = config.ROOT / "data_analysis" / "figures"

N_EVAL_VIDEOS = 60      # videos sampled for synthetic-QA queries
Q_PER_VIDEO = 2
POOL = 50               # chunk candidates per retriever before video-dedup
KS = [1, 3, 5, 10]
_TOK = re.compile(r"[a-z0-9']+")


def _tokenize(t):
    return _TOK.findall((t or "").lower())


# --- data -------------------------------------------------------------------
def load_segments_with_meta():
    """Videos that have a segments JSON (chunkable + cleanable)."""
    out = []
    for p in sorted(glob.glob(str(config.SEGMENTS_DIR / "*.json"))):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if d.get("segments"):
            out.append(d)
    return out


def client():
    from openai import OpenAI
    return OpenAI(base_url=config.VLLM_BASE_URL, api_key=config.VLLM_API_KEY)


# --- eval queries -----------------------------------------------------------
def build_eval_queries(segs, embedder):
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    cache = EXP_DIR / "eval_queries.json"
    title_q = [{"query": s["video_title"], "gold": s["video_id"], "set": "title"}
               for s in segs if s.get("video_title")]

    if cache.exists():
        synth = json.loads(cache.read_text())
    else:
        rng = np.random.default_rng(0)
        sample = [segs[i] for i in rng.choice(len(segs), min(N_EVAL_VIDEOS, len(segs)), replace=False)]
        cl = client()
        sys_msg = ("You write realistic questions a struggling person would type into a "
                   "mental-health self-help chatbot. Given a transcript excerpt, output "
                   f"{Q_PER_VIDEO} natural questions it would help answer. Do NOT quote the "
                   "transcript or use its exact phrasing. One question per line, no numbering.")
        synth = []
        for s in sample:
            excerpt = " ".join(seg["text"] for seg in s["segments"][:40])[:1800]
            try:
                r = cl.chat.completions.create(
                    model=config.VLLM_MODEL, temperature=0.7, max_tokens=120,
                    messages=[{"role": "system", "content": sys_msg},
                              {"role": "user", "content": f"Excerpt:\n{excerpt}\n\nQuestions:"}])
                for line in (r.choices[0].message.content or "").splitlines():
                    q = line.strip().lstrip("0123456789.-) ").strip()
                    if len(q) > 8:
                        synth.append({"query": q, "gold": s["video_id"], "set": "synth"})
            except Exception as e:
                print(f"  query-gen failed for {s['video_id']}: {e}")
        cache.write_text(json.dumps(synth, ensure_ascii=False, indent=0))
    print(f"Eval queries: {len(title_q)} title, {len(synth)} synthetic")
    return title_q, synth


# --- chunk records per variant ---------------------------------------------
def base_chunks(segs, embedder):
    """List of (chunk_id, text, video_id) from RAW segments."""
    recs = []
    for s in segs:
        for i, ch in enumerate(chunk_segments(s["segments"], count_tokens=embedder.count_tokens)):
            recs.append((f"{s['video_id']}:{i}", ch["text"], s["video_id"]))
    return recs


def clean_chunks(segs, embedder):
    recs = []
    for s in segs:
        cleaned = transforms.clean_segments(s["segments"])
        for i, ch in enumerate(chunk_segments(cleaned, count_tokens=embedder.count_tokens)):
            recs.append((f"{s['video_id']}:{i}", ch["text"], s["video_id"]))
    return recs


def load_or_make_surrogates(base):
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    cache = EXP_DIR / "surrogates.json"
    sur = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [(cid, txt) for cid, txt, _ in base if cid not in sur]
    if missing:
        print(f"Generating {len(missing)} surrogates via {config.VLLM_MODEL} ...")
        texts = [t for _, t in missing]
        gen = transforms.generate_surrogates(texts, client(), config.VLLM_MODEL, max_workers=16)
        for (cid, _), g in zip(missing, gen):
            sur[cid] = g
        cache.write_text(json.dumps(sur, ensure_ascii=False))
    return sur


# --- a built, queryable variant index --------------------------------------
class Variant:
    def __init__(self, name, records, embedder):
        """records: list of (chunk_id, embed_text, bm25_text, doc_text, video_id)."""
        import chromadb
        from rank_bm25 import BM25Okapi

        self.name = name
        self.embedder = embedder
        self.vid_of = {r[0]: r[4] for r in records}
        EXP_CHROMA.mkdir(parents=True, exist_ok=True)
        cl = chromadb.PersistentClient(path=str(EXP_CHROMA))
        try:
            cl.delete_collection(name)
        except Exception:
            pass
        self.coll = cl.create_collection(name, metadata={"hnsw:space": "cosine"})
        ids = [r[0] for r in records]
        embs = embedder.embed_passages([r[1] for r in records])
        self.coll.add(ids=ids, embeddings=embs, metadatas=[{"v": r[4]} for r in records])
        self.ids = ids
        self._bm25 = BM25Okapi([_tokenize(r[2]) for r in records])

    def _dense(self, q, pool):
        res = self.coll.query(query_embeddings=[self.embedder.embed_query(q)],
                              n_results=min(pool, len(self.ids)))
        return (res.get("ids") or [[]])[0]

    def _sparse(self, q, pool):
        sc = self._bm25.get_scores(_tokenize(q))
        return [self.ids[i] for i in np.argsort(sc)[::-1][:pool]]

    def _videos(self, chunk_ids):
        out = []
        for cid in chunk_ids:
            v = self.vid_of.get(cid)
            if v and v not in out:
                out.append(v)
        return out

    def dense_videos(self, q, pool=POOL):
        return self._videos(self._dense(q, pool))

    def hybrid_videos(self, q, pool=POOL, rrf_k=60):
        dense, sparse = self._dense(q, pool), self._sparse(q, pool)
        score = defaultdict(float)
        for ranking in (dense, sparse):
            for rank, cid in enumerate(ranking, 1):
                score[cid] += 1.0 / (rrf_k + rank)
        ranked = sorted(score, key=lambda c: score[c], reverse=True)
        return self._videos(ranked)


# --- metrics ----------------------------------------------------------------
def reciprocal_rank(videos, gold):
    return 1.0 / (videos.index(gold) + 1) if gold in videos else 0.0


def evaluate(variant, queries, mode):
    rrs = []
    hits = {k: 0 for k in KS}
    for q in queries:
        vids = (variant.hybrid_videos(q["query"]) if mode == "hybrid"
                else variant.dense_videos(q["query"]))
        rrs.append(reciprocal_rank(vids, q["gold"]))
        for k in KS:
            if q["gold"] in vids[:k]:
                hits[k] += 1
    n = len(queries)
    return {"mrr": float(np.mean(rrs)), "rr": rrs,
            **{f"r@{k}": hits[k] / n for k in KS}}


def bootstrap_ci(diffs, n=5000):
    rng = np.random.default_rng(0)
    d = np.array(diffs)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def decide(lo, hi):
    if lo > 0:
        return "✅ INCLUDE"
    if hi < 0:
        return "❌ EXCLUDE"
    return "➖ INCONCLUSIVE"


# --- main -------------------------------------------------------------------
def run():
    segs = load_segments_with_meta()
    print(f"{len(segs)} videos with segments")
    embedder = Embedder()
    title_q, synth_q = build_eval_queries(segs, embedder)
    evalsets = {"title": title_q, "synth": synth_q}

    base = base_chunks(segs, embedder)
    clean = clean_chunks(segs, embedder)
    sur = load_or_make_surrogates(base)

    def aug(text, cid):
        s = sur.get(cid, "")
        return (text + "\n" + s) if s else text

    records = {
        "B0_baseline": [(cid, t, t, t, v) for cid, t, v in base],
        "V1_clean": [(cid, t, t, t, v) for cid, t, v in clean],
        "V2_multirep": [(cid, aug(t, cid), aug(t, cid), t, v) for cid, t, v in base],
    }

    n_chunks = {k: len(v) for k, v in records.items()}
    results = {}  # name -> mode -> evalset -> metrics
    variants = {}
    for name, recs in records.items():
        print(f"Building {name} ({len(recs)} chunks)...")
        variants[name] = Variant(name, recs, embedder)
        results[name] = {}
        for mode in ("dense", "hybrid"):
            results[name][mode] = {es: evaluate(variants[name], q, mode)
                                   for es, q in evalsets.items()}

    write_report(results, n_chunks, evalsets, sur, base)
    print(f"\nWrote {REPORT}")


def write_report(results, n_chunks, evalsets, sur, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["B0_baseline", "V1_clean", "V2_multirep"]
    base_mrr = {(m, es): results["B0_baseline"][m][es]["mrr"]
                for m in ("dense", "hybrid") for es in evalsets}

    # figure: hybrid MRR + R@5 per variant per eval set
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    w = 0.35
    x = np.arange(len(names))
    for j, es in enumerate(["synth", "title"]):
        axes[0].bar(x + (j - 0.5) * w, [results[n]["hybrid"][es]["mrr"] for n in names], w, label=es)
        axes[1].bar(x + (j - 0.5) * w, [results[n]["hybrid"][es]["r@5"] for n in names], w, label=es)
    axes[0].set(title="Hybrid MRR by variant", ylabel="MRR")
    axes[1].set(title="Hybrid Recall@5 by variant", ylabel="Recall@5")
    for ax in axes:
        ax.set_xticks(x, names, rotation=15)
        ax.legend()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "retrieval_experiment.png", dpi=120)
    plt.close(fig)

    surr_cov = sum(1 for cid, _, _ in base if sur.get(cid)) / max(len(base), 1)

    L = ["# Retrieval Experiment — do transcript transforms help?\n",
         "_Test-driven A/B of transcript transforms, evaluated by `data_analysis/"
         "retrieval_experiment.py` on a fixed query benchmark. Lower-noise design: only the "
         "indexed text changes between variants; chunking, embedder (`%s`), retriever and "
         "queries are constant._\n" % config.EMBED_MODEL,
         "## Variants",
         "- **B0 baseline** — raw transcript chunks (current production).",
         "- **V1 clean** — deterministic per-segment cleaning (repetition/filler collapse).",
         f"- **V2 multirep** — chunks augmented with an LLM query-surrogate in dense+lexical "
         f"representations (surrogate coverage {surr_cov:.0%}); original text stays the cited document.",
         "",
         "## Method",
         "Two fixed eval sets: **title** (video title as query) and **synth** (Gemma-generated "
         "user questions). Gold = the source video; we measure **video-level recall@k** and "
         "**reciprocal rank** over **hybrid** (dense+BM25+RRF) and **dense-only** retrieval. "
         "Significance = paired **bootstrap 95% CI** on the per-query reciprocal-rank delta vs B0 "
         "(positive ⇒ better than baseline).\n",
         f"Chunks indexed per variant: " +
         ", ".join(f"{k}={v}" for k, v in n_chunks.items()) + ".\n"]

    for mode in ("hybrid", "dense"):
        L.append(f"## Results — {mode} retrieval\n")
        L.append("| Variant | Eval | R@1 | R@3 | R@5 | R@10 | MRR | ΔMRR vs B0 | 95% CI | Verdict |")
        L.append("|---|---|--:|--:|--:|--:|--:|--:|--|--|")
        for es in ("synth", "title"):
            for n in names:
                r = results[n][mode][es]
                if n == "B0_baseline":
                    dm, ci, verdict = "—", "—", "—"
                else:
                    diffs = np.array(r["rr"]) - np.array(results["B0_baseline"][mode][es]["rr"])
                    lo, hi = bootstrap_ci(diffs)
                    dm = f"{r['mrr'] - base_mrr[(mode, es)]:+.3f}"
                    ci = f"[{lo:+.3f}, {hi:+.3f}]"
                    verdict = decide(lo, hi)
                L.append(f"| {n} | {es} | {r['r@1']:.3f} | {r['r@3']:.3f} | {r['r@5']:.3f} | "
                         f"{r['r@10']:.3f} | {r['mrr']:.3f} | {dm} | {ci} | {verdict} |")
        L.append("")

    L.append("### Summary figure\n\n![Retrieval experiment](figures/retrieval_experiment.png)\n")

    # automated per-choice recommendation (based on hybrid + synth, the realistic case)
    L.append("## Decision (auto-generated from the numbers)\n")
    for n in ["V1_clean", "V2_multirep"]:
        verdicts = {}
        for es in ("synth", "title"):
            diffs = np.array(results[n]["hybrid"][es]["rr"]) - np.array(results["B0_baseline"]["hybrid"][es]["rr"])
            lo, hi = bootstrap_ci(diffs)
            verdicts[es] = (lo, hi, decide(lo, hi))
        L.append(f"- **{n}** — hybrid: synth {verdicts['synth'][2]} (CI "
                 f"[{verdicts['synth'][0]:+.3f},{verdicts['synth'][1]:+.3f}]), "
                 f"title {verdicts['title'][2]} (CI [{verdicts['title'][0]:+.3f},{verdicts['title'][1]:+.3f}]).")
    L.append("\n_Decision rule: INCLUDE only if the bootstrap CI of ΔMRR excludes 0 on the realistic "
             "(synth) set and the title set agrees in direction; otherwise EXCLUDE / INCONCLUSIVE, "
             "weighed against cost (V2 needs an LLM call per chunk)._")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
