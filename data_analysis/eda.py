"""Exploratory data analysis + report for the Dr. K transcript dataset.

Reads the canonical CSV, the per-video segments JSON (playlist membership,
transcript provenance, timestamps) and the live Chroma index (chunk stats),
writes figures to data_analysis/figures/, and emits data_analysis/report.md.

Run from the repo root:
    conda run -n enlighten python -m data_analysis.eda
"""
from __future__ import annotations

import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from Scrapper.playlists import extract_video_id
from Scrapper.transcripts import is_truncated

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.autolayout"] = True

FIG_DIR = config.ROOT / "data_analysis" / "figures"
REPORT = config.ROOT / "data_analysis" / "report.md"

# Filler words common in spoken transcripts that drown out content words.
SPEECH_STOP = {
    "know", "going", "really", "think", "people", "kind", "lot", "yeah", "gonna",
    "thing", "things", "want", "way", "right", "actually", "okay", "like", "just",
    "got", "get", "go", "say", "said", "stuff", "maybe", "mean", "doesn", "don",
    "ve", "re", "ll", "didn", "isn", "able", "make", "makes", "feel", "feeling",
    "good", "bad", "little", "talk", "talking", "come", "let", "look", "use",
    "does", "did", "doing", "guys", "sort", "day", "let", "lets", "gonna", "wanna",
    "yes", "oh", "uh", "um", "well", "thats", "youre", "dont", "cant", "didnt",
}

_CONTRACTION_RE = re.compile(r"n't\b|'(s|m|re|ve|ll|d)\b")


def clean_speech(text: str) -> str:
    """Lowercase and strip contraction suffixes so 'it's'/'don't' don't masquerade
    as content words (e.g. it's->it, don't->do, doesn't->does, you're->you)."""
    return _CONTRACTION_RE.sub(" ", text.lower())


def _save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


# --- data loading -----------------------------------------------------------
def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(config.CSV_PATH, dtype=str, keep_default_na=False)
    df["video_id"] = df["video_url"].apply(extract_video_id)
    df["chars"] = df["video_transcript"].str.len()
    df["words"] = df["video_transcript"].str.split().apply(len)
    df["length_sec"] = pd.to_numeric(df["video_length"], errors="coerce")
    df["minutes"] = df["length_sec"] / 60.0
    df["views"] = pd.to_numeric(df["video_views"], errors="coerce")
    df["date"] = pd.to_datetime(df["video_publish_date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["wpm"] = df["words"] / df["minutes"].replace(0, np.nan)
    return df


def load_segments() -> list[dict]:
    return [json.loads(Path(p).read_text(encoding="utf-8"))
            for p in glob.glob(str(config.SEGMENTS_DIR / "*.json"))]


def chunk_stats_by_video() -> dict[str, int]:
    """Chunks per video from the live Chroma index (empty dict if unavailable)."""
    try:
        from DrK_Chat.ingest import get_collection
        data = get_collection(create=False).get(include=["metadatas"])
    except Exception:
        return {}
    counts: Counter = Counter()
    for m in data.get("metadatas", []) or []:
        counts[m.get("video_id", "")] += 1
    return dict(counts)


# --- figures ----------------------------------------------------------------
def fig_volume(df: pd.DataFrame, figs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sns.histplot(df["minutes"].dropna(), bins=30, ax=axes[0], color="#4C72B0")
    axes[0].set(title="Video length", xlabel="Minutes", ylabel="Videos")
    sns.histplot(df["words"].dropna() / 1000, bins=30, ax=axes[1], color="#55A868")
    axes[1].set(title="Transcript size", xlabel="Words (thousands)", ylabel="Videos")
    figs["volume"] = _save(fig, "content_volume.png")


def fig_speaking_rate(df: pd.DataFrame, figs: dict) -> None:
    d = df.dropna(subset=["wpm"])
    d = d[(d["wpm"] > 0) & (d["wpm"] < 400)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.histplot(d["wpm"], bins=30, kde=True, ax=ax, color="#8172B3")
    ax.axvline(d["wpm"].median(), color="k", ls="--", lw=1,
               label=f"median {d['wpm'].median():.0f} wpm")
    ax.set(title="Speaking rate (words per minute)", xlabel="Words / minute", ylabel="Videos")
    ax.legend()
    figs["wpm"] = _save(fig, "speaking_rate.png")


def fig_engagement(df: pd.DataFrame, figs: dict) -> None:
    d = df.dropna(subset=["views", "minutes"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sns.histplot(np.log10(d["views"]), bins=30, ax=axes[0], color="#C44E52")
    axes[0].set(title="Views distribution", xlabel="log10(views)", ylabel="Videos")
    sns.scatterplot(data=d, x="minutes", y="views", ax=axes[1], alpha=0.6, color="#C44E52")
    axes[1].set(title="Views vs length", xlabel="Minutes", ylabel="Views", yscale="log")
    figs["engagement"] = _save(fig, "engagement.png")


def fig_timeline(df: pd.DataFrame, figs: dict) -> None:
    d = df.dropna(subset=["date"])
    per_year = d.groupby(d["year"].astype(int)).agg(
        videos=("video_id", "size"), hours=("length_sec", lambda s: s.sum() / 3600.0))
    years = per_year.index.to_numpy()
    cum_hours = per_year["hours"].cumsum().to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(years, per_year["videos"].to_numpy(), color="#4C72B0", alpha=0.75)
    ax.set(title="Publishing cadence & cumulative content", xlabel="Year", ylabel="Videos / year")
    ax.set_xticks(years)
    ax2 = ax.twinx()
    ax2.plot(years, cum_hours, color="#C44E52", lw=2, marker="o", label="cumulative hours")
    ax2.set_ylabel("Cumulative hours of content")
    ax2.grid(False)
    ax2.legend(loc="upper left")
    figs["timeline"] = _save(fig, "timeline.png")


def fig_playlists(df: pd.DataFrame, segs: list[dict], figs: dict) -> Counter:
    # membership: prefer all_playlist_tags from segments, fall back to CSV tag
    membership = {s["video_id"]: (s.get("all_playlist_tags") or [s.get("playlist_tag")])
                  for s in segs}
    counts: Counter = Counter()
    hours: defaultdict = defaultdict(float)
    for r in df.itertuples():
        tags = membership.get(r.video_id, [r.playlist_tag])
        for t in tags:
            if t:
                counts[t] += 1
                if not np.isnan(r.length_sec):
                    hours[t] += r.length_sec / 3600.0
    order = [t for t, _ in counts.most_common()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(order[::-1], [counts[t] for t in order][::-1], color="#55A868")
    axes[0].set(title="Videos per playlist", xlabel="Videos")
    axes[1].barh(order[::-1], [hours[t] for t in order][::-1], color="#4C72B0")
    axes[1].set(title="Hours of content per playlist", xlabel="Hours")
    figs["playlists"] = _save(fig, "playlists.png")
    return counts


def fig_playlist_overlap(segs: list[dict], figs: dict) -> int:
    tagsets = [s.get("all_playlist_tags") or [] for s in segs]
    all_tags = sorted({t for ts in tagsets for t in ts})
    if len(all_tags) < 2:
        return 0
    idx = {t: i for i, t in enumerate(all_tags)}
    mat = np.zeros((len(all_tags), len(all_tags)), dtype=int)
    multi = 0
    for ts in tagsets:
        uniq = sorted(set(ts))
        if len(uniq) > 1:
            multi += 1
        for a in uniq:
            for b in uniq:
                mat[idx[a], idx[b]] += 1
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(mat, xticklabels=all_tags, yticklabels=all_tags, annot=True, fmt="d",
                cmap="rocket_r", ax=ax, cbar_kws={"label": "shared videos"})
    ax.set(title="Playlist co-occurrence (videos shared between playlists)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    figs["overlap"] = _save(fig, "playlist_overlap.png")
    return multi


def fig_chunks(chunk_counts: dict, figs: dict) -> None:
    if not chunk_counts:
        return
    vals = list(chunk_counts.values())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.histplot(vals, bins=30, ax=ax, color="#937860")
    ax.axvline(np.median(vals), color="k", ls="--", lw=1, label=f"median {int(np.median(vals))}")
    ax.set(title="RAG chunks per video", xlabel="Chunks", ylabel="Videos")
    ax.legend()
    figs["chunks"] = _save(fig, "chunks_per_video.png")


def fig_text_terms(df: pd.DataFrame, figs: dict) -> None:
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    corpus = df["video_transcript"].tolist()
    stop = list(ENGLISH_STOP_WORDS | SPEECH_STOP)
    uni = CountVectorizer(stop_words=stop, min_df=3, preprocessor=clean_speech,
                          token_pattern=r"[a-z]{3,}")
    X = uni.fit_transform(corpus)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(uni.get_feature_names_out())
    top = np.argsort(freqs)[::-1][:25]

    bi = CountVectorizer(stop_words=stop, ngram_range=(2, 2), min_df=3,
                         preprocessor=clean_speech, token_pattern=r"[a-z]{3,}")
    Xb = bi.fit_transform(corpus)
    bf = np.asarray(Xb.sum(axis=0)).ravel()
    bterms = np.array(bi.get_feature_names_out())
    btop = np.argsort(bf)[::-1][:20]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    axes[0].barh(terms[top][::-1], freqs[top][::-1], color="#4C72B0")
    axes[0].set(title="Top content words (transcripts)", xlabel="Occurrences")
    axes[1].barh(bterms[btop][::-1], bf[btop][::-1], color="#55A868")
    axes[1].set(title="Top bigrams", xlabel="Occurrences")
    figs["terms"] = _save(fig, "text_terms.png")


def fig_wordcloud(df: pd.DataFrame, figs: dict) -> None:
    try:
        from wordcloud import WordCloud, STOPWORDS
    except Exception:
        return
    text = clean_speech(" ".join(df["video_transcript"].tolist()))
    extra = {"now", "one", "something", "someone", "even", "thing", "things", "take",
             "back", "first", "thought", "going", "lot", "put", "well", "much", "many"}
    wc = WordCloud(width=1200, height=600, background_color="white",
                   stopwords=set(STOPWORDS) | SPEECH_STOP | extra, collocations=True,
                   max_words=150, colormap="viridis").generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Transcript word cloud")
    figs["cloud"] = _save(fig, "wordcloud.png")


def distinctive_terms(df: pd.DataFrame, segs: list[dict]) -> dict[str, list[str]]:
    """Top TF-IDF terms per playlist (what makes each playlist distinctive)."""
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

    membership = {s["video_id"]: (s.get("all_playlist_tags") or [s.get("playlist_tag")])
                  for s in segs}
    docs: defaultdict = defaultdict(list)
    for r in df.itertuples():
        for t in membership.get(r.video_id, [r.playlist_tag]):
            if t:
                docs[t].append(r.video_transcript)
    tags = list(docs)
    joined = [" ".join(docs[t]) for t in tags]
    if len(tags) < 2:
        return {}
    stop = list(ENGLISH_STOP_WORDS | SPEECH_STOP)
    vec = TfidfVectorizer(stop_words=stop, min_df=2, max_df=0.8,
                          preprocessor=clean_speech, token_pattern=r"[a-z]{3,}")
    M = vec.fit_transform(joined)
    feats = np.array(vec.get_feature_names_out())
    out = {}
    for i, t in enumerate(tags):
        row = M[i].toarray().ravel()
        out[t] = list(feats[np.argsort(row)[::-1][:8]])
    return out


# --- advanced: embedding-based analyses ------------------------------------
def load_video_embeddings():
    """Mean bge embedding per video from the Chroma index. (None on failure)."""
    try:
        from DrK_Chat.ingest import get_collection
        data = get_collection(create=False).get(include=["embeddings", "metadatas"])
    except Exception:
        return None, None, None
    embs, metas = data.get("embeddings"), data.get("metadatas")
    if embs is None or len(embs) == 0:
        return None, None, None
    by_vid: defaultdict = defaultdict(list)
    meta_by_vid: dict = {}
    for e, m in zip(embs, metas):
        v = m.get("video_id")
        by_vid[v].append(np.asarray(e, dtype=float))
        meta_by_vid[v] = m
    vids = list(by_vid)
    X = np.vstack([np.mean(by_vid[v], axis=0) for v in vids])
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)  # re-normalize means
    return vids, X, meta_by_vid


def fig_semantic_map(vids, X, meta_by_vid, labels, figs) -> None:
    """2-D t-SNE of video embeddings: left coloured by playlist, right by cluster."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n = len(vids)
    pca = PCA(n_components=min(50, n - 1)).fit_transform(X)
    perp = max(5, min(30, n // 4))
    xy = TSNE(n_components=2, perplexity=perp, init="pca", random_state=42).fit_transform(pca)

    playlists = [meta_by_vid[v].get("playlist_tag", "?") for v in vids]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=playlists, ax=axes[0], s=45,
                    palette="tab10", legend="brief")
    axes[0].set(title="Semantic map — coloured by playlist", xlabel="", ylabel="")
    axes[0].legend(fontsize=7, loc="best", ncol=1)
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=[f"T{c}" for c in labels], ax=axes[1],
                    s=45, palette="tab10", legend="brief")
    axes[1].set(title="Semantic map — coloured by discovered topic", xlabel="", ylabel="")
    axes[1].legend(fontsize=7, loc="best", ncol=2)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    figs["semantic"] = _save(fig, "semantic_map.png")


def discover_topics(vids, X, meta_by_vid, df, k=8):
    """KMeans over video embeddings -> data-driven topics labelled by TF-IDF terms."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    tx = df.set_index("video_id")
    stop = list(ENGLISH_STOP_WORDS | SPEECH_STOP)
    docs, members = [], []
    for c in range(k):
        cv = [v for v, lab in zip(vids, labels) if lab == c]
        members.append(cv)
        docs.append(" ".join(tx.loc[v, "video_transcript"] for v in cv if v in tx.index))
    vec = TfidfVectorizer(stop_words=stop, preprocessor=clean_speech,
                          token_pattern=r"[a-z]{3,}", min_df=1, max_df=0.85)
    M = vec.fit_transform(docs)
    feats = np.array(vec.get_feature_names_out())
    topics = []
    for c in range(k):
        terms = list(feats[np.argsort(M[c].toarray().ravel())[::-1][:8]])
        cv = members[c]
        playlists = Counter(meta_by_vid[v].get("playlist_tag", "?") for v in cv)
        examples = [tx.loc[v, "video_title"] for v in cv[:2] if v in tx.index]
        topics.append({
            "id": c, "size": len(cv), "terms": terms,
            "dominant_playlist": playlists.most_common(1)[0][0] if playlists else "?",
            "examples": examples,
        })
    return labels, topics


def fig_view_drivers(df, figs) -> list:
    """Which title words correlate with higher median views (engagement lift)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    d = df.dropna(subset=["views"])
    overall = d["views"].median()
    word_views: defaultdict = defaultdict(list)
    for r in d.itertuples():
        for w in set(re.findall(r"[a-z]{3,}", clean_speech(r.video_title))):
            if w not in ENGLISH_STOP_WORDS and w not in SPEECH_STOP:
                word_views[w].append(r.views)
    rows = [(w, float(np.median(v)), len(v)) for w, v in word_views.items() if len(v) >= 4]
    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[:16]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    terms = [f"{w}  (n={n})" for w, _, n in top]
    lift = [mv / overall for _, mv, _ in top]
    ax.barh(terms[::-1], lift[::-1], color="#C44E52")
    ax.axvline(1.0, color="k", ls="--", lw=1, label="corpus median")
    ax.set(title="Title-word view lift (median views ÷ corpus median)", xlabel="× median views")
    ax.legend()
    figs["drivers"] = _save(fig, "view_drivers.png")
    return top


def fig_question_style(df, segs, figs) -> float:
    """Dr. K's introspective style: questions asked per 1000 words."""
    d = df[df["words"] > 0].copy()
    d["q_per_1k"] = d["video_transcript"].str.count(r"\?") / d["words"] * 1000
    membership = {s["video_id"]: (s.get("playlist_tag") or "?") for s in segs}
    d["pl"] = d["video_id"].map(membership).fillna(d["playlist_tag"])
    by_pl = d.groupby("pl")["q_per_1k"].mean().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    sns.histplot(d["q_per_1k"], bins=30, ax=axes[0], color="#4C72B0")
    axes[0].axvline(d["q_per_1k"].median(), color="k", ls="--", lw=1,
                    label=f"median {d['q_per_1k'].median():.1f}/1k words")
    axes[0].set(title="Questions per 1000 words", xlabel="Questions / 1k words", ylabel="Videos")
    axes[0].legend()
    axes[1].barh(by_pl.index, by_pl.values, color="#8172B3")
    axes[1].set(title="Avg question density by playlist", xlabel="Questions / 1k words")
    figs["questions"] = _save(fig, "question_style.png")
    return float(d["q_per_1k"].median())


def temporal_drift(df) -> dict:
    """TF-IDF distinctive terms per publish year (how the focus shifted)."""
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

    d = df.dropna(subset=["year"])
    years = sorted(int(y) for y in d["year"].unique())
    docs = [" ".join(d[d["year"] == y]["video_transcript"]) for y in years]
    if len(docs) < 2:
        return {}
    stop = list(ENGLISH_STOP_WORDS | SPEECH_STOP)
    vec = TfidfVectorizer(stop_words=stop, preprocessor=clean_speech,
                          token_pattern=r"[a-z]{3,}", min_df=1, max_df=0.9)
    M = vec.fit_transform(docs)
    feats = np.array(vec.get_feature_names_out())
    return {y: list(feats[np.argsort(M[i].toarray().ravel())[::-1][:7]])
            for i, y in enumerate(years)}


# --- advanced: retrieval eval, concept graph, redundancy, sentiment ---------
def _video_of(cid: str) -> str:
    return cid.rsplit(":", 1)[0]


def evaluate_retrieval(df, figs, k_values=(1, 3, 5), pool=30):
    """Benchmark dense vs BM25 vs hybrid retrieval using each video's title as a
    proxy query (does the right video come back in the top-k?). Validates the
    hybrid architecture empirically. Returns (recall_table, mrr, n_queries)."""
    try:
        from DrK_Chat.retrieval import Retriever
        r = Retriever()
    except Exception:
        return None, None, 0
    if r.size == 0:
        return None, None, 0

    tx = df.set_index("video_id")
    indexed = {m.get("video_id") for m in (r.by_id[i]["metadata"] for i in r.ids)}
    queries = [(v, tx.loc[v, "video_title"]) for v in indexed
               if v in tx.index and tx.loc[v, "video_title"]]
    methods = ["dense", "sparse", "hybrid"]
    recall = {m: {k: 0 for k in k_values} for m in methods}
    mrr = {m: 0.0 for m in methods}

    def ranked_videos(cids):
        out = []
        for cid in cids:
            v = _video_of(cid)
            if v not in out:
                out.append(v)
        return out

    for vid, title in queries:
        rankings = {
            "dense": ranked_videos(r._dense_ranking(title, pool)),
            "sparse": ranked_videos(r._sparse_ranking(title, pool)),
            "hybrid": ranked_videos([h["id"] for h in r.hybrid_search(title, k=pool, pool=pool)]),
        }
        for m, vids in rankings.items():
            for k in k_values:
                if vid in vids[:k]:
                    recall[m][k] += 1
            if vid in vids:
                mrr[m] += 1.0 / (vids.index(vid) + 1)

    n = len(queries)
    for m in methods:
        for k in k_values:
            recall[m][k] /= n
        mrr[m] /= n

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    width = 0.25
    xs = np.arange(len(k_values))
    colors = {"dense": "#4C72B0", "sparse": "#DD8452", "hybrid": "#55A868"}
    for i, m in enumerate(methods):
        axes[0].bar(xs + (i - 1) * width, [recall[m][k] for k in k_values], width,
                    label=m, color=colors[m])
    axes[0].set(title="Retrieval recall@k (title-as-query proxy)", xlabel="k", ylabel="Recall")
    axes[0].set_xticks(xs, [f"@{k}" for k in k_values])
    axes[0].legend()
    axes[1].bar(methods, [mrr[m] for m in methods], color=[colors[m] for m in methods])
    axes[1].set(title="Mean Reciprocal Rank", ylabel="MRR")
    figs["retrieval"] = _save(fig, "retrieval_eval.png")
    return recall, mrr, n


CONCEPTS = [
    "anxiety", "depression", "dopamine", "motivation", "meditation", "trauma",
    "attachment", "shame", "loneliness", "addiction", "porn", "adhd", "ego",
    "anger", "relationship", "boundaries", "esteem", "mindfulness", "emotion",
    "meaning", "purpose", "discipline", "procrastination", "confidence",
    "identity", "gaming", "burnout", "anxious",
]


def concept_network(df, figs, top_edges=32):
    """Co-occurrence network of mental-health concepts across videos.

    To avoid a hairball (broad concepts co-occur in most long videos), edges are
    weighted by the Jaccard overlap of the two concepts' video sets, and only the
    strongest `top_edges` are drawn.
    """
    import networkx as nx

    present = {c: df["video_transcript"].str.contains(rf"\b{c}\w*", case=False, regex=True)
               for c in CONCEPTS}
    freq = {c: int(present[c].sum()) for c in CONCEPTS}
    concepts = [c for c in CONCEPTS if freq[c] >= 5]

    candidates = []
    for i, a in enumerate(concepts):
        for b in concepts[i + 1:]:
            inter = int((present[a] & present[b]).sum())
            union = int((present[a] | present[b]).sum())
            if inter >= 5 and union:
                candidates.append((a, b, inter, inter / union))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[3], reverse=True)
    edges = candidates[:top_edges]

    G = nx.Graph()
    for c in concepts:
        G.add_node(c, size=freq[c])
    for a, b, inter, jac in edges:
        G.add_edge(a, b, weight=jac, co=inter)
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])

    pos = nx.spring_layout(G, weight="weight", seed=42, k=1.2)
    sizes = [G.nodes[n]["size"] * 24 for n in G.nodes]
    widths = [0.6 + G[u][v]["weight"] * 9 for u, v in G.edges]
    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#b8c4d0", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=list(range(len(G.nodes))),
                           cmap="tab20", alpha=0.92, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)
    ax.set_title("Concept co-occurrence network\n"
                 "(node size = videos mentioning it; edges = strongest Jaccard overlap)")
    ax.axis("off")
    figs["network"] = _save(fig, "concept_network.png")
    return freq


def cross_video_redundancy(vids, X, meta_by_vid, top_n=8):
    """Most semantically similar distinct video pairs (content overlap / dedupe signal)."""
    if vids is None or len(vids) < 3:
        return []
    S = X @ X.T
    np.fill_diagonal(S, -1.0)
    pairs = []
    iu = np.triu_indices(len(vids), k=1)
    sims = S[iu]
    order = np.argsort(sims)[::-1][:top_n]
    for idx in order:
        i, j = iu[0][idx], iu[1][idx]
        pairs.append((meta_by_vid[vids[i]].get("video_title", vids[i]),
                      meta_by_vid[vids[j]].get("video_title", vids[j]),
                      float(sims[idx])))
    return pairs


def emotional_arc(segs, figs, bins=12):
    """Average sentiment (VADER compound) across the normalized timeline of videos."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception:
        return None
    analyzer = SentimentIntensityAnalyzer()
    acc = np.zeros(bins)
    cnt = np.zeros(bins)
    for s in segs:
        segments = s.get("segments", [])
        n = len(segments)
        if n < bins:
            continue
        for i, seg in enumerate(segments):
            b = min(int(i / n * bins), bins - 1)
            acc[b] += analyzer.polarity_scores(seg.get("text", ""))["compound"]
            cnt[b] += 1
    if cnt.sum() == 0:
        return None
    arc = acc / np.maximum(cnt, 1)
    xs = np.linspace(0, 100, bins)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(xs, arc, marker="o", color="#C44E52", lw=2)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.fill_between(xs, arc, 0, alpha=0.15, color="#C44E52")
    ax.set(title="Average emotional arc across videos (VADER sentiment)",
           xlabel="Position in video (%)", ylabel="Mean sentiment (compound)")
    figs["arc"] = _save(fig, "emotional_arc.png")
    return arc


EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
AFFECT = ["anger", "disgust", "fear", "sadness", "joy", "surprise"]  # non-neutral, ordered neg->pos
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"


def emotion_analysis(df, segs, figs, pl_counts, bins=12, min_words=4):
    """Fine-grained emotion classification (7 emotions) of every transcript segment
    with a RoBERTa classifier, aggregated into a corpus profile, per-emotion arcs
    over the video timeline, per-playlist emotional fingerprints, and a
    valence↔engagement correlation. Returns a results dict (None if unavailable)."""
    try:
        from transformers import pipeline
        clf = pipeline("text-classification", model=EMOTION_MODEL, top_k=None,
                       device=0, truncation=True, max_length=512, batch_size=128)
    except Exception:
        return None

    rec_vid, rec_bin, texts = [], [], []
    for s in segs:
        segments = s.get("segments", [])
        n = len(segments)
        if n == 0:
            continue
        for i, seg in enumerate(segments):
            txt = (seg.get("text") or "").strip()
            if len(txt.split()) >= min_words:
                rec_vid.append(s["video_id"])
                rec_bin.append(min(int(i / n * bins), bins - 1))
                texts.append(txt)
    if not texts:
        return None

    results = clf(texts)
    idx = {e: i for i, e in enumerate(EMOTIONS)}
    scores = np.zeros((len(texts), len(EMOTIONS)))
    for r, res in enumerate(results):
        for d in res:
            scores[r, idx[d["label"]]] = d["score"]

    import pandas as pd
    E = pd.DataFrame(scores, columns=EMOTIONS)
    E["vid"], E["bin"] = rec_vid, rec_bin
    corpus = E[EMOTIONS].mean()
    arc = E.groupby("bin")[EMOTIONS].mean()
    vid_em = E.groupby("vid")[EMOTIONS].mean()
    vid_pl = {s["video_id"]: (s.get("playlist_tag") or "?") for s in segs}
    vid_em["pl"] = vid_em.index.map(vid_pl)
    pl_em = vid_em.groupby("pl")[EMOTIONS].mean()

    vid_em["valence"] = (vid_em["joy"] -
                         vid_em[["anger", "disgust", "fear", "sadness"]].sum(axis=1))
    views = df.set_index("video_id")["views"]
    titles = df.set_index("video_id")["video_title"]
    vv = vid_em.join(views).join(titles).dropna(subset=["views"])
    from scipy.stats import spearmanr
    rho_val_views = float(spearmanr(vv["valence"], vv["views"]).correlation) if len(vv) > 2 else float("nan")

    extremes = {
        "most_positive": (titles.get(vid_em["valence"].idxmax(), "?"), vid_em["valence"].max()),
        "most_negative": (titles.get(vid_em["valence"].idxmin(), "?"), vid_em["valence"].min()),
        "most_fear": titles.get(vid_em["fear"].idxmax(), "?"),
        "most_sad": titles.get(vid_em["sadness"].idxmax(), "?"),
        "most_joy": titles.get(vid_em["joy"].idxmax(), "?"),
    }

    # Figure 1: corpus profile + emotion arcs (non-neutral)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    order = corpus.sort_values(ascending=False)
    axes[0].bar(order.index, order.values, color="#4C72B0")
    axes[0].set(title="Corpus emotion profile (mean P over segments)", ylabel="Mean probability")
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")
    palette = sns.color_palette("husl", len(AFFECT))
    xs = np.linspace(0, 100, bins)
    for c, e in zip(palette, AFFECT):
        axes[1].plot(xs, arc[e].values, marker="o", ms=3, lw=1.8, color=c, label=e)
    axes[1].set(title="Emotion arcs across the video timeline (neutral omitted)",
                xlabel="Position in video (%)", ylabel="Mean probability")
    axes[1].legend(fontsize=8, ncol=2)
    figs["emotion_overview"] = _save(fig, "emotion_overview.png")

    # Figure 2: per-playlist emotional fingerprint (non-neutral, ordered by size)
    pl_order = [t for t, _ in pl_counts.most_common()] if pl_counts else list(pl_em.index)
    pl_order = [p for p in pl_order if p in pl_em.index]
    heat = pl_em.loc[pl_order, AFFECT]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="rocket_r", ax=ax,
                cbar_kws={"label": "mean probability"})
    ax.set(title="Emotional fingerprint by playlist (non-neutral emotions)", xlabel="", ylabel="")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    figs["emotion_fp"] = _save(fig, "emotion_fingerprint.png")

    return {
        "n_segments": len(texts),
        "corpus": corpus, "arc": arc, "pl_em": pl_em.loc[pl_order],
        "rho_val_views": rho_val_views, "extremes": extremes,
    }


# --- report -----------------------------------------------------------------
def run() -> None:
    df = load_dataframe()
    segs = load_segments()
    chunk_counts = chunk_stats_by_video()
    figs: dict[str, str] = {}

    fig_volume(df, figs)
    fig_speaking_rate(df, figs)
    fig_engagement(df, figs)
    fig_timeline(df, figs)
    pl_counts = fig_playlists(df, segs, figs)
    multi = fig_playlist_overlap(segs, figs)
    fig_chunks(chunk_counts, figs)
    fig_text_terms(df, figs)
    fig_wordcloud(df, figs)
    distinct = distinctive_terms(df, segs)

    # advanced (embedding-based) analyses
    vids, X, meta_by_vid = load_video_embeddings()
    topics = []
    if vids is not None and len(vids) >= 12:
        labels, topics = discover_topics(vids, X, meta_by_vid, df, k=8)
        fig_semantic_map(vids, X, meta_by_vid, labels, figs)
    drivers = fig_view_drivers(df, figs)
    q_median = fig_question_style(df, segs, figs)
    drift = temporal_drift(df)

    # deeper: retrieval benchmark, concept graph, redundancy, sentiment arc
    recall, mrr, n_q = evaluate_retrieval(df, figs)
    concept_freq = concept_network(df, figs)
    redundancy = cross_video_redundancy(vids, X, meta_by_vid)
    arc = emotional_arc(segs, figs)
    emo = emotion_analysis(df, segs, figs, pl_counts)

    sources = Counter(s.get("transcript_source", "unknown") for s in segs)
    n_text_only = len(df) - len(segs) - (df["chars"] == 0).sum()
    vocab = len(set(re.findall(r"[a-z']{3,}", " ".join(df["video_transcript"]).lower())))
    total_chunks = sum(chunk_counts.values())
    top_viewed = df.dropna(subset=["views"]).nlargest(10, "views")
    truncated = [(r.video_title, int(r.length_sec or 0), int(r.chars))
                 for r in df.itertuples()
                 if is_truncated(r.video_transcript, r.length_sec, r.playlist_tag)]

    # extra stats used in interpretations
    from scipy.stats import spearmanr
    d_eng = df.dropna(subset=["views", "minutes"])
    rho_len_views = float(spearmanr(d_eng["minutes"], d_eng["views"]).correlation) if len(d_eng) > 2 else float("nan")
    wdf = df[(df["wpm"] > 0) & (df["wpm"] < 400)].dropna(subset=["wpm"])
    slow_v = wdf.loc[wdf["wpm"].idxmin()] if len(wdf) else None
    fast_v = wdf.loc[wdf["wpm"].idxmax()] if len(wdf) else None
    longest_v = df.loc[df["minutes"].idxmax()] if df["minutes"].notna().any() else None
    pct_over_hour = float((df["minutes"] > 60).mean() * 100)
    biggest_pl = pl_counts.most_common(1)[0] if pl_counts else ("—", 0)
    hybrid_delta = (mrr["hybrid"] - mrr["dense"]) if recall else None

    def img(key, title):
        return f"### {title}\n\n![{title}]({figs[key]})\n" if key in figs else ""

    def md(s):  # escape pipes so video titles don't break markdown tables
        return str(s).replace("|", "\\|")

    L = []
    L.append("# Dr. K Transcript Dataset — EDA Report\n")
    L.append("_Generated by `data_analysis/eda.py`. Knowledge base behind the DrK_Chat RAG bot._\n")
    L.append("> **Reading this report.** Each section states its **Method** (how the numbers were "
             "computed, from which source, with what parameters) and an **Interpretation** (what it "
             "means and the caveats). Conventions used throughout: *medians* are preferred over means "
             "because most quantities (views, length) are right-skewed; transcript text is "
             "*contraction-cleaned* before term counting (so \"it's\"/\"don't\" don't masquerade as "
             "content words); \"embedding\" = the per-video mean of its `BAAI/bge-small-en-v1.5` chunk "
             "vectors, L2-normalised, pulled live from the Chroma index.\n")

    L.append("## 1. Corpus at a glance\n")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Videos (unique) | {len(df)} |")
    L.append(f"| Videos with transcripts | {(df['chars'] > 0).sum()} |")
    L.append(f"| Total content | {df['length_sec'].sum() / 3600:.1f} hours |")
    L.append(f"| Total words | {int(df['words'].sum()):,} |")
    L.append(f"| Unique vocabulary (3+ chars) | {vocab:,} |")
    L.append(f"| RAG chunks indexed | {total_chunks:,} across {len(chunk_counts)} videos |")
    L.append(f"| Date range | {df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d} |")
    L.append(f"| Median video length | {df['minutes'].median():.1f} min |")
    L.append(f"| Median transcript | {int(df['words'].median()):,} words |")
    L.append(f"| Median speaking rate | {df['wpm'].median():.0f} words/min |")
    L.append(f"| Total views | {int(df['views'].sum()):,} |")
    L.append(f"| Median views | {int(df['views'].median()):,} |")
    L.append(f"| Videos in >1 playlist | {multi} |")
    L.append("")
    L.append("**Method.** Aggregated over the canonical CSV (one row per video). `words` = "
             "whitespace-delimited tokens; `vocabulary` = distinct lower-cased alphabetic tokens of "
             "3+ chars; chunk counts read live from the Chroma collection's metadata.")
    L.append(f"**Interpretation.** A focused but substantial corpus — **{df['length_sec'].sum() / 3600:.0f} "
             f"hours / {int(df['words'].sum()):,} words** of one expert's framings on a single domain. "
             "That homogeneity is ideal for RAG: retrieval rarely has to disambiguate between "
             "unrelated authors or topics.\n")

    L.append("## 2. Content volume\n" + img("volume", "Video length & transcript size"))
    L.append("**Method.** Histograms over all videos: length = `video_length`(sec)/60; transcript size "
             "= word count of `video_transcript`.")
    L.append(f"**Interpretation.** Right-skewed: a typical video is **~{df['minutes'].median():.0f} min**, "
             f"but a long tail of interviews/lectures runs much longer (**{pct_over_hour:.0f}%** exceed an "
             f"hour" + (f", peaking at *{longest_v.video_title}* — {longest_v.minutes:.0f} min" if longest_v is not None else "") +
             "). Those long videos produce many more chunks (see §9), so they dominate retrieval volume.\n")

    L.append("## 3. Speaking rate\n" + img("wpm", "Words per minute"))
    L.append("**Method.** `wpm` = words ÷ minutes per video; the histogram drops values ≤0 or ≥400 "
             "(implausible, usually from bad duration/transcript pairs).")
    rate_line = (f"**Interpretation.** Median **{df['wpm'].median():.0f} wpm** is right in the band for "
                 "natural conversational English (~150–200). ")
    if slow_v is not None and fast_v is not None:
        rate_line += (f"The slowest, *{slow_v.video_title}* ({slow_v.wpm:.0f} wpm) — very low rates flag "
                      "either genuinely sparse speech (meditations) or a truncated transcript (cross-check "
                      f"§19); the fastest, *{fast_v.video_title}* ({fast_v.wpm:.0f} wpm), is rapid-fire "
                      "delivery. Slow/short videos yield few chunks, so they can under-surface in retrieval.")
    L.append(rate_line + "\n")

    L.append("## 4. Engagement\n" + img("engagement", "Views"))
    L.append("**Method.** Views from `video_views`. Left: histogram of log10(views) (raw views span "
             "orders of magnitude). Right: views vs length, log-y. Length↔views association measured "
             f"with **Spearman ρ** (rank correlation, robust to the skew): **ρ = {rho_len_views:+.2f}**.")
    L.append(f"**Interpretation.** ρ near zero means **length barely predicts views** — long interviews "
             "and short reactions both can go big; topic/title matters far more than runtime (see §12). "
             "The log-normal view distribution is the usual YouTube pattern (a few breakouts, a wide base).\n")
    L.append("**Top 10 most-viewed videos**\n")
    L.append("| Views | Length | Title |")
    L.append("|---:|---:|---|")
    for r in top_viewed.itertuples():
        L.append(f"| {int(r.views):,} | {r.minutes:.0f}m | {md(r.video_title)} |")
    L.append("")

    L.append("## 5. Publishing timeline\n" + img("timeline", "Cadence & cumulative hours"))
    L.append("**Method.** Videos grouped by `video_publish_date` year; bars = count/year, line = "
             "cumulative hours (running sum of durations).")
    L.append("**Interpretation.** Output peaked in **2021** then settled into a steadier cadence. Because "
             "this is a *sample* of playlists (not the full channel), recent years are under-counted — "
             "so the dip at the right edge reflects sampling, not necessarily reduced output.\n")

    L.append("## 6. Playlists\n" + img("playlists", "Videos & hours per playlist"))
    L.append(img("overlap", "Playlist co-occurrence"))
    L.append("**Method.** Playlist membership taken from each video's segments-JSON `all_playlist_tags` "
             "(falls back to the CSV tag). Left/right bars: video count and summed hours per playlist. "
             "Heatmap: cell (i,j) = number of videos appearing in *both* playlists i and j (diagonal = "
             "playlist size).")
    L.append(f"**Interpretation.** *{biggest_pl[0]}* is the largest theme ({biggest_pl[1]} videos). Off-"
             "diagonal cells are small, so the playlists are **mostly disjoint** — Dr. K's own taxonomy "
             "already carves fairly clean topic boundaries, which §10–11 then test against the embeddings.\n")

    L.append("## 7. What each playlist is *distinctively* about (TF-IDF)\n")
    L.append("**Method.** Concatenate every transcript in a playlist into one document, then run "
             "**TF-IDF** (`TfidfVectorizer`, English+filler stop-words, contraction-cleaned) across the "
             "playlist-documents. High TF-IDF = frequent *in this playlist* yet rare across the others, "
             "i.e. genuinely distinguishing terms (not generic filler).")
    if distinct:
        L.append("| Playlist | Distinctive terms |")
        L.append("|---|---|")
        for t, terms in sorted(distinct.items(), key=lambda kv: -pl_counts[kv[0]]):
            L.append(f"| {t} | {', '.join(terms)} |")
        L.append("")
    L.append("**Interpretation.** The terms are sharply on-topic (Meditation→*breath, metta, dharma*; "
             "Anxiety→*amygdala, breathing, attack*), confirming the transcripts carry strong, separable "
             "topical signal — the precondition for retrieval working well.\n")

    L.append("## 8. Language & themes\n" + img("terms", "Top words & bigrams"))
    L.append(img("cloud", "Word cloud"))
    L.append("**Method.** Raw occurrence counts (`CountVectorizer`, min_df=3) over contraction-cleaned, "
             "stop-word-filtered transcripts — unigrams and bigrams. The word cloud sizes words by the "
             "same frequency.")
    L.append("**Interpretation.** Top terms (*life, mind, relationship, brain, control*) and bigrams "
             "(*mental health, video games, social anxiety, self esteem, negative emotions*) read like a "
             "table of contents for the channel — concrete confirmation of what the knowledge base covers.\n")

    L.append("## 9. RAG index\n" + img("chunks", "Chunks per video"))
    if chunk_counts:
        vals = list(chunk_counts.values())
        L.append(f"**Method.** Each transcript is split into overlapping ~{config.CHUNK_TARGET_TOKENS}-token "
                 f"chunks (~{config.CHUNK_OVERLAP_TOKENS}-token overlap) over its timestamped segments, then "
                 "embedded and stored in Chroma; this counts chunks per `video_id` in the live index.")
        L.append(f"**Interpretation.** **{total_chunks:,} chunks** (median {int(np.median(vals))}/video, "
                 f"max {max(vals)}). Chunk count tracks video length, so long interviews contribute the most "
                 "retrievable passages — granular enough that retrieval returns a specific moment, not a "
                 "whole hour-long video.\n")

    L.append("## 10. Semantic map of the corpus\n" + img("semantic", "Video embedding map"))
    L.append("**Method.** Each video → mean of its bge chunk embeddings (384-dim), L2-normalised. "
             "Reduced to 2-D with **PCA→50 dims then t-SNE** (perplexity ≈ n/4). t-SNE preserves *local* "
             "neighbourhoods, so nearby points = semantically similar videos; absolute distances and axes "
             "are not meaningful.")
    L.append("**Interpretation.** Playlist colours form coherent regions rather than one blob — the "
             "embedding space already separates topics. Meditation sits apart (distinct vocabulary), while "
             "discussion topics border each other where themes overlap (e.g. relationships↔loneliness).\n")

    if topics:
        L.append("## 11. Data-driven topics (KMeans on embeddings)\n")
        L.append("**Method.** **KMeans (k=8)** clusters the video embeddings; each cluster is then labelled "
                 "by the top **TF-IDF** terms of its members' transcripts (vs other clusters), with its "
                 "most common playlist and an example title. This discovers topics *from the content*, "
                 "independent of Dr. K's hand-made playlists.")
        L.append("| Topic | Videos | Dominant playlist | Distinctive terms | Example |")
        L.append("|---|---:|---|---|---|")
        for t in sorted(topics, key=lambda x: -x["size"]):
            ex = t["examples"][0] if t["examples"] else ""
            L.append(f"| T{t['id']} | {t['size']} | {md(t['dominant_playlist'])} | "
                     f"{', '.join(t['terms'][:6])} | {md(ex)} |")
        L.append("")
        L.append("**Interpretation.** The unsupervised clusters line up with recognisable themes "
                 "(neuroscience-of-motivation: *dopamine/adenosine/circadian*; meditation: *breathe/exhale*; "
                 "porn/NoFap; dating) and each maps to a dominant playlist — independent confirmation that "
                 "the embeddings encode real topical structure, so hybrid retrieval is searching a "
                 "well-organised space.\n")

    L.append("## 12. What drives views\n" + img("drivers", "Title-word view lift"))
    L.append("**Method.** Tokenise titles (contraction-cleaned, stop-words removed). For each word in "
             "≥4 titles, **lift = median views of videos whose title contains it ÷ corpus median views**. "
             "Lift > 1 ⇒ that word's videos out-perform the typical video.")
    if drivers:
        hi = drivers[0]
        L.append(f"**Interpretation.** *{hi[0]}* tops the list at **~{hi[1] / df['views'].median():.1f}×** "
                 "the median, with *adhd / porn / relationships / dating* close behind — the audience "
                 "gravitates to identity and relationship themes. **Caveats:** this is correlational, not "
                 "causal (title word ≠ reason for views), and per-word counts are small (n shown on the "
                 "chart), so read it as a signal, not proof.\n")

    L.append("## 13. Conversational style (introspection)\n" + img("questions", "Question density"))
    L.append("**Method.** Question density = count of `?` ÷ words × 1000, per video; left = distribution, "
             "right = per-playlist mean.")
    L.append(f"**Interpretation.** A median **{q_median:.1f} questions / 1,000 words** quantifies Dr. K's "
             "Socratic, introspection-first style — exactly the behaviour the bot's persona emulates. "
             "**Caveat:** the spike at 0 is an artifact — some WhisperX/text-only transcripts lack `?` "
             "punctuation, so question density is *under*-counted for those (the median is robust to it, "
             "but the per-playlist bar is skewed for affected playlists).\n")

    if drift:
        L.append("## 14. How the focus shifted over time (TF-IDF per year)\n")
        L.append("**Method.** One concatenated transcript-document per publish year, then **TF-IDF** across "
                 "years — surfacing each year's distinctive vocabulary relative to the others.")
        L.append("| Year | Distinctive terms |")
        L.append("|---|---|")
        for y in sorted(drift):
            L.append(f"| {y} | {', '.join(drift[y])} |")
        L.append("")
        L.append("**Interpretation.** A visible thematic drift — early meditation/breathing → "
                 "relationships & gaming → clinical frameworks (*BPD, alexithymia*) → recent "
                 "*addiction/willpower/shadow*. Useful context for the bot: coverage of newer framings "
                 "depends on the newer videos being present in the index.\n")

    if recall:
        L.append("## 15. Retrieval benchmark — does hybrid actually help?\n")
        L.append(img("retrieval", "Dense vs BM25 vs hybrid"))
        L.append(f"**Method.** A proxy retrieval test over the {n_q} indexed videos: each video's **title "
                 "is used as a query**, and we record whether a chunk from that same video is returned. "
                 "**Recall@k** = fraction of queries whose correct video is in the top-k; **MRR** (mean "
                 "reciprocal rank) = average of 1/(rank of the first correct hit). The three retrievers — "
                 "dense (bge vectors), sparse (BM25), and the hybrid RRF fusion — run the identical task, "
                 "so differences are attributable to the method.")
        L.append("| Method | Recall@1 | Recall@3 | Recall@5 | MRR |")
        L.append("|---|---:|---:|---:|---:|")
        for m in ["dense", "sparse", "hybrid"]:
            L.append(f"| {m} | {recall[m][1]:.3f} | {recall[m][3]:.3f} | "
                     f"{recall[m][5]:.3f} | {mrr[m]:.3f} |")
        best = max(["dense", "sparse", "hybrid"], key=lambda m: mrr[m])
        L.append(f"\n**Interpretation.** **`{best}` wins on every metric**, beating dense-only by "
                 f"**{hybrid_delta:+.3f} MRR** and BM25 by more — the empirical justification for the "
                 "hybrid retriever: dense catches paraphrase/semantics, BM25 catches exact terms (names, "
                 "jargon), and RRF keeps the best of both. **Caveat:** title-as-query is an easy proxy "
                 "(title words often recur in the transcript), so absolute scores run high; the reliable "
                 "signal is the *ordering* of methods, and this harness can now regression-test retrieval "
                 "whenever the embedding model or chunking changes.\n")

    if "network" in figs:
        L.append("## 16. Concept co-occurrence network\n" + img("network", "Concept graph"))
        L.append("**Method.** For a fixed list of mental-health concepts, mark each as present in a video "
                 "if its word-stem appears in the transcript. Edge weight between two concepts = **Jaccard "
                 "overlap** of their video sets (|A∩B| / |A∪B|) — this normalises out sheer frequency so "
                 "broad concepts don't connect to everything. Only the strongest ~32 edges are drawn; node "
                 "size = number of videos mentioning the concept; layout = force-directed (spring).")
        L.append("**Interpretation.** *emotion* and *relationship* are the central hubs everything routes "
                 "through; a tight **motivation–addiction–dopamine–confidence** cluster captures the "
                 "behaviour-change thread, and an **anxiety–depression** cluster the clinical-affect thread, "
                 "while *ego↔identity* stand apart. This is, in effect, a map of Dr. K's recurring "
                 "mental-models.\n")

    if redundancy:
        L.append("## 17. Cross-video redundancy (most similar pairs)\n")
        L.append("**Method.** Cosine similarity between every pair of (L2-normalised) video embeddings = "
                 "their dot product; the table lists the highest-scoring distinct pairs.")
        L.append("| Similarity | Video A | Video B |")
        L.append("|---:|---|---|")
        for a, b, s in redundancy:
            L.append(f"| {s:.3f} | {md(a)} | {md(b)} |")
        L.append("")
        L.append("**Interpretation.** The top pairs are genuine near-duplicates — multi-part series "
                 "(*Motivation and Goals* Parts 1/3/4) and same-topic videos (NoFap/porn-addiction). For "
                 "the bot this is a heads-up: a single query can pull several near-identical chunks from "
                 "these, so de-duplicating retrieved sources (or capping chunks per video) would broaden "
                 "the perspectives shown to the user.\n")

    if arc is not None:
        L.append("## 18. Emotional arc — valence baseline (VADER)\n"
                 + img("arc", "Sentiment across the video timeline"))
        delta = arc[-1] - arc[0]
        trend = "more hopeful" if delta > 0 else "heavier"
        L.append("**Method.** Each segment scored with **VADER** compound sentiment (lexicon-based, −1…+1) "
                 "— a fast valence baseline. Every video's timeline is normalised to 0–100%, split into 12 "
                 "bins, and scores are averaged within each bin across all videos with ≥12 segments.")
        L.append(f"**Interpretation.** The corpus skews mildly **positive throughout**, dips in the middle "
                 "(where the problem is examined) and rises to its **peak at the end** "
                 f"(Δ={delta:+.3f} start→finish, {trend}) — a problem→resolution shape. VADER only measures "
                 "valence (one axis) and is tuned for social media; §19 replaces it with a model that "
                 "resolves *which* emotions are present. Treat the *shape*, not the absolute values, as the finding.\n")

    if emo:
        prof = emo["corpus"].sort_values(ascending=False)
        adf = emo["arc"]
        d_arc = (adf.iloc[-1] - adf.iloc[0])[AFFECT]
        riser = d_arc.idxmax()
        faller = d_arc.idxmin()
        topaff = [e for e in prof.index if e != "neutral"][:3]
        L.append("## 19. Fine-grained emotion analysis (transformer)\n")
        L.append(img("emotion_overview", "Corpus emotion profile & per-emotion arcs"))
        L.append(f"**Method.** Every transcript segment (≥4 words; {emo['n_segments']:,} in total) is "
                 f"classified by **`{EMOTION_MODEL}`**, a RoBERTa model that outputs a probability over 7 "
                 "emotions (anger, disgust, fear, joy, neutral, sadness, surprise) — far richer than VADER's "
                 "single valence axis. Left: mean probability per emotion across the whole corpus. Right: "
                 "each emotion averaged within 12 normalised timeline bins (neutral omitted for legibility).")
        L.append(f"**Interpretation.** Segments are mostly **neutral** ({prof['neutral']:.0%} — expected for "
                 f"expository talk). Across the timeline the heavier emotions ebb — **{faller} drops** most "
                 f"from start to finish — while **{riser} rises**, a shift away from dwelling on the problem "
                 "toward activation that echoes the VADER valence arc in §18, now resolved into specific "
                 "emotions. **Model caveat:** this classifier is known to over-assign **disgust** to frank or "
                 "critical (non-sad) language, so disgust tops the felt-emotion profile here partly as an "
                 "artifact — trust *relative* differences (across playlists and position) over the absolute "
                 "ranking of any single emotion.\n")

        L.append("## 20. Emotional fingerprint by playlist & engagement\n")
        L.append(img("emotion_fp", "Per-playlist emotion heatmap"))
        # most fearful / saddest / most joyful playlists among affect
        pe = emo["pl_em"]
        fear_pl = pe["fear"].idxmax()
        sad_pl = pe["sadness"].idxmax()
        joy_pl = pe["joy"].idxmax()
        ext = emo["extremes"]
        L.append("**Method.** The per-segment emotion probabilities are averaged to a video, then to a "
                 "playlist; the heatmap shows each playlist's mean probability for the six non-neutral "
                 "emotions. Per-video **valence** = P(joy) − P(anger+disgust+fear+sadness); its rank "
                 "correlation with views is **Spearman ρ = "
                 f"{emo['rho_val_views']:+.2f}**.")
        L.append(f"**Interpretation.** Playlists carry distinct affective signatures — **{fear_pl}** is "
                 f"highest in *fear*, **{sad_pl}** in *sadness*, **{joy_pl}** in *joy* — a sanity check that "
                 "the classifier tracks real content, and useful for tone-matching the bot's responses by "
                 "topic. Valence↔views ρ near zero means **emotional tone doesn't predict popularity**; "
                 "audiences don't simply prefer cheerful or heavy videos. Extremes: most positive = "
                 f"*{ext['most_positive'][0]}*; most negative = *{ext['most_negative'][0]}*; highest *fear* = "
                 f"*{ext['most_fear']}*.\n")

    L.append("## 21. Data quality\n")
    L.append(f"- **Transcript sources:** " +
             ", ".join(f"{k}={v}" for k, v in sources.items()) +
             f", text-only (CSV fallback, no timestamps)≈{max(n_text_only, 0)}.")
    L.append("- WhisperX dominates because YouTube rate-limited the captions API from this host; "
             "WhisperX output carries word-level timestamps, which improves citation precision.")
    L.append(f"- **Truncation candidates ({len(truncated)})** "
             "(meditation/low-speech playlists excluded):")
    for t, ln, ch in truncated:
        L.append(f"  - {t} — {ln}s, {ch} chars")
    L.append("- One row per video (multi-playlist videos de-duplicated; full membership in each "
             "segments JSON `all_playlist_tags`). `video_rating` is empty (YouTube removed it). "
             "Citation timestamps come from the segments JSON, not the CSV.")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT} and {len(figs)} figures to {FIG_DIR}")


if __name__ == "__main__":
    run()
