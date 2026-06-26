"""Cross-encoder reranker: re-scores retrieved candidates for query relevance.

A bi-encoder (the bge embedder) scores query and passage independently; a
cross-encoder reads the (query, passage) pair jointly, which is more accurate but
too slow to run over the whole corpus. So it's used as a *second stage*: retrieve
a pool with hybrid search, then rerank that pool here. Whether it actually helps
is decided empirically in `data_analysis/rerank_experiment.py`.
"""
from __future__ import annotations

import config


class Reranker:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or config.RERANK_MODEL
        self.model = CrossEncoder(self.model_name, device=device or config.EMBED_DEVICE,
                                  max_length=512)

    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        """Reorder candidates (each a dict with a 'text' key) by cross-encoder
        relevance to the query, descending. Adds 'rerank_score' to each."""
        if not candidates:
            return candidates
        scores = self.model.predict([(query, c.get("text", "")) for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        out = [{**candidates[i], "rerank_score": float(scores[i])} for i in order]
        return out[:top_k] if top_k else out
