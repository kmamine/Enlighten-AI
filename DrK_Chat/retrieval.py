"""Hybrid retrieval: dense (Chroma) + sparse (BM25), fused with RRF.

Chroma is the single source of truth for chunks; the BM25 index is *derived*
from the collection's documents at construction time (so there's no separate
index to keep in sync). Dense and sparse candidate lists are merged with
Reciprocal Rank Fusion, which combines rankings without having to reconcile
cosine-distance and BM25 score scales.
"""
from __future__ import annotations

import re

import config
from .embeddings import Embedder
from .ingest import get_collection

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class Retriever:
    def __init__(self, embedder: Embedder | None = None):
        from rank_bm25 import BM25Okapi

        self.embedder = embedder or Embedder()
        self.collection = get_collection(create=True)

        data = self.collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = data.get("ids", []) or []
        docs: list[str] = data.get("documents", []) or []
        metas: list[dict] = data.get("metadatas", []) or []
        self.by_id = {
            i: {"text": d, "metadata": m}
            for i, d, m in zip(self.ids, docs, metas)
        }
        self._bm25 = BM25Okapi([_tokenize(d) for d in docs]) if docs else None

    @property
    def size(self) -> int:
        return len(self.ids)

    def _dense_ranking(self, query: str, pool: int) -> list[str]:
        if not self.ids:
            return []
        qvec = self.embedder.embed_query(query)
        res = self.collection.query(
            query_embeddings=[qvec], n_results=min(pool, len(self.ids))
        )
        return (res.get("ids") or [[]])[0]

    def _sparse_ranking(self, query: str, pool: int) -> list[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in ranked[:pool]]

    def hybrid_search(self, query: str, k: int | None = None,
                      pool: int | None = None, rrf_k: int | None = None) -> list[dict]:
        k = k or config.RETRIEVE_K
        pool = pool or config.RETRIEVE_POOL
        rrf_k = rrf_k or config.RRF_K

        dense = self._dense_ranking(query, pool)
        sparse = self._sparse_ranking(query, pool)

        scores: dict[str, float] = {}
        for ranking in (dense, sparse):
            for rank, cid in enumerate(ranking, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

        top = sorted(scores, key=lambda c: scores[c], reverse=True)[:k]
        out = []
        for cid in top:
            rec = self.by_id.get(cid)
            if rec:
                out.append({"id": cid, "score": scores[cid], **rec})
        return out
