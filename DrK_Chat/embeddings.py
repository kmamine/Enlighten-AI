"""Sentence-transformers embedding wrapper (GPU), shared by ingest and retrieval.

Loading the model is expensive, so callers should construct one `Embedder` and
reuse it. Embeddings are L2-normalized so cosine similarity == dot product, which
matches the Chroma collection's cosine space.
"""
from __future__ import annotations

import config


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.EMBED_MODEL
        self.device = device or config.EMBED_DEVICE
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed passages (documents). bge passages need no instruction prefix."""
        vecs = self.model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with the bge query instruction prefix."""
        vec = self.model.encode(
            [config.EMBED_QUERY_PREFIX + text], normalize_embeddings=True
        )
        return vec[0].tolist()

    def count_tokens(self, text: str) -> int:
        """Token count using the model's own tokenizer (for chunk sizing)."""
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))
