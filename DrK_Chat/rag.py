"""RAG query: hybrid-retrieve -> grounded prompt -> vLLM Gemma -> answer + sources.

Exposes both a streaming path (for the Streamlit UI) and a one-shot `answer()`
(for scripts/tests). The Retriever and OpenAI client are cached at module level
so repeated calls don't reload the embedding model or reopen connections.
"""
from __future__ import annotations

import config
from . import prompts
from . import guardrails
from .retrieval import Retriever

_retriever: Retriever | None = None
_reranker = None
_client = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_reranker():
    global _reranker
    if _reranker is None:
        from .rerank import Reranker
        _reranker = Reranker()
    return _reranker


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=config.VLLM_BASE_URL, api_key=config.VLLM_API_KEY)
    return _client


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    """One entry per video, keeping the highest-ranked (earliest) occurrence."""
    seen, out = set(), []
    for s in sources:
        vid = s.get("video_id")
        if vid in seen:
            continue
        seen.add(vid)
        out.append(s)
    return out


def build(query: str, history: list[dict] | None = None, k: int | None = None,
          retriever: Retriever | None = None):
    """Retrieve + assemble messages. Returns (messages, deduped_sources, chunks)."""
    retriever = retriever or get_retriever()
    k = k or config.RETRIEVE_K
    if config.RERANK_ENABLED:
        # retrieve a wider pool, then cross-encoder rerank down to k (off by default;
        # see data_analysis/rerank_experiment.md — no significant gain on our eval)
        cands = retriever.hybrid_search(query, k=config.RERANK_POOL, pool=config.RERANK_POOL)
        chunks = get_reranker().rerank(query, cands, top_k=k)
    else:
        chunks = retriever.hybrid_search(query, k=k)
    messages, sources = prompts.build_messages(history or [], query, chunks)
    return messages, _dedupe_sources(sources), chunks


def stream_completion(messages: list[dict]):
    """Yield response text deltas from the vLLM endpoint."""
    stream = get_client().chat.completions.create(
        model=config.VLLM_MODEL,
        messages=messages,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def answer(query: str, history: list[dict] | None = None, k: int | None = None,
           retriever: Retriever | None = None) -> dict:
    """One-shot (non-streaming) answer. Returns {answer, sources, chunks}."""
    messages, sources, chunks = build(query, history, k, retriever)
    resp = get_client().chat.completions.create(
        model=config.VLLM_MODEL,
        messages=messages,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": sources,
        "chunks": chunks,
    }


def safe_answer(query: str, history: list[dict] | None = None, k: int | None = None,
                retriever: Retriever | None = None) -> dict:
    """Guardrailed one-shot answer: input screen short-circuits to a vetted safety
    response on crisis/harm/method; otherwise normal RAG + output screen."""
    g = guardrails.screen_input(query, client=get_client())
    if g["action"] != "allow":
        return {"answer": g["response"], "sources": [], "chunks": [], "guard": g}
    out = answer(query, history, k, retriever)
    out["answer"], out["guard_output_flagged"] = guardrails.screen_output(out["answer"])
    out["guard"] = g
    return out
