"""Transcript transforms evaluated for their effect on retrieval quality.

Two transforms, each A/B-tested in `data_analysis/retrieval_experiment.py`:

* **clean** — deterministic, lossless-ish denoising applied *per segment* so the
  timestamps that power citations are preserved. Targets the artifacts our EDA
  actually found (e.g. WhisperX immediate-repetition stutters like "it's it's").
* **surrogate** — an LLM-generated set of search queries the chunk answers
  (doc2query-style), used to *augment* the embedded/lexical representation while
  the original verbatim chunk remains what is stored and cited.
"""
from __future__ import annotations

import re

# --- deterministic cleaning -------------------------------------------------
# Standalone speech fillers (whole-word, case-insensitive). Deliberately
# conservative: we do NOT strip "like"/"you know" (often content here).
_FILLER = re.compile(r"\b(?:um+|uh+|uhm+|erm+|mm+|mhm+|hmm+)\b", re.IGNORECASE)
# Immediate duplicate word ("it's it's", "the the the") -> single. Backrefs are
# case-insensitive under IGNORECASE; the kept token keeps the first occurrence.
_DUP_WORD = re.compile(r"\b([\w']+)(?:\s+\1\b)+", re.IGNORECASE)
# Immediate duplicate bigram ("i think i think") -> single.
_DUP_BIGRAM = re.compile(r"\b([\w']+\s+[\w']+)(?:\s+\1\b)+", re.IGNORECASE)
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Collapse stutter repetitions, drop fillers, normalize whitespace."""
    if not text:
        return ""
    t = _FILLER.sub(" ", text)
    t = _DUP_BIGRAM.sub(r"\1", t)
    t = _DUP_WORD.sub(r"\1", t)
    return _WS.sub(" ", t).strip()


def clean_segments(segments: list[dict]) -> list[dict]:
    """Clean each segment's text in place-style (new list); drop emptied segments.

    Segment start/end times are preserved untouched, so citation timestamps are
    unaffected by cleaning.
    """
    out = []
    for s in segments:
        txt = clean_text(s.get("text", ""))
        if txt:
            out.append({**s, "text": txt})
    return out


# --- multi-representation surrogate (doc2query-style) -----------------------
SURROGATE_SYSTEM = (
    "You generate search queries for a retrieval system over mental-health talk "
    "transcripts. Given a passage, output 3 short, varied queries (natural questions "
    "or keyword phrases) that a person might type to find this passage. Output ONLY "
    "the queries, one per line, no numbering, no preamble."
)


def surrogate_messages(chunk_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SURROGATE_SYSTEM},
        {"role": "user", "content": f"Passage:\n\"\"\"\n{chunk_text[:2000]}\n\"\"\"\n\nQueries:"},
    ]


def generate_surrogates(chunk_texts: list[str], client, model: str,
                        max_workers: int = 16) -> list[str]:
    """Generate a query-surrogate per chunk concurrently. Falls back to '' on error
    (caller then keeps the original text for that chunk)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(text: str) -> str:
        try:
            r = client.chat.completions.create(
                model=model, messages=surrogate_messages(text),
                temperature=0.3, max_tokens=80,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception:
            return ""

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(one, chunk_texts))
