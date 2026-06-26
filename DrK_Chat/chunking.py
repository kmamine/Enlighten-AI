"""Chunk timestamped transcript segments into overlapping, citable passages.

Chunks are formed over the *segment* list (not raw characters) so each chunk
keeps real start/end times -> citation links jump to the right moment in the
video. Sizing is token-aware via an injected `count_tokens` callable (the
embedding model's tokenizer), so chunks respect the embedder's context window.
"""
from __future__ import annotations

from typing import Callable

import config

Segment = dict   # {"start": float, "end": float, "text": str}
Chunk = dict     # {"text", "start_time", "end_time", "n_tokens"}


def _approx_tokens(text: str) -> int:
    """Fallback token estimate (~0.75 words/token) when no tokenizer is given."""
    return max(1, int(len(text.split()) / 0.75))


def chunk_segments(
    segments: list[Segment],
    count_tokens: Callable[[str], int] | None = None,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """Greedily pack consecutive segments up to `target_tokens`, with overlap.

    Each emitted chunk starts a little before the previous one ended (by roughly
    `overlap_tokens`) so context isn't lost at chunk boundaries.
    """
    count = count_tokens or _approx_tokens
    target = target_tokens or config.CHUNK_TARGET_TOKENS
    overlap = overlap_tokens or config.CHUNK_OVERLAP_TOKENS

    segments = [s for s in segments if (s.get("text") or "").strip()]
    n = len(segments)
    if n == 0:
        return []

    chunks: list[Chunk] = []
    i = 0
    while i < n:
        texts: list[str] = []
        tokens = 0
        start_time = float(segments[i].get("start", 0.0))
        end_time = start_time
        j = i
        while j < n:
            seg_tokens = count(segments[j]["text"])
            # always include at least one segment even if it alone exceeds target
            if texts and tokens + seg_tokens > target:
                break
            texts.append(segments[j]["text"])
            tokens += seg_tokens
            end_time = float(segments[j].get("end", end_time))
            j += 1

        chunks.append({
            "text": " ".join(texts).strip(),
            "start_time": start_time,
            "end_time": end_time,
            "n_tokens": tokens,
        })

        if j >= n:
            break

        # back up to create overlap, but always make forward progress
        back = j
        ov = 0
        while back > i + 1 and ov < overlap:
            back -= 1
            ov += count(segments[back]["text"])
        i = max(back, i + 1)

    return chunks
