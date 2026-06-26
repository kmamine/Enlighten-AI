"""Transcript acquisition: YouTube captions first, WhisperX as fallback.

A transcript is represented as a list of segments: ``{"start": float, "end":
float, "text": str}``. Captions (fast, no GPU) are tried first; the WhisperX
fallback (GPU) is invoked by the orchestrator only when captions are missing or
an existing transcript looks truncated.
"""
from __future__ import annotations

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)

import config

Segment = dict  # {"start": float, "end": float, "text": str}

# Reference speaking rate for English speech (characters per second).
_REF_CPS = 14.0


def _raw_to_segments(raw: list[dict]) -> list[Segment]:
    """Normalize youtube-transcript-api {text, start, duration} to start/end/text."""
    segs: list[Segment] = []
    for item in raw:
        start = float(item.get("start", 0.0))
        dur = float(item.get("duration", 0.0))
        text = (item.get("text") or "").replace("\n", " ").strip()
        if text:
            segs.append({"start": start, "end": start + dur, "text": text})
    return segs


def fetch_captions(video_id: str) -> list[Segment] | None:
    """Fetch the best English caption track, or None if unavailable.

    Preference: manually-created EN > auto-generated EN > any track translated to EN.
    """
    api = YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
            CouldNotRetrieveTranscript):
        return None
    except Exception:
        return None

    candidates = []
    try:
        # 1) manual EN, 2) generated EN
        try:
            candidates.append(transcript_list.find_manually_created_transcript(["en", "en-US", "en-GB"]))
        except Exception:
            pass
        try:
            candidates.append(transcript_list.find_generated_transcript(["en", "en-US", "en-GB"]))
        except Exception:
            pass
        # 3) any translatable track -> EN
        if not candidates:
            for t in transcript_list:
                if t.is_translatable:
                    try:
                        candidates.append(t.translate("en"))
                        break
                    except Exception:
                        continue
    except Exception:
        return None

    for transcript in candidates:
        try:
            raw = transcript.fetch().to_raw_data()
        except Exception:
            continue
        segs = _raw_to_segments(raw)
        if segs:
            return segs
    return None


def segments_to_text(segments: list[Segment]) -> str:
    """Join segment texts into a single transcript string."""
    return " ".join(s["text"] for s in segments if s.get("text")).strip()


def is_truncated(transcript: str, video_length: float | str, playlist_tag: str) -> bool:
    """Heuristic: does an existing transcript look cut off / incomplete?

    Combines independent signals and requires >=2 to flag (avoids false positives
    on legitimately sparse speech such as meditations). For playlists known to be
    sparse, the rate-based signals are disabled, so they are never re-transcribed.
    """
    text = (transcript or "").strip()
    if not text:
        return True  # empty transcript is definitely incomplete
    try:
        length = float(video_length)
    except (TypeError, ValueError):
        return False
    if length <= 0:
        return False

    sparse = playlist_tag in config.SPARSE_SPEECH_PLAYLISTS
    signals = 0

    if not sparse:
        cps = len(text) / length
        if cps < 9:
            signals += 1
        expected_chars = _REF_CPS * length
        if len(text) < 0.6 * expected_chars:
            signals += 1

    # Mid-sentence cutoff: transcript doesn't end on terminal punctuation.
    if text[-1] not in ".?!\"')]":
        signals += 1

    return signals >= 2
