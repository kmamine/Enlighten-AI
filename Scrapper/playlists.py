"""yt-dlp based playlist enumeration and metadata extraction.

Replaces the abandoned `pytube` pipeline. Two-stage by design:
  1. `iter_playlist_entries` does a cheap *flat* extraction (ids only, one network
     call per playlist) so we can diff against what's already scraped.
  2. `fetch_video_metadata` does a full extraction only for the ids we actually
     need, mapping the result onto the original 14-column CSV schema.
"""
from __future__ import annotations

import json
import re
from typing import Iterator

from yt_dlp import YoutubeDL

import config

_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/)([\w-]{11})")

# Shared yt-dlp options: quiet, no download, resilient to single-video errors.
_BASE_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "ignoreerrors": True,
    "retries": 5,
    "sleep_interval_requests": 1,
}


def extract_video_id(url: str) -> str | None:
    """Pull the 11-char YouTube id from any watch/short/youtu.be URL."""
    if not url:
        return None
    m = _VIDEO_ID_RE.search(url)
    return m.group(1) if m else None


def canonical_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def load_playlists() -> list[tuple[str, str]]:
    """Read Data.json -> list of (playlist_tag, playlist_url)."""
    data = json.loads(config.PLAYLISTS_JSON.read_text())
    out: list[tuple[str, str]] = []
    for entry in data["playlists"]:
        for tag, url in entry.items():
            out.append((tag, url))
    return out


def iter_playlist_entries(playlist_url: str) -> Iterator[str]:
    """Yield video ids in a playlist using flat (cheap) extraction."""
    opts = {**_BASE_OPTS, "extract_flat": "in_playlist"}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    if not info:
        return
    for entry in info.get("entries") or []:
        if not entry:
            continue
        vid = entry.get("id") or extract_video_id(entry.get("url", ""))
        if vid:
            yield vid


def fetch_video_metadata(video_id: str) -> dict | None:
    """Full metadata extraction for one video. Returns yt-dlp info dict or None."""
    with YoutubeDL(_BASE_OPTS) as ydl:
        return ydl.extract_info(canonical_watch_url(video_id), download=False)


def _fmt_float_str(value) -> str:
    """Match the original CSV's float-string style (e.g. 1469 -> '1469.0')."""
    if value is None:
        return ""
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return ""


def _fmt_publish_date(upload_date: str | None) -> str:
    """yt-dlp 'YYYYMMDD' -> original '%Y-%m-%d 00:00:00' style."""
    if not upload_date or len(upload_date) != 8:
        return ""
    return f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]} 00:00:00"


def entry_to_row(info: dict, playlist_tag: str, playlist_url: str) -> dict:
    """Map a yt-dlp info dict onto the 14-column CSV schema.

    Backfills `video_keywords` and `video_description` (empty in the legacy data).
    `video_rating` stays empty (YouTube removed public dislike/rating data).
    `video_transcript` is filled later by the transcription stage.
    """
    vid = info.get("id", "")
    return {
        "playlist_tag": playlist_tag,
        "playlist_url": playlist_url,
        "video_url": canonical_watch_url(vid),
        "channel_id": info.get("channel_id", "") or "",
        "channel_url": info.get("channel_url") or info.get("uploader_url", "") or "",
        "video_title": info.get("title", "") or "",
        "video_length": _fmt_float_str(info.get("duration")),
        "video_publish_date": _fmt_publish_date(info.get("upload_date")),
        "video_rating": "",
        "video_views": _fmt_float_str(info.get("view_count")),
        "video_author": info.get("uploader") or info.get("channel", "") or "",
        "video_keywords": ", ".join(info.get("tags") or []),
        "video_description": info.get("description", "") or "",
        "video_transcript": "",
    }
