"""Orchestrator: build/refresh the Dr. K transcript dataset.

Pipeline per video:  yt-dlp metadata  ->  captions (youtube-transcript-api)
                     ->  WhisperX fallback (if missing/truncated)  ->  CSV + segments JSON

Design notes:
  * The dataset is keyed by **video_id** (one row per video). The legacy CSV had
    duplicate rows for videos appearing in multiple playlists; those are merged,
    with the primary playlist = the first one the video is found in and every
    playlist recorded in the per-video segments JSON (`all_playlist_tags`).
  * Crash-safe: the CSV is written via a temp file + atomic rename every few
    videos, so an interrupted run never corrupts/loses the dataset.
  * Idempotent: re-running skips videos that already have a transcript + segments
    unless they look truncated (`--refresh-truncated`) or metadata is being
    refreshed.

CLI examples:
  python -m Scrapper.build_dataset --playlist "Anxiety" --limit 3   # smoke test
  python -m Scrapper.build_dataset --only-new                       # fast incremental
  python -m Scrapper.build_dataset --refresh-truncated              # rebuild cut-off transcripts
  python -m Scrapper.build_dataset                                  # full refresh + new videos
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import OrderedDict

import pandas as pd
from tqdm import tqdm

import config
from . import playlists as pl
from . import transcripts as tr
from . import whisperx_backend as wx

SAVE_EVERY = 5


# --- CSV / segments IO -----------------------------------------------------
def load_existing() -> tuple[OrderedDict, dict[str, list[str]]]:
    """Load the canonical CSV deduped by video_id.

    Returns (rows_by_id, legacy_tags_by_id). Seeds from a legacy copy if the
    canonical CSV is absent.
    """
    path = config.CSV_PATH
    if not path.exists():
        for legacy in config.LEGACY_CSV_PATHS:
            if legacy.exists():
                config.ensure_dirs()
                shutil.copyfile(legacy, path)
                break
    rows_by_id: OrderedDict = OrderedDict()
    tags_by_id: dict[str, list[str]] = {}
    if not path.exists():
        return rows_by_id, tags_by_id

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for _, r in df.iterrows():
        row = {c: r.get(c, "") for c in config.CSV_COLUMNS}
        vid = pl.extract_video_id(row.get("video_url", ""))
        if not vid:
            continue
        tag = row.get("playlist_tag", "")
        tags_by_id.setdefault(vid, [])
        if tag and tag not in tags_by_id[vid]:
            tags_by_id[vid].append(tag)
        if vid not in rows_by_id:  # keep first occurrence as canonical row
            rows_by_id[vid] = row
    return rows_by_id, tags_by_id


def save_csv(rows_by_id: OrderedDict) -> None:
    """Atomically write the canonical CSV from the in-memory rows."""
    config.ensure_dirs()
    df = pd.DataFrame(list(rows_by_id.values()), columns=config.CSV_COLUMNS)
    tmp = config.CSV_PATH.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, config.CSV_PATH)


def reconcile_legacy_copies() -> None:
    """Copy the canonical CSV over the two legacy copies to retire the drift."""
    for legacy in config.LEGACY_CSV_PATHS:
        try:
            legacy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config.CSV_PATH, legacy)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] could not reconcile {legacy}: {exc}")


def write_segments_json(video_id: str, row: dict, all_tags: list[str],
                        segments: list[tr.Segment], source: str) -> None:
    config.ensure_dirs()
    payload = {
        "video_id": video_id,
        "video_url": row.get("video_url", ""),
        "video_title": row.get("video_title", ""),
        "playlist_tag": row.get("playlist_tag", ""),
        "all_playlist_tags": all_tags,
        "transcript_source": source,
        "segments": segments,
    }
    (config.SEGMENTS_DIR / f"{video_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False)
    )


# --- transcript acquisition -------------------------------------------------
def acquire_transcript(video_id: str, row: dict, no_whisper: bool):
    """Return (segments, text, source) or (None, '', None).

    Captions first; WhisperX fallback when captions are missing or look truncated.
    """
    tag = row.get("playlist_tag", "")
    length = row.get("video_length", "")

    segs = tr.fetch_captions(video_id)
    source = "captions" if segs else None
    text = tr.segments_to_text(segs) if segs else ""

    if (not segs) or tr.is_truncated(text, length, tag):
        if not no_whisper:
            wsegs = wx.transcribe(video_id)
            if wsegs:
                segs, text, source = wsegs, tr.segments_to_text(wsegs), "whisperx"
    return segs, text, source


# --- main run ---------------------------------------------------------------
def run(args) -> None:
    config.ensure_dirs()
    rows_by_id, legacy_tags = load_existing()
    print(f"Loaded {len(rows_by_id)} existing videos.")

    # Enumerate playlists -> ordered membership.
    wanted = pl.load_playlists()
    if args.playlist:
        wanted = [(t, u) for (t, u) in wanted if t == args.playlist]
        if not wanted:
            raise SystemExit(f"No playlist named {args.playlist!r} in Data.json")

    membership: "OrderedDict[str, dict]" = OrderedDict()
    print("Enumerating playlists...")
    for tag, url in wanted:
        try:
            ids = list(pl.iter_playlist_entries(url))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] failed to enumerate {tag!r}: {exc}")
            continue
        print(f"  {tag}: {len(ids)} videos")
        for vid in ids:
            if vid not in membership:
                membership[vid] = {"primary": (tag, url), "all_tags": [tag]}
            elif tag not in membership[vid]["all_tags"]:
                membership[vid]["all_tags"].append(tag)

    order = list(membership.keys())
    if args.limit:
        order = order[: args.limit]

    failures: list[tuple[str, str]] = []
    processed = 0

    for i, vid in enumerate(tqdm(order, desc="videos")):
        primary_tag, primary_url = membership[vid]["primary"]
        all_tags = membership[vid]["all_tags"]
        # merge any legacy playlist tags we already knew about
        for t in legacy_tags.get(vid, []):
            if t not in all_tags:
                all_tags.append(t)

        existing = rows_by_id.get(vid)
        seg_exists = (config.SEGMENTS_DIR / f"{vid}.json").exists()

        if existing is not None and args.only_new:
            continue

        # --- metadata ---
        if existing is None:
            info = pl.fetch_video_metadata(vid)
            if not info:
                failures.append((vid, "metadata fetch failed"))
                continue
            row = pl.entry_to_row(info, primary_tag, primary_url)
        else:
            row = dict(existing)
            row["playlist_tag"], row["playlist_url"] = primary_tag, primary_url
            # backfill legacy empties or force refresh
            if args.refresh_metadata or not row.get("video_description"):
                info = pl.fetch_video_metadata(vid)
                if info:
                    kept = row.get("video_transcript", "")
                    row = pl.entry_to_row(info, primary_tag, primary_url)
                    row["video_transcript"] = kept

        # --- transcript ---
        has_text = bool(row.get("video_transcript", "").strip())
        truncated = tr.is_truncated(row.get("video_transcript", ""),
                                    row.get("video_length", ""), primary_tag)
        need_transcript = (
            not has_text
            or not seg_exists
            or (args.refresh_truncated and truncated)
        )
        if need_transcript:
            segs, text, source = acquire_transcript(vid, row, args.no_whisper)
            if segs:
                row["video_transcript"] = text
                write_segments_json(vid, row, all_tags, segs, source)
            elif not has_text:
                failures.append((vid, "no transcript (captions+whisper failed)"))

        rows_by_id[vid] = row
        processed += 1
        if processed % SAVE_EVERY == 0:
            save_csv(rows_by_id)

    save_csv(rows_by_id)
    reconcile_legacy_copies()

    print(f"\nDone. {processed} videos processed, {len(rows_by_id)} total in dataset.")
    if failures:
        print(f"{len(failures)} failures:")
        for vid, why in failures:
            print(f"  {vid}: {why}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build/refresh the Dr. K transcript dataset.")
    p.add_argument("--limit", type=int, default=0, help="process at most N videos (testing)")
    p.add_argument("--playlist", type=str, default="", help="restrict to one playlist tag")
    p.add_argument("--only-new", action="store_true", help="skip videos already in the CSV")
    p.add_argument("--refresh-truncated", action="store_true",
                   help="re-transcribe existing videos whose transcript looks cut off")
    p.add_argument("--refresh-metadata", action="store_true",
                   help="force re-fetch metadata for existing videos (else only backfills empties)")
    p.add_argument("--no-whisper", action="store_true",
                   help="disable the WhisperX GPU fallback (captions only)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
