"""Ingest timestamped segments into a persistent Chroma collection.

Primary source of truth for ingestion is `data/segments/*.json` (each has the
metadata and timestamps needed for citations). Per video we delete any existing
chunks then upsert fresh ones, so re-runs are idempotent even when a transcript
was re-chunked. Chunk id = ``<video_id>:<chunk_index>``.

As a fallback, any video in the canonical CSV that has a transcript but no
segments JSON (e.g. members-only/age-gated videos whose audio couldn't be
re-fetched, leaving only the legacy plain-text transcript) is still ingested
from its CSV text — chunked without timestamps so it remains searchable; its
citations link to the video start.
"""
from __future__ import annotations

import argparse
import glob
import json
import re

import pandas as pd
from tqdm import tqdm

import config
from Scrapper.playlists import extract_video_id
from .chunking import chunk_segments
from .embeddings import Embedder
from .transforms import clean_segments, clean_text

_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")


def get_collection(create: bool = True):
    """Return the persistent Chroma collection (records the embed model in metadata)."""
    import chromadb

    config.ensure_dirs()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if create:
        return client.get_or_create_collection(
            config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine", "embed_model": config.EMBED_MODEL},
        )
    return client.get_collection(config.CHROMA_COLLECTION)


def _timestamp_url(video_id: str, start: float) -> str:
    return f"https://youtu.be/{video_id}?t={int(start)}s"


def _chunk_to_record(video_id: str, meta: dict, idx: int, chunk: dict):
    cid = f"{video_id}:{idx}"
    metadata = {
        "video_id": video_id,
        "video_title": meta.get("video_title", ""),
        "video_url": meta.get("video_url", ""),
        "playlist_tag": meta.get("playlist_tag", ""),
        # Chroma metadata must be scalar -> join the tag list to a string.
        "all_playlist_tags": ", ".join(meta.get("all_playlist_tags", []) or []),
        "start_time": float(chunk["start_time"]),
        "end_time": float(chunk["end_time"]),
        "chunk_index": idx,
        "timestamp_url": _timestamp_url(video_id, chunk["start_time"]),
    }
    return cid, chunk["text"], metadata


def ingest(rebuild: bool = False) -> None:
    import chromadb

    embedder = Embedder()
    config.ensure_dirs()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if rebuild:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
            print("Dropped existing collection (rebuild).")
        except Exception:
            pass
    coll = client.get_or_create_collection(
        config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine", "embed_model": config.EMBED_MODEL},
    )

    files = sorted(glob.glob(str(config.SEGMENTS_DIR / "*.json")))
    if not files:
        raise SystemExit(f"No segments found in {config.SEGMENTS_DIR}. Run the scraper first.")

    total_chunks = 0
    ingested_vids: set[str] = set()
    for path in tqdm(files, desc="ingest"):
        meta = json.loads(open(path, encoding="utf-8").read())
        vid = meta["video_id"]
        segments = meta.get("segments", [])
        if config.CLEAN_TRANSCRIPTS:
            segments = clean_segments(segments)
        chunks = chunk_segments(segments, count_tokens=embedder.count_tokens)
        if not chunks:
            continue
        total_chunks += _upsert_video(coll, embedder, vid, meta, chunks)
        ingested_vids.add(vid)

    fb_videos, fb_chunks = _ingest_csv_fallback(coll, embedder, ingested_vids)
    total_chunks += fb_chunks

    print(f"\nIngested {total_chunks} chunks from {len(ingested_vids) + fb_videos} videos "
          f"({len(ingested_vids)} with timestamps, {fb_videos} text-only). "
          f"Collection now holds {coll.count()} chunks.")


def _upsert_video(coll, embedder, vid: str, meta: dict, chunks: list[dict]) -> int:
    ids, docs, metas = [], [], []
    for idx, ch in enumerate(chunks):
        cid, text, m = _chunk_to_record(vid, meta, idx, ch)
        ids.append(cid)
        docs.append(text)
        metas.append(m)
    embeddings = embedder.embed_passages(docs)
    try:  # clean re-ingest: drop this video's old chunks first
        coll.delete(where={"video_id": vid})
    except Exception:
        pass
    coll.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    return len(ids)


def _ingest_csv_fallback(coll, embedder, ingested_vids: set[str]) -> tuple[int, int]:
    """Ingest CSV videos that have transcript text but no segments JSON.

    These are chunked from plain text (no timestamps) so they stay searchable.
    """
    if not config.CSV_PATH.exists():
        return 0, 0
    df = pd.read_csv(config.CSV_PATH, dtype=str, keep_default_na=False)
    n_videos = n_chunks = 0
    for r in df.itertuples():
        text = (getattr(r, "video_transcript", "") or "").strip()
        vid = extract_video_id(getattr(r, "video_url", "") or "")
        if not vid or not text or vid in ingested_vids:
            continue
        if config.CLEAN_TRANSCRIPTS:
            text = clean_text(text)
        sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
        pseudo = [{"start": 0.0, "end": 0.0, "text": s} for s in sentences] or \
                 [{"start": 0.0, "end": 0.0, "text": text}]
        chunks = chunk_segments(pseudo, count_tokens=embedder.count_tokens)
        if not chunks:
            continue
        meta = {
            "video_id": vid,
            "video_title": getattr(r, "video_title", ""),
            "video_url": getattr(r, "video_url", ""),
            "playlist_tag": getattr(r, "playlist_tag", ""),
            "all_playlist_tags": [getattr(r, "playlist_tag", "")],
        }
        n_chunks += _upsert_video(coll, embedder, vid, meta, chunks)
        n_videos += 1
    if n_videos:
        print(f"  [fallback] ingested {n_videos} text-only videos (no timestamps).")
    return n_videos, n_chunks


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest transcript segments into Chroma.")
    p.add_argument("--rebuild", action="store_true",
                   help="drop and recreate the collection before ingesting")
    p.add_argument("--incremental", action="store_true",
                   help="(default behavior) idempotent per-video upsert")
    args = p.parse_args()
    ingest(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
