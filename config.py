"""Central configuration for Enlighten AI / DrK_Chat.

All paths are absolute and rooted at this file's directory so the modules work
regardless of the current working directory. Runtime secrets/endpoints are read
from a `.env` file (see `.env` in the repo root) with sensible fallbacks.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
SEGMENTS_DIR = DATA_DIR / "segments"
CHROMA_DIR = DATA_DIR / "chroma"

# Canonical knowledge-base CSV (single source of truth).
CSV_PATH = DATA_DIR / "DrK_videos.csv"
# Legacy copies kept in sync for backward compatibility (see build_dataset.py).
LEGACY_CSV_PATHS = [ROOT / "Scrapper" / "DrK_videos.csv", ROOT / "dataset" / "DrK_videos.csv"]

# Playlists to scrape (the 10 Dr. K playlists).
PLAYLISTS_JSON = ROOT / "Scrapper" / "Data.json"

# CSV schema — order matters; matches the original dataset exactly.
CSV_COLUMNS = [
    "playlist_tag", "playlist_url", "video_url", "channel_id", "channel_url",
    "video_title", "video_length", "video_publish_date", "video_rating",
    "video_views", "video_author", "video_keywords", "video_description",
    "video_transcript",
]

# --- LLM (vLLM, OpenAI-compatible) -----------------------------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:50033/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy-key")
VLLM_MODEL = os.getenv("VLLM_MODEL", "google/gemma-4-E4B-it")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# --- Embeddings ------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda:0")
# bge models want this instruction prefixed to *queries* (not passages).
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- WhisperX --------------------------------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
WHISPER_DEVICE_INDEX = int(os.getenv("WHISPER_DEVICE_INDEX", "0"))

# --- Chroma / retrieval ----------------------------------------------------
CHROMA_COLLECTION = "drk_transcripts"

# Chunking (token-aware; bge-small caps at 512 tokens).
CHUNK_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "450"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))

# Deterministic transcript cleaning before chunking. Empirically validated to
# improve retrieval (ΔMRR +0.024 hybrid / +0.032 dense on realistic queries,
# bootstrap CI excludes 0) at zero cost — see data_analysis/retrieval_experiment.md.
CLEAN_TRANSCRIPTS = os.getenv("CLEAN_TRANSCRIPTS", "1") not in ("0", "false", "False")

# Hybrid retrieval.
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "6"))         # final chunks fed to the LLM
RETRIEVE_POOL = int(os.getenv("RETRIEVE_POOL", "30"))  # candidates per retriever before fusion
RRF_K = int(os.getenv("RRF_K", "60"))                   # Reciprocal Rank Fusion constant

# Reranker (cross-encoder): re-scores the top-RERANK_POOL hybrid candidates for
# query relevance before taking the final RETRIEVE_K. Default off until the A/B
# experiment (data_analysis/rerank_experiment.py) justifies the added latency.
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_POOL = int(os.getenv("RERANK_POOL", "50"))      # candidates fed to the reranker
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "0") not in ("0", "false", "False")

# Playlists whose transcripts are legitimately sparse (skip truncation re-do).
SPARSE_SPEECH_PLAYLISTS = {"Meditation"}

# --- Guardrails ------------------------------------------------------------
# Safety screening of user input (crisis/self-harm/harm-to-others) and model
# output. Deterministic lexicon always runs; the LLM classifier adds recall for
# nuanced phrasing at the cost of one extra short model call per message.
GUARD_ENABLED = os.getenv("GUARD_ENABLED", "1") not in ("0", "false", "False")
GUARD_LLM_CLASSIFIER = os.getenv("GUARD_LLM_CLASSIFIER", "1") not in ("0", "false", "False")
# Reject over-long single messages (DoS / cost guard for a chat turn; ~1500 words).
GUARD_MAX_INPUT_CHARS = int(os.getenv("GUARD_MAX_INPUT_CHARS", "8000"))


def ensure_dirs() -> None:
    """Create the data directories if missing (safe to call repeatedly)."""
    for d in (DATA_DIR, AUDIO_DIR, SEGMENTS_DIR, CHROMA_DIR):
        d.mkdir(parents=True, exist_ok=True)
