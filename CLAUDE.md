# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Enlighten AI / **DrK_Chat** is a RAG chatbot that grounds mental-health/self-help conversations in transcripts of Dr. K (HealthyGamerGG) YouTube videos. It is a reflective companion, **not** a therapist replacement — the mental-health safety layer (disclaimer + crisis-resource injection) is load-bearing, not decoration.

The git repository and all code live in the **`Enlighten-AI/` subdirectory** (the working directory is its parent). All paths below are relative to `Enlighten-AI/`.

## Environment & commands

Everything runs in the **`enlighten` conda env** (Python 3.12, `uv`). The vLLM Gemma server and the embedding/WhisperX models all share the local H100s.

```bash
conda activate enlighten          # or prefix commands with: conda run -n enlighten
python -m Scrapper.build_dataset [--only-new|--refresh-truncated|--playlist NAME|--limit N|--no-whisper]
python -m DrK_Chat.ingest [--rebuild]
python -m data_analysis.eda
streamlit run DrK_Chat/app.py
```

Run module commands from the repo root so `config.py` (a top-level module) and the `Scrapper`/`DrK_Chat` packages resolve. Dependencies install via `uv pip install`; **torch CUDA build must be installed before whisperx** or whisperx silently pulls a CPU torch.

## Architecture (data flows in one direction)

1. **Scrape/transcribe** (`Scrapper/`): `build_dataset.py` is the orchestrator. `playlists.py` uses **yt-dlp** (pytube is gone) to enumerate the 10 playlists in `Data.json` and map metadata onto the 14-column CSV schema. `transcripts.py` fetches **YouTube captions first** (`youtube-transcript-api` 1.x — instance API: `.list()`/`.fetch()`); `whisperx_backend.py` is the **GPU fallback** only when captions are missing or `transcripts.is_truncated()` flags them. Output: the canonical `data/DrK_videos.csv` **plus** per-video `data/segments/<id>.json` (timestamped segments — these drive citation links).
2. **Ingest** (`DrK_Chat/ingest.py`): reads `data/segments/*.json` (NOT the CSV — segments carry timestamps), chunks via `chunking.py`, embeds via `embeddings.py` (bge-small on GPU), upserts into a persistent **Chroma** collection. Chunk id = `<video_id>:<chunk_index>`; per video it deletes-then-upserts, so re-runs are idempotent.
3. **Retrieve** (`DrK_Chat/retrieval.py`): **hybrid** — dense (Chroma) + sparse (**BM25 built in-memory from Chroma's own documents at startup**, no separate index), fused with Reciprocal Rank Fusion. An optional cross-encoder **reranker** (`DrK_Chat/rerank.py`, gated by `config.RERANK_ENABLED`, **default off**) can re-score the top candidates — but `data_analysis/rerank_experiment.py` found no significant gain on our eval at ~164 ms/query, so it stays off.
4. **Generate** (`DrK_Chat/rag.py` + `prompts.py`): builds a grounded prompt (safety/persona as a `system` message — verified honored by this Gemma build — context in the user turn), calls vLLM via the `openai` client, returns answer + deduped sources.
4a. **Guardrails** (`DrK_Chat/guardrails.py`, gated by `config.GUARD_ENABLED`, default on): the *hard* safety layer (the system prompt is the soft one). `screen_input` runs a high-precision lexicon + an LLM classifier (`GUARD_LLM_CLASSIFIER`) and **short-circuits crisis/self-harm/harm-to-others/method-seeking/prompt-injection to vetted responses without invoking RAG**; `screen_output` blocks system-prompt leaks (canary) and appends the not-a-professional disclaimer on clinical over-reach. Use `rag.safe_answer()` (not bare `answer()`) for the guarded path; the Streamlit app already does.
4b. **Prompt-injection defense** is layered: input detection (above) + **prompt hardening** in `prompts.py` (retrieved excerpts are spotlighted/delimited as untrusted DATA, plus an anti-injection system directive) + a **canary** (`prompts.SYSTEM_CANARY`) that `screen_output` blocks if leaked. This stops *indirect* injection (poisoned retrieved text) that input screening structurally cannot see. The input screen also **normalizes obfuscation** (`guardrails.normalize`/`compact`: NFKC homoglyph folding, zero-width stripping, leetspeak/spacing de-obfuscation) and enforces a **DoS input-length cap** (`GUARD_MAX_INPUT_CHARS`). Output is rendered with `st.markdown` (no `unsafe_allow_html`) so model/title text can't inject HTML.
4c. **Empirical security coverage** (all green): crisis 100%/0% (`guardrails_eval.py`), injection 100% detection / 0% FP / 0-of-5 end-to-end (`injection_eval.py`), and a red-team over the AI-pentest checklist (`redteam_eval.py`) — **0/15** obfuscated injection/jailbreak/extraction compromised (base64, leetspeak, zero-width, homoglyph, multilingual, payload-split, virtualization, policy-puppetry, DAN, indirect) and **0/5** obfuscated crises missed. Note: obfuscation resistance leans on the LLM classifier — with `GUARD_LLM_CLASSIFIER=0` the hardened lexicon still catches all but pure euphemistic crises, so keep it on. N/A vectors (no surface): tool/plugin abuse, SSRF, command injection, MLOps/cloud, model theft.
5. **UI** (`DrK_Chat/app.py`): Streamlit chat, caches the `Retriever` with `@st.cache_resource`, streams responses, shows timestamped citation links.

`config.py` centralizes all paths, model names, endpoint, and retrieval/chunking knobs (overridable via `.env`).

## Conventions & gotchas

- **Single canonical CSV**: `data/DrK_videos.csv` is the source of truth. `build_dataset.py` reconciles the two legacy copies (`Scrapper/`, `dataset/`) by copying to them at the end — don't edit those directly.
- **One row per video**: the legacy data had duplicate rows for videos in multiple playlists. The dataset is now de-duplicated by `video_id`; full playlist membership lives in each segments JSON's `all_playlist_tags`. CSV row count (157) is intentionally lower than the old 166.
- **`wc -l` on the CSV lies** — backfilled `video_description` fields contain newlines that the CSV quotes. Use pandas to count rows.
- **Sparse-speech playlists** (`config.SPARSE_SPEECH_PLAYLISTS`, e.g. Meditation) are excluded from truncation re-transcription so legitimately quiet videos aren't needlessly re-run.
- **Embedding model is recorded in the Chroma collection metadata**; changing `EMBED_MODEL` requires `ingest --rebuild` (dimensions won't mix).
- **Ingest applies deterministic transcript cleaning** (`transforms.clean_segments`, gated by `config.CLEAN_TRANSCRIPTS`, default on) before chunking — collapses WhisperX stutter repetitions/fillers while preserving segment timestamps. This was A/B-validated (`data_analysis/retrieval_experiment.py` → `retrieval_experiment.md`): cleaning improved retrieval MRR with a bootstrap CI excluding 0; LLM query-surrogate augmentation (V2) did **not** beat baseline on realistic queries and was rejected. So the stored/cited chunk text is lightly cleaned, not byte-identical to the CSV transcript.
- **Out of scope**: `dataset/dataformatting.py` is unrelated Unsloth fine-tuning boilerplate (we do RAG, not fine-tuning). `Scrapper/scrap_data.ipynb` and `scrape_data.py` are the deprecated pytube originals — do not import or build on them.
- **WhisperX uses Silero VAD** (`vad_method="silero"`) and no diarization, deliberately avoiding the gated pyannote model / HF-token requirement.
