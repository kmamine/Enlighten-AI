"""WhisperX transcription fallback (GPU).

Used only when YouTube captions are unavailable or an existing transcript looks
truncated. Downloads the audio with yt-dlp, transcribes with faster-whisper
(via WhisperX) and force-aligns for word-level timestamps. Diarization is NOT
used (avoids the gated pyannote model / HF token). The heavy models are loaded
lazily and cached for the lifetime of the process.
"""
from __future__ import annotations

from yt_dlp import YoutubeDL

import config
from .playlists import canonical_watch_url
from .transcripts import Segment

_model = None                # FasterWhisperPipeline
_align_cache: dict[str, tuple] = {}  # language_code -> (align_model, metadata)


def _get_model():
    global _model
    if _model is None:
        import whisperx
        _model = whisperx.load_model(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            device_index=config.WHISPER_DEVICE_INDEX,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            language=config.WHISPER_LANGUAGE,
            vad_method="silero",  # avoids gated pyannote model
        )
    return _model


def _get_align_model(language_code: str):
    if language_code not in _align_cache:
        import whisperx
        _align_cache[language_code] = whisperx.load_align_model(
            language_code=language_code, device=config.WHISPER_DEVICE
        )
    return _align_cache[language_code]


def download_audio(video_id: str) -> str | None:
    """Download bestaudio for a video to data/audio/<id>.wav. Returns path or None."""
    config.ensure_dirs()
    out_template = str(config.AUDIO_DIR / "%(id)s.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "retries": 5,
        "sleep_interval_requests": 1,
    }
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([canonical_watch_url(video_id)])
    except Exception as exc:  # noqa: BLE001
        print(f"  [whisperx] audio download failed for {video_id}: {exc}")
        return None
    wav = config.AUDIO_DIR / f"{video_id}.wav"
    return str(wav) if wav.exists() else None


def transcribe(video_id: str, batch_size: int = 16) -> list[Segment] | None:
    """Transcribe + align a video with WhisperX. Returns segments or None.

    The temporary audio file is always cleaned up.
    """
    import whisperx

    audio_path = download_audio(video_id)
    if not audio_path:
        return None
    try:
        audio = whisperx.load_audio(audio_path)
        result = _get_model().transcribe(audio, batch_size=batch_size)
        language = result.get("language", config.WHISPER_LANGUAGE)
        align_model, metadata = _get_align_model(language)
        aligned = whisperx.align(
            result["segments"], align_model, metadata, audio,
            config.WHISPER_DEVICE, return_char_alignments=False,
        )
        segs: list[Segment] = []
        for s in aligned.get("segments", []):
            text = (s.get("text") or "").strip()
            if text:
                segs.append({
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "text": text,
                })
        return segs or None
    except Exception as exc:  # noqa: BLE001
        print(f"  [whisperx] transcription failed for {video_id}: {exc}")
        return None
    finally:
        try:
            from pathlib import Path
            Path(audio_path).unlink(missing_ok=True)
        except Exception:
            pass
