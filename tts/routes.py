"""
tts/routes.py
FastAPI APIRouter — all three TTS endpoints.
Keeps route handlers thin: delegate to model / text_utils / streaming modules.
"""
import asyncio
import gc
import logging
import time

import numpy as np
import torch
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import Response, StreamingResponse

from .audio_utils import numpy_to_wav, vram_info
from .config import (
    CHUNK_JOIN_SILENCE_MS,
    MAX_TEXT_LEN,
    MIN_SPEAKER_ID,
    MAX_SPEAKER_ID,
    MODEL_CHUNK_TEXT_LEN,
    MODEL_ID,
    PRE_ROLL_CHUNKS,
    SAMPLE_RATE,
    VOICE_BY_SPEAKER,
    VOICE_LABEL_BY_SPEAKER,
)
from .model import get_model, is_loaded, kokoro_generate
from .streaming import stream_audio
from .text_utils import split_text, validate_speaker, validate_text

log = logging.getLogger("tts")

router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health")
async def health():
    """Return current model status, config, and VRAM usage."""
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_loaded": is_loaded(),
        "sample_rate": SAMPLE_RATE,
        "max_text_length": MAX_TEXT_LEN,
        "model_chunk_text_len": MODEL_CHUNK_TEXT_LEN,
        "chunk_join_silence_ms": CHUNK_JOIN_SILENCE_MS,
        "pre_roll_chunks": PRE_ROLL_CHUNKS,
        "speaker_id_range": [MIN_SPEAKER_ID, MAX_SPEAKER_ID],
        "voices": VOICE_BY_SPEAKER,
        "voice_labels": VOICE_LABEL_BY_SPEAKER,
        "vram": vram_info(),
    }


# ---------------------------------------------------------------------------
# Non-streaming generation
# ---------------------------------------------------------------------------


@router.post("/tts/generate")
async def tts_generate(text: str = Form(...), speaker_id: int = Form(0)):
    """Generate a complete WAV file and return it as a binary response."""
    validate_text(text)
    validate_speaker(speaker_id)

    mc = await get_model()
    pipeline = mc["pipeline"]

    chunks = split_text(text)
    log.info(
        "/tts/generate start: chars=%d chunks=%d speaker=%d",
        len(text),
        len(chunks),
        speaker_id,
    )

    try:
        parts: list[np.ndarray] = []
        for idx, tc in enumerate(chunks):
            t0 = time.perf_counter()
            audio_np, output_sr = await asyncio.get_event_loop().run_in_executor(
                None, lambda t=tc: kokoro_generate(pipeline, t, speaker_id)
            )
            log.info(
                "/tts/generate chunk %d/%d done in %.2fs (samples=%d sr=%d)",
                idx + 1,
                len(chunks),
                time.perf_counter() - t0,
                len(audio_np),
                output_sr,
            )
            if idx > 0 and CHUNK_JOIN_SILENCE_MS > 0:
                parts.append(
                    np.zeros(
                        int(output_sr * CHUNK_JOIN_SILENCE_MS / 1000), dtype=np.float32
                    )
                )
            parts.append(audio_np)

        merged = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        wav = numpy_to_wav(merged, SAMPLE_RATE)
        log.info("/tts/generate complete: wav_bytes=%d", len(wav))

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="CUDA OOM - try shorter text.")
    except Exception as exc:
        log.exception("/tts/generate failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------


@router.post("/tts/generate/stream")
async def tts_generate_stream(text: str = Form(...), speaker_id: int = Form(0)):
    """Stream WAV chunks as NDJSON.  Each line is a JSON object with a base-64 chunk."""
    validate_text(text)
    validate_speaker(speaker_id)

    mc = await get_model()
    pipeline = mc["pipeline"]

    chunks = split_text(text)
    log.info(
        "/tts/generate/stream start: chars=%d chunks=%d speaker=%d",
        len(text),
        len(chunks),
        speaker_id,
    )

    def _make_iter(tc: str):
        def _iter():
            audio_np, sr = kokoro_generate(pipeline, tc, speaker_id)
            yield audio_np, sr, None

        return _iter

    return StreamingResponse(
        stream_audio([_make_iter(tc) for tc in chunks]),
        media_type="application/x-ndjson",
    )
