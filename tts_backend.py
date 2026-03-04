"""
tts_backend.py — Qwen3-TTS FastAPI Backend
GTX 1650 optimised (4 GB VRAM)

Streaming smoothness fixes (this revision)
───────────────────────────────────────────
SMOOTH-FIX-1  Per-chunk WAV headers caused structural glitches
  Root cause: every model chunk was wrapped in a complete standalone WAV file
  (44-byte header + PCM body). The client decoded a fresh WAV header on every
  message, treated each chunk as a separate audio object, and tried to stitch
  mini-files together in application code. Gaps at boundaries were unavoidable.
  Fix: send ONE streaming WAV header (0xFFFFFFFF sentinel sizes) as the first
  message; every subsequent chunk is raw little-endian int16 PCM. The client
  concatenates into one continuous buffer — the audio player never sees a
  boundary.

SMOOTH-FIX-2  STREAM_CHUNK_SIZE=10 tokens → ~833 ms per chunk
  At the model's 12 Hz codec rate, chunk_size=10 means the client must wait
  ~833 ms between yields, well above the ~200-300 ms threshold where gaps
  become perceptible. Default lowered to 4 tokens (~333 ms).

SMOOTH-FIX-3  No pre-roll buffer to absorb GPU inference jitter
  GPU inference time is not constant. Without a small pre-roll the client's
  audio queue could starve between chunks on a slow GPU frame. A two-chunk
  pre-roll (~667 ms) is held back before the first byte is sent to the client;
  this trades a small fixed latency for continuous playback.

Previous fixes
──────────────
FIX-1  tempfile / pathlib imported inside function bodies -> moved to top level
FIX-2  asyncio.Lock added to guard _model_cache / _active_model_id
FIX-3  _load_model blocked the event loop -> split into sync + async wrapper
FIX-4  _run_in_executor misleading *args removed
FIX-5  lambda closures captured loop variables by reference -> default-arg bind
FIX-6  Unhandled exceptions in stream generators silently truncated the stream
"""

import gc
import io
import json
import base64
import asyncio
import logging
import os
import pathlib
import struct
import tempfile
import time
import re
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

# ── Memory config ─────────────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tts")


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default
    if value < min_value:
        log.warning("%s=%s is below minimum %s; using default %s", name, value, min_value, default)
        return default
    return value


# ── Constants ─────────────────────────────────────────────────────────────────
CUSTOM_MODEL_ID       = os.getenv("CUSTOM_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
BASE_MODEL_ID         = os.getenv("BASE_MODEL_ID",   "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
MAX_TEXT_LEN          = _env_int("MAX_TEXT_LEN",          300_000, min_value=1_000)
MODEL_CHUNK_TEXT_LEN  = _env_int("MODEL_CHUNK_TEXT_LEN",  700,     min_value=100)
CHUNK_JOIN_SILENCE_MS = _env_int("CHUNK_JOIN_SILENCE_MS", 0,       min_value=0)
# SMOOTH-FIX-2: was 10 tokens (~833 ms). 4 tokens ~= 333 ms.
STREAM_CHUNK_SIZE     = _env_int("STREAM_CHUNK_SIZE",     4,       min_value=1)
# SMOOTH-FIX-3: buffer this many model chunks before the first send.
PRE_ROLL_CHUNKS       = _env_int("PRE_ROLL_CHUNKS",       2,       min_value=0)
SAMPLE_RATE           = 24_000

SPEAKERS = ["serena", "vivian", "uncle_fu", "ryan", "aiden",
            "ono_anna", "sohee", "eric", "dylan"]

# ── Model cache ───────────────────────────────────────────────────────────────
_model_cache: dict = {}
_active_model_id: Optional[str] = None
_model_lock = asyncio.Lock()


def _load_model_sync(model_id: str):
    global _active_model_id
    if _active_model_id == model_id and model_id in _model_cache:
        return _model_cache[model_id]

    if _active_model_id and _active_model_id in _model_cache:
        log.info("Evicting model %s from VRAM ...", _active_model_id)
        del _model_cache[_active_model_id]
        gc.collect()
        torch.cuda.empty_cache()

    from faster_qwen3_tts import FasterQwen3TTS
    log.info("Loading %s ...", model_id)
    t0 = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(model_id)
    log.info("Loaded in %.1fs", time.perf_counter() - t0)

    _model_cache[model_id] = model
    _active_model_id = model_id
    return model


async def _load_model(model_id: str):
    async with _model_lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _load_model_sync, model_id)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await _load_model(CUSTOM_MODEL_ID)
        log.info("Startup complete — CustomVoice model ready.")
    except Exception as exc:
        log.error("Failed to load model at startup: %s", exc)
    yield
    log.info("Shutting down ...")
    _model_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Qwen3-TTS API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Audio helpers ─────────────────────────────────────────────────────────────
def _to_wav_bytes(audio_list: list, sr: int = SAMPLE_RATE) -> bytes:
    """Non-streaming path: encode the full concatenated audio as one WAV file."""
    audio = np.concatenate(audio_list)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# SMOOTH-FIX-1 ─────────────────────────────────────────────────────────────────
# Old helper (REMOVED):
#
#   def _chunk_to_wav_bytes(chunk, sr):
#       buf = io.BytesIO()
#       sf.write(buf, chunk, sr, format="WAV", subtype="PCM_16")
#       return buf.getvalue()
#
# Why it caused glitches: every call produced a complete, self-contained WAV
# file. The client received a sequence of mini-files and had to decode a 44-byte
# header on every message. At each file boundary the audio player encountered a
# "new file start", causing a click/gap. There was no way for the player to
# buffer across boundaries because each file declared its own exact data length.
#
# Replacement: one header + raw PCM stream.
# ──────────────────────────────────────────────────────────────────────────────

def _streaming_wav_header(sr: int, channels: int = 1) -> bytes:
    """44-byte WAV header with sentinel sizes for open-ended (streaming) WAV.

    Setting RIFF chunk-size and data sub-chunk size to 0xFFFFFFFF is the
    de-facto standard for streaming WAV. Browsers' MediaSource implementations
    and most native players accept it and consume audio continuously until the
    connection closes.
    """
    bits_per_sample = 16
    byte_rate   = sr * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF,    # total RIFF chunk size — unknown / streaming
        b"WAVE",
        b"fmt ", 16,            # fmt sub-chunk length (always 16 for PCM)
        1,                      # audio format: PCM
        channels,
        sr,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data", 0xFFFFFFFF,    # data sub-chunk size — unknown / streaming
    )


def _audio_to_pcm16(audio_chunk: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] numpy array to raw little-endian int16 PCM."""
    clipped = np.clip(audio_chunk, -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()


def _chunk_pause_pcm(sr: int) -> bytes:
    """Return inter-chunk silence as raw PCM16 bytes (empty when disabled)."""
    pause_samples = int(sr * (CHUNK_JOIN_SILENCE_MS / 1000.0))
    if pause_samples <= 0:
        return b""
    return b"\x00" * (pause_samples * 2)   # int16 zero = 2 zero bytes/sample


def _validate_text(text: str):
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="text is required and must not be empty.")
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(
            status_code=422,
            detail=(
                f"text exceeds {MAX_TEXT_LEN} character limit ({len(text)} chars). "
                "Increase MAX_TEXT_LEN if needed, or submit a shorter input."
            ),
        )


def _split_text_for_tts(text: str, max_chars: int = MODEL_CHUNK_TEXT_LEN) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    sentences = re.split(r"(?<=[.!?;:。！？])\s+", normalized)
    chunks: list[str] = []
    current = ""

    def _flush_current():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            _flush_current()
            words = sentence.split(" ")
            word_chunk = ""
            for word in words:
                if not word:
                    continue
                candidate = f"{word_chunk} {word}".strip() if word_chunk else word
                if len(candidate) <= max_chars:
                    word_chunk = candidate
                    continue
                if word_chunk:
                    chunks.append(word_chunk)
                if len(word) <= max_chars:
                    word_chunk = word
                else:
                    for i in range(0, len(word), max_chars):
                        piece = word[i:i + max_chars]
                        if len(piece) == max_chars:
                            chunks.append(piece)
                        else:
                            word_chunk = piece
            if word_chunk:
                chunks.append(word_chunk)
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            _flush_current()
            current = sentence

    _flush_current()
    return chunks


def _vram_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated(0)
    reserved  = torch.cuda.memory_reserved(0)
    total     = props.total_memory
    return {
        "available": True,
        "device": props.name,
        "total_mb":     round(total     / 1024**2, 1),
        "allocated_mb": round(allocated / 1024**2, 1),
        "reserved_mb":  round(reserved  / 1024**2, 1),
        "free_mb":      round((total - reserved) / 1024**2, 1),
    }


async def _run_in_executor(func):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func)


# ── Shared streaming generator factory ────────────────────────────────────────
def _make_stream_generator(iter_fns: list, sr: int, pause_pcm: bytes):
    """Return a generator that applies all three smoothness fixes.

    iter_fns  — list of zero-argument callables, one per text chunk.  Each
                returns an iterable of (audio_chunk, sr, timing) tuples,
                matching the model's streaming API.  Using a list of callables
                keeps both /tts/custom/stream and /tts/clone/stream DRY.
    sr        — sample rate to declare in the WAV header.
    pause_pcm — raw PCM16 silence to insert between text chunks (b"" to skip).
    """

    def _generator():
        t_start = time.perf_counter()
        first_yield = True
        ttfa_ms = None
        total_samples = 0

        # SMOOTH-FIX-3: accumulate PRE_ROLL_CHUNKS model chunks before sending
        # anything. Absorbs variable-latency GPU frames so the client queue
        # never starves mid-stream.
        pre_roll_buf: list[str] = []
        pre_roll_filled = (PRE_ROLL_CHUNKS == 0)

        def _emit(line: str):
            nonlocal first_yield, ttfa_ms
            if first_yield:
                ttfa_ms = round((time.perf_counter() - t_start) * 1000, 1)
                first_yield = False
            return line

        try:
            # SMOOTH-FIX-1 part A: send one streaming WAV header before any PCM.
            yield _emit(json.dumps({
                "format":      "wav_header",
                "chunk":       base64.b64encode(_streaming_wav_header(sr)).decode(),
                "sample_rate": sr,
                "done":        False,
            }) + "\n")

            for iter_fn in iter_fns:
                for audio_chunk, chunk_sr, _timing in iter_fn():
                    # SMOOTH-FIX-1 part B: raw PCM instead of per-chunk WAV.
                    pcm = _audio_to_pcm16(audio_chunk)
                    total_samples += len(audio_chunk)
                    line = json.dumps({
                        "format":      "pcm_s16le",
                        "chunk":       base64.b64encode(pcm).decode(),
                        "sample_rate": chunk_sr,
                        "done":        False,
                    }) + "\n"

                    if not pre_roll_filled:
                        pre_roll_buf.append(line)
                        if len(pre_roll_buf) >= PRE_ROLL_CHUNKS:
                            pre_roll_filled = True
                            for buffered in pre_roll_buf:
                                yield _emit(buffered)
                            pre_roll_buf.clear()
                    else:
                        yield _emit(line)

                # Silence between text segments (raw PCM, no header needed).
                if pause_pcm:
                    yield _emit(json.dumps({
                        "format":      "pcm_s16le",
                        "chunk":       base64.b64encode(pause_pcm).decode(),
                        "sample_rate": sr,
                        "done":        False,
                    }) + "\n")

            # Flush pre-roll when the whole text was shorter than PRE_ROLL_CHUNKS.
            for buffered in pre_roll_buf:
                yield _emit(buffered)

            elapsed = time.perf_counter() - t_start
            total_duration = total_samples / max(sr, 1)
            rtf = round(elapsed / max(total_duration, 1e-6), 3)
            yield json.dumps({
                "format":      "done",
                "chunk":       "",
                "sample_rate": sr,
                "done":        True,
                "rtf":         rtf,
                "ttfa_ms":     ttfa_ms,
            }) + "\n"

        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            yield json.dumps({"error": "CUDA OOM", "done": True}) + "\n"
        except Exception as exc:
            log.exception("Unexpected error in stream generator")
            yield json.dumps({"error": str(exc), "done": True}) + "\n"
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    return _generator


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_model": _active_model_id,
        "speakers": SPEAKERS,
        "max_text_length": MAX_TEXT_LEN,
        "model_chunk_text_length": MODEL_CHUNK_TEXT_LEN,
        "chunk_join_silence_ms": CHUNK_JOIN_SILENCE_MS,
        "stream_chunk_size": STREAM_CHUNK_SIZE,
        "pre_roll_chunks": PRE_ROLL_CHUNKS,
        "vram": _vram_info(),
    }


# ── /tts/custom ───────────────────────────────────────────────────────────────
@app.post("/tts/custom")
async def tts_custom(
    text:     str           = Form(...),
    language: str           = Form("English"),
    speaker:  str           = Form("aiden"),
    instruct: Optional[str] = Form(None),
):
    _validate_text(text)
    if speaker not in SPEAKERS:
        raise HTTPException(status_code=422, detail=f"Unknown speaker '{speaker}'. Choose from: {SPEAKERS}")

    model = await _load_model(CUSTOM_MODEL_ID)
    text_chunks = _split_text_for_tts(text)

    try:
        merged_audio: list[np.ndarray] = []
        sr = SAMPLE_RATE
        pause = np.zeros(int(sr * CHUNK_JOIN_SILENCE_MS / 1000), dtype=np.float32)

        for idx, text_chunk in enumerate(text_chunks):
            kwargs = dict(text=text_chunk, language=language, speaker=speaker)
            if instruct:
                kwargs["instruct"] = instruct

            audio_list, sr = await _run_in_executor(
                lambda kw=kwargs: model.generate_custom_voice(**kw)
            )
            if idx > 0 and pause.size > 0:
                merged_audio.append(pause)
            merged_audio.extend(audio_list)

        wav = _to_wav_bytes(merged_audio, sr)
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="CUDA OOM — try shorter text.")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


# ── /tts/custom/stream ────────────────────────────────────────────────────────
@app.post("/tts/custom/stream")
async def tts_custom_stream(
    text:     str           = Form(...),
    language: str           = Form("English"),
    speaker:  str           = Form("aiden"),
    instruct: Optional[str] = Form(None),
):
    _validate_text(text)
    if speaker not in SPEAKERS:
        raise HTTPException(status_code=422, detail=f"Unknown speaker '{speaker}'. Choose from: {SPEAKERS}")

    model = await _load_model(CUSTOM_MODEL_ID)
    text_chunks = _split_text_for_tts(text)
    pause_pcm = _chunk_pause_pcm(SAMPLE_RATE)
    extra = {"instruct": instruct} if instruct else {}

    def _make_iter(tc):
        def _iter():
            return model.generate_custom_voice_streaming(
                text=tc, language=language, speaker=speaker,
                chunk_size=STREAM_CHUNK_SIZE, **extra,
            )
        return _iter

    gen = _make_stream_generator([_make_iter(tc) for tc in text_chunks], SAMPLE_RATE, pause_pcm)
    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ── /tts/clone ────────────────────────────────────────────────────────────────
@app.post("/tts/clone")
async def tts_clone(
    text:      str        = Form(...),
    language:  str        = Form("English"),
    ref_text:  str        = Form(...),
    ref_audio: UploadFile = File(...),
):
    _validate_text(text)
    text_chunks = _split_text_for_tts(text)

    audio_bytes = await ref_audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    model = await _load_model(BASE_MODEL_ID)

    try:
        merged_audio: list[np.ndarray] = []
        sr = SAMPLE_RATE
        pause = np.zeros(int(sr * CHUNK_JOIN_SILENCE_MS / 1000), dtype=np.float32)

        for idx, text_chunk in enumerate(text_chunks):
            audio_list, sr = await _run_in_executor(
                lambda tc=text_chunk: model.generate_voice_clone(
                    text=tc, language=language,
                    ref_audio=tmp_path, ref_text=ref_text,
                )
            )
            if idx > 0 and pause.size > 0:
                merged_audio.append(pause)
            merged_audio.extend(audio_list)

        wav = _to_wav_bytes(merged_audio, sr)
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="CUDA OOM — try shorter text.")
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)
        gc.collect()
        torch.cuda.empty_cache()

    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


# ── /tts/clone/stream ─────────────────────────────────────────────────────────
@app.post("/tts/clone/stream")
async def tts_clone_stream(
    text:      str        = Form(...),
    language:  str        = Form("English"),
    ref_text:  str        = Form(...),
    ref_audio: UploadFile = File(...),
):
    _validate_text(text)
    text_chunks = _split_text_for_tts(text)

    audio_bytes = await ref_audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    model = await _load_model(BASE_MODEL_ID)
    pause_pcm = _chunk_pause_pcm(SAMPLE_RATE)

    def _make_iter(tc):
        def _iter():
            return model.generate_voice_clone_streaming(
                text=tc, language=language,
                ref_audio=tmp_path, ref_text=ref_text,
                chunk_size=STREAM_CHUNK_SIZE,
            )
        return _iter

    base_gen = _make_stream_generator([_make_iter(tc) for tc in text_chunks], SAMPLE_RATE, pause_pcm)

    # Ensure tmp_path is deleted when the stream finishes, even on error.
    # Control has already left the route handler at this point so we wrap
    # the generator rather than relying on a try/finally in the route body.
    def _gen_with_cleanup():
        try:
            yield from base_gen()
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

    return StreamingResponse(_gen_with_cleanup(), media_type="application/x-ndjson")