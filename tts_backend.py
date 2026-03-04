"""
tts_backend.py — Qwen3-TTS FastAPI Backend
GTX 1650 optimised (4 GB VRAM)

Slow-motion + jitter fixes (this revision)
───────────────────────────────────────────

BUG-1  SLOW MOTION — hardcoded sample rate in WAV header
  The streaming WAV header was built with SAMPLE_RATE=24_000 BEFORE any
  model output was seen.  The model returns the actual sample rate in every
  chunk as chunk_sr.  If chunk_sr != SAMPLE_RATE (e.g. model variant outputs
  22 050 Hz, or a resampling wrapper returns 48 000 Hz) then:
      playback_speed = declared_sr / actual_sr
  A 2x mismatch (declared 24 000, actual 48 000) gives exactly the
  "talking in slow motion" symptom — pitch and tempo both halved.

  Fix (part of BUG-2 fix below): every queue item is a complete WAV file
  whose header is built from the chunk's own chunk_sr, so the declared rate
  always matches the data regardless of what the model returns.

BUG-2  SLOW MOTION / FORMAT — streaming WAV header + raw PCM over NDJSON
  The previous revision split the audio into a one-time "wav_header" message
  followed by raw "pcm_s16le" messages.  This custom binary protocol breaks
  every standard audio decoder:

    • Browser decodeAudioData (most common path):
        wav_header msg  → decodeAudioData(44 bytes) → error / silence
        pcm_s16le msg   → decodeAudioData(raw int16) → error
      Result: SILENCE.

    • Client buffers everything then decodes:
      Must receive the full stream before playback starts — not streaming.

    • Client manually converts int16 → Float32Array:
      The ONLY path that works. Requires bespoke code. Any mistake in the
      client's assumed sample rate gives SLOW MOTION (e.g. using the
      AudioContext's default 48 000 Hz instead of 24 000 Hz for the buffer
      means 24 000 samples play in 24000/48000 = 0.5 seconds → half-speed).

  Fix: revert to per-chunk COMPLETE WAV files.  Each queue item is a
  standalone WAV (44-byte header + PCM body) at the correct chunk_sr.
  Browser decodeAudioData works on every chunk independently, and the
  embedded sample rate is always correct.

  The boundary glitch that existed in the original code was a CLIENT-SIDE
  scheduling bug (AudioBufferSourceNode not given an explicit start time),
  NOT a server format bug.  The right fix is proper client scheduling, not
  changing the server format.

BUG-3  JITTER — queue too shallow for RTF > 1 on GTX 1650
  Queue size was PRE_ROLL_CHUNKS(2) + 2 = 4 chunks × 333 ms = 1.33 s.
  At RTF=1.3 (typical for this card on longer text) the queue drains at
  100 ms/chunk and is empty after only 4.3 s of audio.  All subsequent
  chunks arrive late → repeating ~100 ms gaps.

  Fix: raise PRE_ROLL_CHUNKS default to 6 → queue = 8 chunks = 2.67 s.
  At RTF=1.3 the queue now covers 6.7 s before any starvation, which is
  enough for the majority of TTS requests.

Previous fixes
──────────────
SMOOTH-FIX-1  Per-chunk WAV headers → one streaming header + raw PCM
              (REVERTED — the streaming header format was the bug)
SMOOTH-FIX-2  STREAM_CHUNK_SIZE 10→4 tokens (833 ms → 333 ms)  [kept]
SMOOTH-FIX-3  PRE_ROLL_CHUNKS global pre-roll                    [extended to 6]
JITTER-FIX-1  Async producer-consumer pipeline                   [kept]
JITTER-FIX-2  Per-segment pre-roll via bounded queue             [kept]
FIX-1   tempfile/pathlib inside functions → module level
FIX-2   asyncio.Lock guards _model_cache / _active_model_id
FIX-3   _load_model blocked event loop → async + run_in_executor
FIX-4   _run_in_executor misleading *args removed
FIX-5   Lambda closures by reference → default-arg capture
FIX-6   Unhandled exceptions silently truncated stream
"""

import gc
import io
import json
import base64
import asyncio
import logging
import os
import pathlib
import queue as _stdlib_queue
import tempfile
import threading
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
# 4 tokens × (24 000 / 12) samples/token = 8 000 samples = 333 ms per chunk.
STREAM_CHUNK_SIZE     = _env_int("STREAM_CHUNK_SIZE",     4,       min_value=1)
# BUG-3 FIX: was 2 (1.33 s buffer). 6 gives 2.67 s → covers RTF≈1.3 for
# up to 6.7 s of audio before any queue starvation.
PRE_ROLL_CHUNKS       = _env_int("PRE_ROLL_CHUNKS",       6,       min_value=0)
SAMPLE_RATE           = 24_000   # fallback; actual rate always comes from chunk_sr

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
    """Non-streaming: encode the full concatenated audio as one WAV file."""
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(audio_list), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _chunk_to_wav_bytes(audio_chunk: np.ndarray, sr: int) -> bytes:
    """BUG-1 + BUG-2 FIX: encode a single chunk as a complete, standalone WAV.

    Using sr=chunk_sr (from the model) instead of the global SAMPLE_RATE
    constant means the WAV header ALWAYS matches the actual data rate,
    regardless of which model variant or wrapper is in use.

    A complete WAV per chunk is required because:
      • Browser decodeAudioData needs a full container (header + data).
      • Clients can schedule each decoded AudioBuffer at a precise time
        to achieve seamless, gap-free playback.
      • No bespoke PCM-handling code is needed on the client side.
    """
    buf = io.BytesIO()
    sf.write(buf, audio_chunk, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _silence_wav_bytes(duration_ms: int, sr: int) -> bytes:
    """Return a complete WAV file containing silence of the given duration."""
    n_samples = int(sr * duration_ms / 1000)
    if n_samples <= 0:
        return b""
    return _chunk_to_wav_bytes(np.zeros(n_samples, dtype=np.float32), sr)


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

    def _flush():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            _flush()
            word_chunk = ""
            for word in sentence.split(" "):
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
            _flush()
            current = sentence

    _flush()
    return chunks


def _vram_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated(0)
    reserved  = torch.cuda.memory_reserved(0)
    total     = props.total_memory
    return {
        "available":    True,
        "device":       props.name,
        "total_mb":     round(total     / 1024**2, 1),
        "allocated_mb": round(allocated / 1024**2, 1),
        "reserved_mb":  round(reserved  / 1024**2, 1),
        "free_mb":      round((total - reserved) / 1024**2, 1),
    }


async def _run_in_executor(func):
    return await asyncio.get_event_loop().run_in_executor(None, func)


# ── Async streaming pipeline ──────────────────────────────────────────────────
_QUEUE_DONE  = object()
_QUEUE_ERROR = object()


async def _stream_audio(
    iter_fns:  list,
    fallback_sr: int,
    cleanup:   Optional[callable] = None,
) -> None:
    """Async generator — concurrent GPU inference + network I/O pipeline.

    Each queue item is a COMPLETE WAV file (complete header + PCM body) built
    from the model's own chunk_sr.  The client calls decodeAudioData on each
    chunk and schedules playback with precise AudioContext timing.

    Queue size = PRE_ROLL_CHUNKS + 2 so the producer thread is always
    PRE_ROLL_CHUNKS chunks ahead of the consumer, hiding both initial
    startup latency and per-segment prefill latency (BUG-3 FIX).
    """
    q            = _stdlib_queue.Queue(maxsize=max(PRE_ROLL_CHUNKS, 0) + 2)
    cancel_event = threading.Event()

    def _producer():
        try:
            for seg_idx, iter_fn in enumerate(iter_fns):
                # Silence BEFORE new segment so the consumer plays it
                # concurrently with the model's prefill for that segment.
                if seg_idx > 0 and CHUNK_JOIN_SILENCE_MS > 0:
                    if cancel_event.is_set():
                        return
                    # Use fallback_sr for silence; rate doesn't matter for zeros.
                    sil = _silence_wav_bytes(CHUNK_JOIN_SILENCE_MS, fallback_sr)
                    if sil:
                        q.put(("wav", sil, fallback_sr, 0))

                for audio_chunk, chunk_sr, _timing in iter_fn():
                    if cancel_event.is_set():
                        return
                    # BUG-1 + BUG-2 FIX: complete WAV using model's chunk_sr.
                    wav = _chunk_to_wav_bytes(audio_chunk, chunk_sr)
                    q.put(("wav", wav, chunk_sr, len(audio_chunk)))

            q.put((_QUEUE_DONE,))
        except Exception as exc:
            q.put((_QUEUE_ERROR, exc))

    loop         = asyncio.get_event_loop()
    producer_fut = loop.run_in_executor(None, _producer)

    t_start       = time.perf_counter()
    ttfa_ms       = None
    first         = True
    total_samples = 0

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            tag  = item[0]

            if tag is _QUEUE_DONE:
                break
            if tag is _QUEUE_ERROR:
                raise item[1]

            _, wav_bytes, chunk_sr, n_samples = item
            total_samples += n_samples

            if first:
                ttfa_ms = round((time.perf_counter() - t_start) * 1000, 1)
                first   = False

            yield json.dumps({
                "chunk":       base64.b64encode(wav_bytes).decode(),
                "sample_rate": chunk_sr,
                "done":        False,
            }) + "\n"

        elapsed        = time.perf_counter() - t_start
        total_duration = total_samples / max(fallback_sr, 1)
        yield json.dumps({
            "chunk":       "",
            "sample_rate": fallback_sr,
            "done":        True,
            "rtf":         round(elapsed / max(total_duration, 1e-6), 3),
            "ttfa_ms":     ttfa_ms,
        }) + "\n"

    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        yield json.dumps({"error": "CUDA OOM", "done": True}) + "\n"
    except Exception as exc:
        log.exception("Unexpected error in audio stream")
        yield json.dumps({"error": str(exc), "done": True}) + "\n"
    finally:
        # Signal producer and drain queue so any blocking q.put() can exit.
        cancel_event.set()
        try:
            while True:
                q.get_nowait()
        except _stdlib_queue.Empty:
            pass
        await producer_fut
        if cleanup:
            cleanup()
        gc.collect()
        torch.cuda.empty_cache()


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

    model       = await _load_model(CUSTOM_MODEL_ID)
    text_chunks = _split_text_for_tts(text)
    pause       = np.zeros(int(SAMPLE_RATE * CHUNK_JOIN_SILENCE_MS / 1000), dtype=np.float32)

    try:
        merged: list[np.ndarray] = []
        sr = SAMPLE_RATE
        for idx, tc in enumerate(text_chunks):
            kw = dict(text=tc, language=language, speaker=speaker)
            if instruct:
                kw["instruct"] = instruct
            audio_list, sr = await _run_in_executor(lambda k=kw: model.generate_custom_voice(**k))
            if idx > 0 and pause.size > 0:
                merged.append(pause)
            merged.extend(audio_list)
        wav = _to_wav_bytes(merged, sr)
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="CUDA OOM — try shorter text.")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return Response(content=wav, media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=output.wav"})


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

    model       = await _load_model(CUSTOM_MODEL_ID)
    text_chunks = _split_text_for_tts(text)
    extra       = {"instruct": instruct} if instruct else {}

    def _make_iter(tc):
        def _iter():
            return model.generate_custom_voice_streaming(
                text=tc, language=language, speaker=speaker,
                chunk_size=STREAM_CHUNK_SIZE, **extra,
            )
        return _iter

    return StreamingResponse(
        _stream_audio([_make_iter(tc) for tc in text_chunks], SAMPLE_RATE),
        media_type="application/x-ndjson",
    )


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
    pause = np.zeros(int(SAMPLE_RATE * CHUNK_JOIN_SILENCE_MS / 1000), dtype=np.float32)

    try:
        merged: list[np.ndarray] = []
        sr = SAMPLE_RATE
        for idx, tc in enumerate(text_chunks):
            audio_list, sr = await _run_in_executor(
                lambda t=tc: model.generate_voice_clone(
                    text=t, language=language,
                    ref_audio=tmp_path, ref_text=ref_text,
                )
            )
            if idx > 0 and pause.size > 0:
                merged.append(pause)
            merged.extend(audio_list)
        wav = _to_wav_bytes(merged, sr)
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="CUDA OOM — try shorter text.")
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)
        gc.collect()
        torch.cuda.empty_cache()

    return Response(content=wav, media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=output.wav"})


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

    def _make_iter(tc):
        def _iter():
            return model.generate_voice_clone_streaming(
                text=tc, language=language,
                ref_audio=tmp_path, ref_text=ref_text,
                chunk_size=STREAM_CHUNK_SIZE,
            )
        return _iter

    return StreamingResponse(
        _stream_audio(
            [_make_iter(tc) for tc in text_chunks],
            SAMPLE_RATE,
            cleanup=lambda: pathlib.Path(tmp_path).unlink(missing_ok=True),
        ),
        media_type="application/x-ndjson",
    )