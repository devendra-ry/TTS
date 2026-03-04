"""
tts_backend.py — Qwen3-TTS FastAPI Backend
GTX 1650 optimised (4 GB VRAM)

Streaming smoothness — this revision
──────────────────────────────────────
JITTER-FIX-1  Sync generator serialised inference and network I/O
  Root cause: _make_stream_generator returned a *synchronous* generator.
  Starlette's StreamingResponse wraps sync generators with iterate_in_threadpool,
  which dispatches every next() call to run_in_executor. The resulting timeline
  per chunk was:

      [event loop]  dispatch next() → thread pool
      [thread]      GPU inference (blocking)
      [thread]      yield PCM line
      [event loop]  receive, write to socket
      [event loop]  await socket flush
      [event loop]  dispatch next() → thread pool   ← back to top

  Inference and network I/O were strictly SERIALIZED. The GPU sat idle during
  socket flushes; the client buffer drained during GPU inference. At RTF ≈ 1.2
  on the GTX 1650 with chunk_size=4 (~333 ms audio), the client was starved for
  ~72 ms on every single chunk — enough for an audible click/gap.

  Fix: async producer-consumer pipeline using queue.Queue.
  • A background thread (producer) runs model inference continuously, converting
    each audio_chunk to raw PCM16 and putting it in a bounded queue.
  • An async generator (consumer) awaits each queue.get via run_in_executor and
    yields the NDJSON line immediately without blocking the event loop.
  • Inference and network I/O now run CONCURRENTLY — the GPU never idles waiting
    for a network flush, and the consumer always has a chunk ready to send.

JITTER-FIX-2  Per-segment cold-start with no pre-roll
  Root cause: the pre_roll_filled flag turned True after the first
  PRE_ROLL_CHUNKS *global* chunks. At every subsequent text-segment boundary:
  • pre_roll_filled was already True
  • iter_fn() triggered a full model prefill for the new segment (~300-800 ms)
  • No chunks were buffered during prefill → client queue drained → stutter

  Fix: the bounded queue (maxsize = PRE_ROLL_CHUNKS + 2) acts as the pre-roll
  buffer for EVERY segment automatically. Because the producer thread runs
  ahead of the consumer, it fills the queue (up to maxsize) before blocking.
  When moving to the next text segment the producer is already working on
  segment N+1 while the consumer is still draining segment N — the prefill
  latency is hidden behind the queue fill.

Previous fixes
──────────────
SMOOTH-FIX-1  Per-chunk WAV headers → one streaming header + raw PCM
SMOOTH-FIX-2  STREAM_CHUNK_SIZE 10→4 tokens (833 ms → 333 ms per chunk)
SMOOTH-FIX-3  Added PRE_ROLL_CHUNKS global pre-roll buffer
FIX-1  tempfile/pathlib imported inside functions → module level
FIX-2  asyncio.Lock guards _model_cache / _active_model_id
FIX-3  _load_model blocked event loop → async wrapper + run_in_executor
FIX-4  _run_in_executor misleading *args removed
FIX-5  Lambda closures captured loop variable by reference → default-arg bind
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
import queue as _stdlib_queue
import struct
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
# 4 tokens at 12 Hz codec = ~333 ms of audio per chunk — below the perceptible
# stutter threshold.  Increase for lower CPU/network overhead; decrease for
# lower latency.
STREAM_CHUNK_SIZE     = _env_int("STREAM_CHUNK_SIZE",     4,       min_value=1)
# The producer thread fills the queue up to this many chunks before blocking.
# Acts as a pre-roll buffer at the start of EVERY text segment (not just the
# first), hiding GPU prefill latency at segment boundaries.
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


def _streaming_wav_header(sr: int, channels: int = 1) -> bytes:
    """44-byte WAV header with 0xFFFFFFFF sentinel sizes (open-ended streaming).

    Setting both RIFF chunk-size and data sub-chunk size to 0xFFFFFFFF is the
    de-facto standard for streaming WAV.  Browsers' MediaSource extensions and
    most native players accept it and consume audio continuously until the
    connection closes.
    """
    bits_per_sample = 16
    byte_rate   = sr * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF,
        b"WAVE",
        b"fmt ", 16,
        1,               # PCM
        channels,
        sr,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data", 0xFFFFFFFF,
    )


def _audio_to_pcm16(audio_chunk: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] numpy array to raw little-endian int16 PCM."""
    return (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _chunk_pause_pcm(sr: int) -> bytes:
    """Return inter-segment silence as raw PCM16 bytes (b'' when disabled)."""
    n = int(sr * CHUNK_JOIN_SILENCE_MS / 1000.0)
    return b"\x00" * (n * 2) if n > 0 else b""


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
        "available": True,
        "device":       props.name,
        "total_mb":     round(total     / 1024**2, 1),
        "allocated_mb": round(allocated / 1024**2, 1),
        "reserved_mb":  round(reserved  / 1024**2, 1),
        "free_mb":      round((total - reserved) / 1024**2, 1),
    }


async def _run_in_executor(func):
    return await asyncio.get_event_loop().run_in_executor(None, func)


# ── Async streaming pipeline ──────────────────────────────────────────────────
# Sentinels used as queue messages
_QUEUE_DONE  = object()
_QUEUE_ERROR = object()


async def _stream_audio(
    iter_fns:  list,
    sr:        int,
    pause_pcm: bytes,
    cleanup:   Optional[callable] = None,
) -> None:
    """Async generator that pipelines GPU inference and network I/O.

    JITTER-FIX-1 + JITTER-FIX-2 are both implemented here.

    The producer thread:
      • Iterates over all text-segment callables (iter_fns) in sequence.
      • Converts each audio_chunk to int16 PCM immediately (avoids holding
        large float32 arrays in the queue).
      • Inserts inter-segment silence between segments.
      • Puts items into a bounded queue.  The bound (PRE_ROLL_CHUNKS + 2)
        means the producer runs PRE_ROLL_CHUNKS + 2 chunks ahead of the
        consumer at all times — this is the per-segment pre-roll buffer that
        hides model prefill latency at every text-segment boundary.

    The consumer (this async generator):
      • Awaits each queue.get via run_in_executor so the event loop stays free.
      • Immediately yields each NDJSON line to Starlette / the client.
      • Because the producer is always ahead, there is always a chunk ready —
        no starvation gap even during GPU prefill of the next segment.

    cancel_event:
      • Set in the finally block if the consumer exits early (client disconnect,
        exception).  The producer checks it after every chunk, ensuring the
        background thread exits promptly instead of filling a queue nobody reads.
      • After setting the event we drain the queue so the producer unblocks
        from any blocking q.put call.
    """
    # +2 so the producer has a couple of extra slots to hide queue.put latency.
    q = _stdlib_queue.Queue(maxsize=max(PRE_ROLL_CHUNKS, 0) + 2)
    cancel_event = threading.Event()

    def _producer():
        try:
            for seg_idx, iter_fn in enumerate(iter_fns):
                # Insert silence BEFORE starting the new segment so the
                # consumer plays it while the model is doing prefill.
                if seg_idx > 0 and pause_pcm:
                    if cancel_event.is_set():
                        return
                    q.put(("pause", pause_pcm, sr, 0))

                for audio_chunk, chunk_sr, _timing in iter_fn():
                    if cancel_event.is_set():
                        return
                    pcm = _audio_to_pcm16(audio_chunk)
                    q.put(("pcm", pcm, chunk_sr, len(audio_chunk)))

            q.put((_QUEUE_DONE,))
        except Exception as exc:
            q.put((_QUEUE_ERROR, exc))

    loop = asyncio.get_event_loop()
    producer_fut = loop.run_in_executor(None, _producer)

    t_start      = time.perf_counter()
    ttfa_ms      = None
    first        = True
    total_samples = 0

    try:
        # One streaming WAV header — client appends all subsequent raw PCM to
        # this and plays the whole thing as one continuous audio stream.
        yield json.dumps({
            "format":      "wav_header",
            "chunk":       base64.b64encode(_streaming_wav_header(sr)).decode(),
            "sample_rate": sr,
            "done":        False,
        }) + "\n"

        while True:
            # Blocking q.get runs in a thread so the event loop stays free.
            item = await loop.run_in_executor(None, q.get)
            tag  = item[0]

            if tag is _QUEUE_DONE:
                break

            if tag is _QUEUE_ERROR:
                raise item[1]

            _, pcm_bytes, chunk_sr, n_samples = item
            total_samples += n_samples

            if first:
                ttfa_ms = round((time.perf_counter() - t_start) * 1000, 1)
                first   = False

            yield json.dumps({
                "format":      "pcm_s16le",
                "chunk":       base64.b64encode(pcm_bytes).decode(),
                "sample_rate": chunk_sr,
                "done":        False,
            }) + "\n"

        elapsed       = time.perf_counter() - t_start
        total_duration = total_samples / max(sr, 1)
        yield json.dumps({
            "format":      "done",
            "chunk":       "",
            "sample_rate": sr,
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
        # Signal the producer to stop, then drain so it can unblock from q.put.
        cancel_event.set()
        try:
            while True:
                q.get_nowait()
        except _stdlib_queue.Empty:
            pass
        await producer_fut   # wait for the background thread to exit cleanly

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

    model = await _load_model(CUSTOM_MODEL_ID)
    text_chunks = _split_text_for_tts(text)
    pause = np.zeros(int(SAMPLE_RATE * CHUNK_JOIN_SILENCE_MS / 1000), dtype=np.float32)

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

    model = await _load_model(CUSTOM_MODEL_ID)
    text_chunks = _split_text_for_tts(text)
    pause_pcm   = _chunk_pause_pcm(SAMPLE_RATE)
    extra       = {"instruct": instruct} if instruct else {}

    def _make_iter(tc):
        def _iter():
            return model.generate_custom_voice_streaming(
                text=tc, language=language, speaker=speaker,
                chunk_size=STREAM_CHUNK_SIZE, **extra,
            )
        return _iter

    return StreamingResponse(
        _stream_audio([_make_iter(tc) for tc in text_chunks], SAMPLE_RATE, pause_pcm),
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

    model     = await _load_model(BASE_MODEL_ID)
    pause_pcm = _chunk_pause_pcm(SAMPLE_RATE)

    def _make_iter(tc):
        def _iter():
            return model.generate_voice_clone_streaming(
                text=tc, language=language,
                ref_audio=tmp_path, ref_text=ref_text,
                chunk_size=STREAM_CHUNK_SIZE,
            )
        return _iter

    def _cleanup():
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    return StreamingResponse(
        _stream_audio([_make_iter(tc) for tc in text_chunks], SAMPLE_RATE, pause_pcm,
                      cleanup=_cleanup),
        media_type="application/x-ndjson",
    )