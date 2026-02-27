"""
tts_backend.py — Qwen3-TTS FastAPI Backend
GTX 1650 optimised (4 GB VRAM)
"""

import os
import gc
import io
import json
import base64
import asyncio
import logging
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
CUSTOM_MODEL_ID = os.getenv("CUSTOM_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
BASE_MODEL_ID   = os.getenv("BASE_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
MAX_TEXT_LEN    = _env_int("MAX_TEXT_LEN", 300000, min_value=1000)
MODEL_CHUNK_TEXT_LEN = _env_int("MODEL_CHUNK_TEXT_LEN", 700, min_value=100)
CHUNK_JOIN_SILENCE_MS = _env_int("CHUNK_JOIN_SILENCE_MS", 0, min_value=0)
STREAM_CHUNK_SIZE = _env_int("STREAM_CHUNK_SIZE", 10, min_value=1)
SAMPLE_RATE     = 24_000

SPEAKERS = ["serena", "vivian", "uncle_fu", "ryan", "aiden",
            "ono_anna", "sohee", "eric", "dylan"]

# ── Model cache ───────────────────────────────────────────────────────────────
# We keep at most one model loaded at a time to stay within 4 GB VRAM.
_model_cache: dict = {}   # {"id": model_instance}
_active_model_id: Optional[str] = None


def _load_model(model_id: str):
    global _active_model_id
    if _active_model_id == model_id and model_id in _model_cache:
        return _model_cache[model_id]

    # Evict the previously loaded model
    if _active_model_id and _active_model_id in _model_cache:
        log.info("Evicting model %s from VRAM …", _active_model_id)
        del _model_cache[_active_model_id]
        gc.collect()
        torch.cuda.empty_cache()

    from faster_qwen3_tts import FasterQwen3TTS
    log.info("Loading %s …", model_id)
    t0 = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(model_id)
    log.info("Loaded in %.1fs", time.perf_counter() - t0)

    _model_cache[model_id] = model
    _active_model_id = model_id
    return model


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the CustomVoice model (more common use-case)
    try:
        _load_model(CUSTOM_MODEL_ID)
        log.info("Startup complete — CustomVoice model ready.")
    except Exception as exc:
        log.error("Failed to load model at startup: %s", exc)
    yield
    log.info("Shutting down …")
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_wav_bytes(audio_list: list, sr: int = SAMPLE_RATE) -> bytes:
    audio = np.concatenate(audio_list)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _chunk_to_wav_bytes(audio_chunk: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_chunk, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _validate_text(text: str):
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="text is required and must not be empty.")
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds {MAX_TEXT_LEN} character limit ({len(text)} chars). "
                   "Increase MAX_TEXT_LEN if needed, or submit a shorter input.",
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
                    # Hard split pathological tokens (e.g. long URLs).
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


def _chunk_pause(sr: int) -> np.ndarray:
    pause_samples = int(sr * (CHUNK_JOIN_SILENCE_MS / 1000.0))
    if pause_samples <= 0:
        return np.array([], dtype=np.float32)
    return np.zeros(pause_samples, dtype=np.float32)


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
        "total_mb": round(total / 1024**2, 1),
        "allocated_mb": round(allocated / 1024**2, 1),
        "reserved_mb": round(reserved / 1024**2, 1),
        "free_mb": round((total - reserved) / 1024**2, 1),
    }


async def _run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


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
        "vram": _vram_info(),
    }


# ── /tts/custom ───────────────────────────────────────────────────────────────
@app.post("/tts/custom")
async def tts_custom(
    text:     str = Form(...),
    language: str = Form("English"),
    speaker:  str = Form("aiden"),
    instruct: Optional[str] = Form(None),
):
    _validate_text(text)
    if speaker not in SPEAKERS:
        raise HTTPException(status_code=422, detail=f"Unknown speaker '{speaker}'. Choose from: {SPEAKERS}")

    model = _load_model(CUSTOM_MODEL_ID)
    text_chunks = _split_text_for_tts(text)

    try:
        merged_audio: list[np.ndarray] = []
        sr = SAMPLE_RATE
        pause = _chunk_pause(sr)

        for idx, text_chunk in enumerate(text_chunks):
            kwargs = dict(text=text_chunk, language=language, speaker=speaker)
            if instruct:
                kwargs["instruct"] = instruct

            audio_list, sr = await _run_in_executor(lambda: model.generate_custom_voice(**kwargs))
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
    text:     str = Form(...),
    language: str = Form("English"),
    speaker:  str = Form("aiden"),
    instruct: Optional[str] = Form(None),
):
    _validate_text(text)
    if speaker not in SPEAKERS:
        raise HTTPException(status_code=422, detail=f"Unknown speaker '{speaker}'. Choose from: {SPEAKERS}")

    model = _load_model(CUSTOM_MODEL_ID)
    text_chunks = _split_text_for_tts(text)
    pause = _chunk_pause(SAMPLE_RATE)
    pause_wav_b64 = ""
    if pause.size > 0:
        pause_wav_b64 = base64.b64encode(_chunk_to_wav_bytes(pause, SAMPLE_RATE)).decode()

    def _generator():
        t_start = time.perf_counter()
        first_chunk = True
        ttfa_ms = None
        total_duration = 0.0
        try:
            for idx, text_chunk in enumerate(text_chunks):
                kwargs = dict(text=text_chunk, language=language, speaker=speaker, chunk_size=STREAM_CHUNK_SIZE)
                if instruct:
                    kwargs["instruct"] = instruct

                for audio_chunk, sr, timing in model.generate_custom_voice_streaming(**kwargs):
                    if first_chunk:
                        ttfa_ms = round((time.perf_counter() - t_start) * 1000, 1)
                        first_chunk = False
                    wav_bytes = _chunk_to_wav_bytes(audio_chunk, sr)
                    b64 = base64.b64encode(wav_bytes).decode()
                    total_duration += len(audio_chunk) / sr
                    line = json.dumps({"chunk": b64, "sample_rate": sr, "done": False})
                    yield line + "\n"

                if idx < len(text_chunks) - 1 and pause_wav_b64:
                    total_duration += len(pause) / SAMPLE_RATE
                    yield json.dumps({"chunk": pause_wav_b64, "sample_rate": SAMPLE_RATE, "done": False}) + "\n"
            elapsed = time.perf_counter() - t_start
            rtf = round(elapsed / max(total_duration, 1e-6), 3)
            final = json.dumps({"chunk": "", "sample_rate": SAMPLE_RATE, "done": True,
                                "rtf": rtf, "ttfa_ms": ttfa_ms})
            yield final + "\n"
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            err = json.dumps({"error": "CUDA OOM", "done": True})
            yield err + "\n"
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    return StreamingResponse(_generator(), media_type="application/x-ndjson")


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

    # Save uploaded ref audio to a temp file
    import tempfile, pathlib
    audio_bytes = await ref_audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    model = _load_model(BASE_MODEL_ID)

    try:
        merged_audio: list[np.ndarray] = []
        sr = SAMPLE_RATE
        pause = _chunk_pause(sr)

        for idx, text_chunk in enumerate(text_chunks):
            audio_list, sr = await _run_in_executor(
                lambda: model.generate_voice_clone(
                    text=text_chunk, language=language,
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

    import tempfile, pathlib
    audio_bytes = await ref_audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    model = _load_model(BASE_MODEL_ID)
    pause = _chunk_pause(SAMPLE_RATE)
    pause_wav_b64 = ""
    if pause.size > 0:
        pause_wav_b64 = base64.b64encode(_chunk_to_wav_bytes(pause, SAMPLE_RATE)).decode()

    def _generator():
        t_start = time.perf_counter()
        first_chunk = True
        ttfa_ms = None
        total_duration = 0.0
        try:
            for idx, text_chunk in enumerate(text_chunks):
                for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
                    text=text_chunk, language=language,
                    ref_audio=tmp_path, ref_text=ref_text,
                    chunk_size=STREAM_CHUNK_SIZE,
                ):
                    if first_chunk:
                        ttfa_ms = round((time.perf_counter() - t_start) * 1000, 1)
                        first_chunk = False
                    wav_bytes = _chunk_to_wav_bytes(audio_chunk, sr)
                    b64 = base64.b64encode(wav_bytes).decode()
                    total_duration += len(audio_chunk) / sr
                    yield json.dumps({"chunk": b64, "sample_rate": sr, "done": False}) + "\n"

                if idx < len(text_chunks) - 1 and pause_wav_b64:
                    total_duration += len(pause) / SAMPLE_RATE
                    yield json.dumps({"chunk": pause_wav_b64, "sample_rate": SAMPLE_RATE, "done": False}) + "\n"
            elapsed = time.perf_counter() - t_start
            rtf = round(elapsed / max(total_duration, 1e-6), 3)
            yield json.dumps({"chunk": "", "sample_rate": SAMPLE_RATE, "done": True,
                              "rtf": rtf, "ttfa_ms": ttfa_ms}) + "\n"
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            yield json.dumps({"error": "CUDA OOM", "done": True}) + "\n"
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)
            gc.collect()
            torch.cuda.empty_cache()

    return StreamingResponse(_generator(), media_type="application/x-ndjson")
