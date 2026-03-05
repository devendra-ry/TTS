import asyncio
import base64
import gc
import io
import json
import logging
import os
import queue as stdlib_queue
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Callable, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

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
        log.warning("%s=%s below minimum %s; using default %s", name, value, min_value, default)
        return default
    return value


MODEL_ID = os.getenv("MODEL_ID", "hexgrad/Kokoro-82M")
KOKORO_LANG_CODE = os.getenv("KOKORO_LANG_CODE", "a")
MAX_TEXT_LEN = _env_int("MAX_TEXT_LEN", 300_000, min_value=1_000)
MODEL_CHUNK_TEXT_LEN = _env_int("MODEL_CHUNK_TEXT_LEN", 220, min_value=50)
CHUNK_JOIN_SILENCE_MS = _env_int("CHUNK_JOIN_SILENCE_MS", 0, min_value=0)
PRE_ROLL_CHUNKS = _env_int("PRE_ROLL_CHUNKS", 4, min_value=0)
SAMPLE_RATE = _env_int("SAMPLE_RATE", 24_000, min_value=8_000)

MIN_SPEAKER_ID = 0
MAX_SPEAKER_ID = 9
VOICE_BY_SPEAKER = {
    0: "af_bella",
    1: "af_nicole",
    2: "af_sarah",
    3: "af_sky",
    4: "am_adam",
    5: "am_michael",
    6: "bf_emma",
    7: "bf_isabella",
    8: "bm_george",
    9: "bm_lewis",
}
VOICE_LABEL_BY_SPEAKER = {
    0: "Bella (US Female, A-)",
    1: "Nicole (US Female, B-)",
    2: "Sarah (US Female, C+)",
    3: "Sky (US Female, C-)",
    4: "Adam (US Male, F+)",
    5: "Michael (US Male, C+)",
    6: "Emma (UK Female, B-)",
    7: "Isabella (UK Female, C)",
    8: "George (UK Male, C)",
    9: "Lewis (UK Male, D+)",
}

_HF_TOKEN = os.getenv("HF_TOKEN", "")
if _HF_TOKEN:
    try:
        from huggingface_hub import login as hf_login

        hf_login(token=_HF_TOKEN, add_to_git_credential=False)
        log.info("HuggingFace login OK.")
    except Exception as exc:
        log.warning("HuggingFace login failed: %s", exc)
else:
    log.warning("HF_TOKEN is not set. Set HF_TOKEN=hf_... if model access requires auth.")

_model_cache: dict = {}
_model_loaded: bool = False
_model_lock = asyncio.Lock()


def _numpy_to_wav(audio_np: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _silence_wav(duration_ms: int, sr: int = SAMPLE_RATE) -> bytes:
    n = int(sr * duration_ms / 1000)
    return _numpy_to_wav(np.zeros(n, dtype=np.float32), sr) if n > 0 else b""


def _validate_text(text: str):
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="text is required and must not be empty.")
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds {MAX_TEXT_LEN} chars ({len(text)}).",
        )


def _validate_speaker(sid: int):
    if not (MIN_SPEAKER_ID <= sid <= MAX_SPEAKER_ID):
        raise HTTPException(
            status_code=422,
            detail=f"speaker_id must be {MIN_SPEAKER_ID}-{MAX_SPEAKER_ID}, got {sid}.",
        )


def _split_text(text: str, max_chars: int = MODEL_CHUNK_TEXT_LEN) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    current = ""

    def _flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for sentence in re.split(r"(?<=[.!?;:])\s+", normalized):
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            _flush()
            word_buf = ""
            for word in sentence.split():
                candidate = f"{word_buf} {word}".strip() if word_buf else word
                if len(candidate) <= max_chars:
                    word_buf = candidate
                else:
                    if word_buf:
                        chunks.append(word_buf)
                    word_buf = word if len(word) <= max_chars else ""
                    if len(word) > max_chars:
                        for i in range(0, len(word), max_chars):
                            chunks.append(word[i:i + max_chars])
            if word_buf:
                chunks.append(word_buf)
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            _flush()
            current = sentence

    _flush()
    return [c for c in chunks if c and len(c.strip()) >= 2 and any(ch.isalnum() for ch in c)]


def _vram_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    alloc = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    total = props.total_memory
    return {
        "available": True,
        "device": props.name,
        "total_mb": round(total / 1024**2, 1),
        "allocated_mb": round(alloc / 1024**2, 1),
        "reserved_mb": round(reserved / 1024**2, 1),
        "free_mb": round((total - reserved) / 1024**2, 1),
    }


def _result_to_audio_np(result) -> np.ndarray:
    audio = getattr(result, "audio", None)
    if audio is None and isinstance(result, (tuple, list)) and len(result) >= 3:
        audio = result[2]
    if audio is None:
        raise ValueError("Kokoro returned a chunk without audio.")

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)
    if audio_np.size == 0:
        raise ValueError("Kokoro produced zero samples.")

    return np.clip(audio_np, -1.0, 1.0).astype(np.float32)


def _load_model_sync() -> dict:
    global _model_loaded
    if _model_loaded and _model_cache:
        return _model_cache

    from kokoro import KPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading Kokoro pipeline for %s (lang=%s, device=%s) ...", MODEL_ID, KOKORO_LANG_CODE, device)
    t0 = time.perf_counter()
    pipeline = KPipeline(lang_code=KOKORO_LANG_CODE, repo_id=MODEL_ID, device=device)
    log.info("Model loaded in %.1f s", time.perf_counter() - t0)

    _model_cache["pipeline"] = pipeline
    _model_loaded = True
    return _model_cache


async def _get_model() -> dict:
    async with _model_lock:
        return await asyncio.get_event_loop().run_in_executor(None, _load_model_sync)


def _kokoro_generate(pipeline, text: str, speaker_id: int) -> tuple[np.ndarray, int]:
    text = text.strip()
    if not text or not any(ch.isalnum() for ch in text):
        raise ValueError(f"Skipping degenerate chunk: {text!r}")

    voice = VOICE_BY_SPEAKER.get(speaker_id, VOICE_BY_SPEAKER[0])
    generator = pipeline(text, voice=voice, speed=1.0, split_pattern=r"\n+")
    chunks = [_result_to_audio_np(result) for result in generator]
    if not chunks:
        raise ValueError("Kokoro returned no audio chunks.")

    return np.concatenate(chunks), SAMPLE_RATE


_QUEUE_DONE = object()
_QUEUE_ERROR = object()


async def _stream_audio(iter_fns: list[Callable], cleanup: Optional[Callable] = None):
    q = stdlib_queue.Queue(maxsize=max(PRE_ROLL_CHUNKS, 0) + 2)
    cancel_event = threading.Event()

    def _producer():
        try:
            for seg_idx, iter_fn in enumerate(iter_fns):
                if seg_idx > 0 and CHUNK_JOIN_SILENCE_MS > 0:
                    if cancel_event.is_set():
                        return
                    sil = _silence_wav(CHUNK_JOIN_SILENCE_MS, SAMPLE_RATE)
                    if sil:
                        q.put(("wav", sil, SAMPLE_RATE, 0))

                for audio_np, chunk_sr, _ in iter_fn():
                    if cancel_event.is_set():
                        return
                    q.put(("wav", _numpy_to_wav(audio_np, chunk_sr), chunk_sr, len(audio_np)))

            q.put((_QUEUE_DONE,))
        except Exception as exc:
            q.put((_QUEUE_ERROR, exc))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    loop = asyncio.get_event_loop()
    producer_fut = loop.run_in_executor(None, _producer)

    t_start = time.perf_counter()
    first = True
    ttfa_ms = None
    total_samples = 0

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            tag = item[0]
            if tag is _QUEUE_DONE:
                break
            if tag is _QUEUE_ERROR:
                raise item[1]

            _, wav_bytes, chunk_sr, n_samples = item
            total_samples += n_samples
            if first:
                ttfa_ms = round((time.perf_counter() - t_start) * 1000, 1)
                first = False

            yield json.dumps(
                {
                    "chunk": base64.b64encode(wav_bytes).decode(),
                    "sample_rate": chunk_sr,
                    "done": False,
                }
            ) + "\n"

        elapsed = time.perf_counter() - t_start
        duration = total_samples / max(SAMPLE_RATE, 1)
        yield json.dumps(
            {
                "chunk": "",
                "sample_rate": SAMPLE_RATE,
                "done": True,
                "rtf": round(elapsed / max(duration, 1e-6), 3),
                "ttfa_ms": ttfa_ms,
            }
        ) + "\n"

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield json.dumps({"error": "CUDA OOM - try shorter text.", "done": True}) + "\n"
    except Exception as exc:
        log.exception("Unexpected error in audio stream")
        yield json.dumps({"error": str(exc), "done": True}) + "\n"
    finally:
        cancel_event.set()
        try:
            while True:
                q.get_nowait()
        except stdlib_queue.Empty:
            pass

        await producer_fut
        if cleanup:
            cleanup()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await _get_model()
        log.info("Startup complete - Kokoro ready.")
    except Exception as exc:
        log.error("Failed to load model at startup: %s", exc)
    yield
    log.info("Shutting down ...")
    _model_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Kokoro TTS API", version="1.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_loaded": _model_loaded,
        "sample_rate": SAMPLE_RATE,
        "max_text_length": MAX_TEXT_LEN,
        "model_chunk_text_len": MODEL_CHUNK_TEXT_LEN,
        "chunk_join_silence_ms": CHUNK_JOIN_SILENCE_MS,
        "pre_roll_chunks": PRE_ROLL_CHUNKS,
        "speaker_id_range": [MIN_SPEAKER_ID, MAX_SPEAKER_ID],
        "voices": VOICE_BY_SPEAKER,
        "voice_labels": VOICE_LABEL_BY_SPEAKER,
        "vram": _vram_info(),
    }


@app.post("/tts/generate")
async def tts_generate(text: str = Form(...), speaker_id: int = Form(0)):
    _validate_text(text)
    _validate_speaker(speaker_id)

    mc = await _get_model()
    pipeline = mc["pipeline"]

    chunks = _split_text(text)
    log.info("/tts/generate start: chars=%d chunks=%d speaker=%d", len(text), len(chunks), speaker_id)

    try:
        parts: list[np.ndarray] = []
        for idx, tc in enumerate(chunks):
            t0 = time.perf_counter()
            audio_np, output_sr = await asyncio.get_event_loop().run_in_executor(
                None, lambda t=tc: _kokoro_generate(pipeline, t, speaker_id)
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
                parts.append(np.zeros(int(output_sr * CHUNK_JOIN_SILENCE_MS / 1000), dtype=np.float32))
            parts.append(audio_np)

        merged = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        wav = _numpy_to_wav(merged, SAMPLE_RATE)
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


@app.post("/tts/generate/stream")
async def tts_generate_stream(text: str = Form(...), speaker_id: int = Form(0)):
    _validate_text(text)
    _validate_speaker(speaker_id)

    mc = await _get_model()
    pipeline = mc["pipeline"]

    chunks = _split_text(text)
    log.info("/tts/generate/stream start: chars=%d chunks=%d speaker=%d", len(text), len(chunks), speaker_id)

    def _make_iter(tc: str):
        def _iter():
            audio_np, sr = _kokoro_generate(pipeline, tc, speaker_id)
            yield audio_np, sr, None

        return _iter

    return StreamingResponse(
        _stream_audio([_make_iter(tc) for tc in chunks]),
        media_type="application/x-ndjson",
    )




