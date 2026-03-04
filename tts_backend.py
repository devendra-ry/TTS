"""
tts_backend.py  —  sesame/csm-1b  FastAPI Backend
GTX 1650 (4 GB VRAM)

Prerequisites
─────────────
1.  pip install "transformers>=4.52.1" fastapi uvicorn soundfile torchaudio accelerate huggingface_hub
2.  Accept the gated-model agreements on HuggingFace for BOTH:
      • https://huggingface.co/sesame/csm-1b
      • https://huggingface.co/meta-llama/Llama-3.2-1B   (CSM's tokeniser depends on it)
3.  Set your token before starting:
      export HF_TOKEN=hf_...

Model facts
────────────
• CsmForConditionalGeneration  +  AutoProcessor  (transformers ≥ 4.52.1)
• English-only (Llama backbone + Mimi vocoder)
• Output sample rate: always 24 000 Hz
• Speaker identity: integer 0–9 passed as role string "0"–"9"
• model.generate(**inputs, output_audio=True) returns a (1, num_samples) float32 tensor
• Voice cloning: reference waveform (numpy float32, 24 kHz) placed in the conversation
  context turn before the target text turn

GTX 1650 VRAM budget
─────────────────────
CSM-1B at float16 ≈ 2 GB → ~2 GB left for KV-cache + activations.
  torch_dtype=torch.float16
  device_map="auto"         (Accelerate maps layers optimally)
  low_cpu_mem_usage=True    (stream weights; avoids double-copy at load)
  torch.no_grad() on every generate call
  gc + cuda.empty_cache() after every request
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
import importlib.metadata
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

# ── transformers version guard ────────────────────────────────────────────────
_TRANSFORMERS_MIN = (4, 52, 1)
try:
    _tv = tuple(int(x) for x in importlib.metadata.version("transformers").split(".")[:3])
    if _tv < _TRANSFORMERS_MIN:
        raise RuntimeError(
            f"transformers {'.'.join(str(x) for x in _tv)} is installed but "
            f">={'.'.join(str(x) for x in _TRANSFORMERS_MIN)} is required for CSM-1B. "
            "Run: pip install -U 'transformers>=4.52.1'"
        )
except importlib.metadata.PackageNotFoundError:
    raise RuntimeError("transformers not installed. Run: pip install 'transformers>=4.52.1'")

# ── Memory / compile config ───────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NO_TORCH_COMPILE", "1")   # Mimi uses torch.compile; skip on GTX 1650

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
        log.warning("%s=%s below minimum %s; using default %s", name, value, min_value, default)
        return default
    return value


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID              = os.getenv("MODEL_ID", "sesame/csm-1b")
MAX_TEXT_LEN          = _env_int("MAX_TEXT_LEN",          300_000, min_value=1_000)
MODEL_CHUNK_TEXT_LEN  = _env_int("MODEL_CHUNK_TEXT_LEN",  200,     min_value=50)
CHUNK_JOIN_SILENCE_MS = _env_int("CHUNK_JOIN_SILENCE_MS", 0,       min_value=0)
PRE_ROLL_CHUNKS       = _env_int("PRE_ROLL_CHUNKS",       6,       min_value=0)
SAMPLE_RATE           = 24_000
MIN_SPEAKER_ID        = 0
MAX_SPEAKER_ID        = 9

# ── HuggingFace authentication ────────────────────────────────────────────────
_HF_TOKEN = os.getenv("HF_TOKEN", "")
if not _HF_TOKEN:
    log.warning(
        "HF_TOKEN is not set. sesame/csm-1b is a gated model that also requires "
        "access to meta-llama/Llama-3.2-1B. Set HF_TOKEN=hf_... before starting. "
        "Accept both model agreements at huggingface.co first."
    )
else:
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_HF_TOKEN, add_to_git_credential=False)
        log.info("HuggingFace login OK.")
    except Exception as _exc:
        log.warning("HuggingFace login failed: %s", _exc)

# ── Model cache ───────────────────────────────────────────────────────────────
_model_cache: dict  = {}   # keys: "model", "processor"
_model_loaded: bool = False
_model_lock         = asyncio.Lock()


def _load_model_sync() -> dict:
    """Load CSM-1B + processor in a thread-pool worker so the event loop is free."""
    global _model_loaded
    if _model_loaded and _model_cache:
        return _model_cache

    from transformers import CsmForConditionalGeneration, AutoProcessor

    log.info("Loading processor for %s …", MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    log.info("Loading model %s (float16, device_map=auto) …", MODEL_ID)
    t0 = time.perf_counter()
    model = CsmForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    log.info("Model loaded in %.1f s", time.perf_counter() - t0)

    _model_cache["model"]     = model
    _model_cache["processor"] = processor
    _model_loaded = True
    return _model_cache


async def _get_model() -> dict:
    async with _model_lock:
        return await asyncio.get_event_loop().run_in_executor(None, _load_model_sync)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await _get_model()
        log.info("Startup complete — CSM-1B ready.")
    except Exception as exc:
        log.error("Failed to load model at startup: %s", exc)
    yield
    log.info("Shutting down …")
    _model_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CSM-1B TTS API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Audio helpers ─────────────────────────────────────────────────────────────
def _tensor_to_numpy(audio: torch.Tensor) -> np.ndarray:
    """Extract a (num_samples,) float32 array from model.generate() output.

    Official API (confirmed from model card):
        audio = model.generate(**inputs, output_audio=True)
        # audio is a (batch=1, num_samples) float32 tensor

    audio[0] → (num_samples,).
    """
    return audio[0].float().cpu().numpy()


def _numpy_to_wav(audio_np: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _silence_wav(duration_ms: int, sr: int = SAMPLE_RATE) -> bytes:
    n = int(sr * duration_ms / 1000)
    return _numpy_to_wav(np.zeros(n, dtype=np.float32), sr) if n > 0 else b""


def _load_ref_audio(path: str) -> np.ndarray:
    """Load WAV → mono float32 numpy at SAMPLE_RATE."""
    wav, orig_sr = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if orig_sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, orig_sr, SAMPLE_RATE)
    return wav.numpy().astype(np.float32)


def _validate_text(text: str):
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="text is required and must not be empty.")
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds {MAX_TEXT_LEN} chars ({len(text)}). "
                   "Increase MAX_TEXT_LEN or submit shorter input.",
        )


def _validate_speaker(sid: int):
    if not (MIN_SPEAKER_ID <= sid <= MAX_SPEAKER_ID):
        raise HTTPException(
            status_code=422,
            detail=f"speaker_id must be {MIN_SPEAKER_ID}–{MAX_SPEAKER_ID}, got {sid}.",
        )


def _split_text(text: str, max_chars: int = MODEL_CHUNK_TEXT_LEN) -> list[str]:
    """Split into sentence-sized chunks each ≤ max_chars."""
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
    return [c for c in chunks if c]


def _vram_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    props    = torch.cuda.get_device_properties(0)
    alloc    = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    total    = props.total_memory
    return {
        "available":    True,
        "device":       props.name,
        "total_mb":     round(total    / 1024**2, 1),
        "allocated_mb": round(alloc    / 1024**2, 1),
        "reserved_mb":  round(reserved / 1024**2, 1),
        "free_mb":      round((total - reserved) / 1024**2, 1),
    }


# ── CSM inference ─────────────────────────────────────────────────────────────

def _csm_generate(model, processor, text: str, speaker_id: int) -> np.ndarray:
    """Generate speech for one text chunk using a built-in speaker embedding.

    Conversation format (confirmed from official model card):
        conversation = [{"role": "0", "content": [{"type": "text", "text": "..."}]}]
        inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True)
        audio  = model.generate(**inputs, output_audio=True)  # → (1, num_samples) tensor
    """
    conversation = [{"role": str(speaker_id), "content": [{"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, return_dict=True
    ).to(model.device)

    with torch.no_grad():
        audio = model.generate(**inputs, output_audio=True)

    out = _tensor_to_numpy(audio)
    del inputs, audio
    return out


def _csm_generate_clone(
    model, processor,
    text: str, speaker_id: int,
    ref_audio_np: np.ndarray, ref_text: str,
) -> np.ndarray:
    """Generate speech matching the voice in ref_audio_np.

    CSM voice cloning works via conversation context: a turn containing both
    the reference transcript and the reference waveform is prepended so the
    model learns the target speaker's characteristics.

    Audio context format (confirmed from official model card):
        {"role": "0", "content": [
            {"type": "text",  "text": ref_text},
            {"type": "audio", "path": ref_audio_numpy_array},  # float32, 24 kHz, mono
        ]}
    """
    conversation = [
        {
            "role": str(speaker_id),
            "content": [
                {"type": "text",  "text": ref_text},
                {"type": "audio", "path": ref_audio_np},  # numpy array accepted directly
            ],
        },
        {
            "role": str(speaker_id),
            "content": [{"type": "text", "text": text}],
        },
    ]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, return_dict=True
    ).to(model.device)

    with torch.no_grad():
        audio = model.generate(**inputs, output_audio=True)

    out = _tensor_to_numpy(audio)
    del inputs, audio
    return out


# ── Async producer-consumer streaming pipeline ────────────────────────────────
_QUEUE_DONE  = object()
_QUEUE_ERROR = object()


async def _stream_audio(iter_fns: list, cleanup: Optional[callable] = None):
    """Async generator: GPU inference runs in a background thread while chunks
    are sent to the client concurrently via the async event loop.

    iter_fns: list of zero-arg callables, one per text chunk.  Each yields
    (audio_np, sr, None) exactly once.  The producer fills a bounded queue
    (PRE_ROLL_CHUNKS + 2 deep) so the GPU is always working ahead of the
    network, hiding per-sentence latency.
    """
    q            = _stdlib_queue.Queue(maxsize=max(PRE_ROLL_CHUNKS, 0) + 2)
    cancel_event = threading.Event()

    def _producer():
        try:
            for seg_idx, iter_fn in enumerate(iter_fns):
                if seg_idx > 0 and CHUNK_JOIN_SILENCE_MS > 0:
                    if cancel_event.is_set():
                        return
                    sil = _silence_wav(CHUNK_JOIN_SILENCE_MS)
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
            torch.cuda.empty_cache()

    loop         = asyncio.get_event_loop()
    producer_fut = loop.run_in_executor(None, _producer)
    t_start = time.perf_counter()
    first   = True
    ttfa_ms = None
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

        elapsed  = time.perf_counter() - t_start
        duration = total_samples / max(SAMPLE_RATE, 1)
        yield json.dumps({
            "chunk": "", "sample_rate": SAMPLE_RATE, "done": True,
            "rtf": round(elapsed / max(duration, 1e-6), 3),
            "ttfa_ms": ttfa_ms,
        }) + "\n"

    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        yield json.dumps({"error": "CUDA OOM — try shorter text.", "done": True}) + "\n"
    except Exception as exc:
        log.exception("Unexpected error in audio stream")
        yield json.dumps({"error": str(exc), "done": True}) + "\n"
    finally:
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
        "status":                "ok",
        "model":                 MODEL_ID,
        "model_loaded":          _model_loaded,
        "sample_rate":           SAMPLE_RATE,
        "max_text_length":       MAX_TEXT_LEN,
        "model_chunk_text_len":  MODEL_CHUNK_TEXT_LEN,
        "chunk_join_silence_ms": CHUNK_JOIN_SILENCE_MS,
        "pre_roll_chunks":       PRE_ROLL_CHUNKS,
        "speaker_id_range":      [MIN_SPEAKER_ID, MAX_SPEAKER_ID],
        "vram":                  _vram_info(),
    }


# ── /tts/generate  (non-streaming) ───────────────────────────────────────────
@app.post("/tts/generate")
async def tts_generate(text: str = Form(...), speaker_id: int = Form(0)):
    """Synthesise the full text and return one WAV file."""
    _validate_text(text); _validate_speaker(speaker_id)
    mc = await _get_model()
    model, processor = mc["model"], mc["processor"]
    chunks  = _split_text(text)
    silence = np.zeros(int(SAMPLE_RATE * CHUNK_JOIN_SILENCE_MS / 1000), dtype=np.float32)
    try:
        parts: list[np.ndarray] = []
        for idx, tc in enumerate(chunks):
            np_ = await asyncio.get_event_loop().run_in_executor(
                None, lambda t=tc: _csm_generate(model, processor, t, speaker_id))
            if idx > 0 and silence.size > 0:
                parts.append(silence)
            parts.append(np_)
        wav = _numpy_to_wav(np.concatenate(parts) if parts else np.array([], dtype=np.float32))
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="CUDA OOM — try shorter text.")
    finally:
        gc.collect(); torch.cuda.empty_cache()
    return Response(content=wav, media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=output.wav"})


# ── /tts/generate/stream ─────────────────────────────────────────────────────
@app.post("/tts/generate/stream")
async def tts_generate_stream(text: str = Form(...), speaker_id: int = Form(0)):
    """Stream speech sentence-by-sentence.
    Each NDJSON message: {"chunk": "<base64-wav>", "sample_rate": 24000, "done": false}
    Final message:       {"chunk": "", "done": true, "rtf": ..., "ttfa_ms": ...}
    """
    _validate_text(text); _validate_speaker(speaker_id)
    mc = await _get_model()
    model, processor = mc["model"], mc["processor"]
    chunks = _split_text(text)

    def _make_iter(tc: str):
        def _iter():
            yield _csm_generate(model, processor, tc, speaker_id), SAMPLE_RATE, None
        return _iter

    return StreamingResponse(
        _stream_audio([_make_iter(tc) for tc in chunks]),
        media_type="application/x-ndjson",
    )


# ── /tts/clone  (non-streaming) ──────────────────────────────────────────────
@app.post("/tts/clone")
async def tts_clone(
    text:       str        = Form(...),
    speaker_id: int        = Form(0),
    ref_text:   str        = Form(...),
    ref_audio:  UploadFile = File(...),
):
    """Synthesise text matching the voice in ref_audio.
    ref_audio: WAV file of the target voice (3+ seconds recommended).
    ref_text:  Exact words spoken in the reference recording.
    """
    _validate_text(text); _validate_speaker(speaker_id)
    if not ref_text.strip():
        raise HTTPException(status_code=422, detail="ref_text must not be empty.")

    audio_bytes = await ref_audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name

    try:
        ref_np = _load_ref_audio(tmp_path)
    except Exception as exc:
        pathlib.Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not read reference audio: {exc}")

    mc = await _get_model()
    model, processor = mc["model"], mc["processor"]
    chunks  = _split_text(text)
    silence = np.zeros(int(SAMPLE_RATE * CHUNK_JOIN_SILENCE_MS / 1000), dtype=np.float32)
    try:
        parts: list[np.ndarray] = []
        for idx, tc in enumerate(chunks):
            np_ = await asyncio.get_event_loop().run_in_executor(
                None, lambda t=tc: _csm_generate_clone(model, processor, t, speaker_id, ref_np, ref_text))
            if idx > 0 and silence.size > 0:
                parts.append(silence)
            parts.append(np_)
        wav = _numpy_to_wav(np.concatenate(parts) if parts else np.array([], dtype=np.float32))
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="CUDA OOM — try shorter text.")
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)
        gc.collect(); torch.cuda.empty_cache()
    return Response(content=wav, media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=output.wav"})


# ── /tts/clone/stream ────────────────────────────────────────────────────────
@app.post("/tts/clone/stream")
async def tts_clone_stream(
    text:       str        = Form(...),
    speaker_id: int        = Form(0),
    ref_text:   str        = Form(...),
    ref_audio:  UploadFile = File(...),
):
    _validate_text(text); _validate_speaker(speaker_id)
    if not ref_text.strip():
        raise HTTPException(status_code=422, detail="ref_text must not be empty.")

    audio_bytes = await ref_audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name

    try:
        ref_np = _load_ref_audio(tmp_path)
    except Exception as exc:
        pathlib.Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not read reference audio: {exc}")

    mc = await _get_model()
    model, processor = mc["model"], mc["processor"]
    chunks = _split_text(text)

    def _make_iter(tc: str):
        def _iter():
            yield _csm_generate_clone(model, processor, tc, speaker_id, ref_np, ref_text), SAMPLE_RATE, None
        return _iter

    return StreamingResponse(
        _stream_audio(
            [_make_iter(tc) for tc in chunks],
            cleanup=lambda: pathlib.Path(tmp_path).unlink(missing_ok=True),
        ),
        media_type="application/x-ndjson",
    )