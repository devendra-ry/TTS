"""
tts/model.py
Model lifecycle: loading, caching, and synchronous inference.
The cache is process-global; concurrent load requests are serialised
with an asyncio.Lock so the heavy synchronous work runs only once.
"""
import asyncio
import logging
import time

import numpy as np
import torch

from .audio_utils import result_to_audio_np
from .config import KOKORO_LANG_CODE, MODEL_ID, SAMPLE_RATE, VOICE_BY_SPEAKER

log = logging.getLogger("tts")

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_cache: dict = {}
_loaded: bool = False
_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_loaded() -> bool:
    return _loaded


def load_model_sync() -> dict:
    """
    Load the Kokoro pipeline synchronously.
    Idempotent — returns the cached dict on subsequent calls.
    """
    global _loaded

    if _loaded and _cache:
        return _cache

    from kokoro import KPipeline  # imported lazily to keep startup fast on CPU

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(
        "Loading Kokoro pipeline for %s (lang=%s, device=%s) ...",
        MODEL_ID,
        KOKORO_LANG_CODE,
        device,
    )
    t0 = time.perf_counter()
    pipeline = KPipeline(lang_code=KOKORO_LANG_CODE, repo_id=MODEL_ID, device=device)
    log.info("Model loaded in %.1f s", time.perf_counter() - t0)

    _cache["pipeline"] = pipeline
    _loaded = True
    return _cache


async def get_model() -> dict:
    """
    Async wrapper around :func:`load_model_sync`.
    Guarantees only one concurrent load attempt via an asyncio.Lock.
    """
    async with _lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, load_model_sync)


def kokoro_generate(
    pipeline, text: str, speaker_id: int
) -> tuple[np.ndarray, int]:
    """
    Run a single Kokoro inference call and return (audio_np, sample_rate).
    Raises ValueError for degenerate inputs or empty outputs.
    """
    text = text.strip()
    if not text or not any(ch.isalnum() for ch in text):
        raise ValueError(f"Skipping degenerate chunk: {text!r}")

    voice = VOICE_BY_SPEAKER.get(speaker_id, VOICE_BY_SPEAKER[0])
    generator = pipeline(text, voice=voice, speed=1.0, split_pattern=r"\n+")
    chunks = [result_to_audio_np(result) for result in generator]
    if not chunks:
        raise ValueError("Kokoro returned no audio chunks.")

    return np.concatenate(chunks), SAMPLE_RATE
