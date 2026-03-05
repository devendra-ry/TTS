"""
tts/audio_utils.py
Pure audio-processing utilities — no FastAPI or model dependencies.
"""
import io

import numpy as np
import soundfile as sf
import torch

from .config import SAMPLE_RATE


def numpy_to_wav(audio_np: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Encode a float32 numpy array as a PCM-16 WAV byte string."""
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def silence_wav(duration_ms: int, sr: int = SAMPLE_RATE) -> bytes:
    """Return a WAV of silence with the given duration (ms), or b'' if duration is 0."""
    n = int(sr * duration_ms / 1000)
    return numpy_to_wav(np.zeros(n, dtype=np.float32), sr) if n > 0 else b""


def result_to_audio_np(result) -> np.ndarray:
    """
    Extract a normalised float32 numpy array from a Kokoro pipeline result.
    Supports both attribute-style (result.audio) and tuple/list-style results.
    """
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


def vram_info() -> dict:
    """Return a summary of current CUDA VRAM usage, or {available: False} on CPU."""
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
