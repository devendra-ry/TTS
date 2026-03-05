"""
tts/config.py
All environment-driven configuration, constants, and voice maps.
HuggingFace authentication is also performed here on import.
"""
import logging
import os

log = logging.getLogger("tts")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    """Read an integer from the environment with a safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default
    if value < min_value:
        log.warning(
            "%s=%s below minimum %s; using default %s", name, value, min_value, default
        )
        return default
    return value


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

MODEL_ID: str = os.getenv("MODEL_ID", "hexgrad/Kokoro-82M")
KOKORO_LANG_CODE: str = os.getenv("KOKORO_LANG_CODE", "a")

MAX_TEXT_LEN: int = _env_int("MAX_TEXT_LEN", 300_000, min_value=1_000)
MODEL_CHUNK_TEXT_LEN: int = _env_int("MODEL_CHUNK_TEXT_LEN", 220, min_value=50)
CHUNK_JOIN_SILENCE_MS: int = _env_int("CHUNK_JOIN_SILENCE_MS", 0, min_value=0)
PRE_ROLL_CHUNKS: int = _env_int("PRE_ROLL_CHUNKS", 4, min_value=0)
SAMPLE_RATE: int = _env_int("SAMPLE_RATE", 24_000, min_value=8_000)

# ---------------------------------------------------------------------------
# Speaker / voice maps
# ---------------------------------------------------------------------------

MIN_SPEAKER_ID: int = 0
MAX_SPEAKER_ID: int = 9

VOICE_BY_SPEAKER: dict[int, str] = {
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

VOICE_LABEL_BY_SPEAKER: dict[int, str] = {
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

# ---------------------------------------------------------------------------
# HuggingFace authentication (runs once on first import)
# ---------------------------------------------------------------------------

_HF_TOKEN: str = os.getenv("HF_TOKEN", "")

if _HF_TOKEN:
    try:
        from huggingface_hub import login as _hf_login

        _hf_login(token=_HF_TOKEN, add_to_git_credential=False)
        log.info("HuggingFace login OK.")
    except Exception as exc:
        log.warning("HuggingFace login failed: %s", exc)
else:
    log.warning(
        "HF_TOKEN is not set. Set HF_TOKEN=hf_... if model access requires auth."
    )
