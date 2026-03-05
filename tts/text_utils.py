"""
tts/text_utils.py
Text preprocessing and request validation utilities.
"""
import re

from fastapi import HTTPException

from .config import MAX_TEXT_LEN, MIN_SPEAKER_ID, MAX_SPEAKER_ID, MODEL_CHUNK_TEXT_LEN


def validate_text(text: str) -> None:
    """Raise HTTP 422 if *text* is empty or over the configured length limit."""
    if not text or not text.strip():
        raise HTTPException(
            status_code=422, detail="text is required and must not be empty."
        )
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds {MAX_TEXT_LEN} chars ({len(text)}).",
        )


def validate_speaker(sid: int) -> None:
    """Raise HTTP 422 if *sid* is outside the supported speaker-ID range."""
    if not (MIN_SPEAKER_ID <= sid <= MAX_SPEAKER_ID):
        raise HTTPException(
            status_code=422,
            detail=f"speaker_id must be {MIN_SPEAKER_ID}-{MAX_SPEAKER_ID}, got {sid}.",
        )


def split_text(text: str, max_chars: int = MODEL_CHUNK_TEXT_LEN) -> list[str]:
    """
    Split *text* into sentence-aware chunks of at most *max_chars* each.

    Strategy:
    1. Collapse whitespace, split on sentence boundaries (. ! ? ; :).
    2. If a sentence fits in the running buffer, append it; otherwise flush
       the buffer and start a new one.
    3. Sentences longer than *max_chars* are split further by word, and
       words longer than *max_chars* are split character-by-character.

    Returns only chunks that contain at least one alphanumeric character
    and are at least 2 characters long.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    current = ""

    def _flush() -> None:
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
                    if len(word) <= max_chars:
                        word_buf = word
                    else:
                        word_buf = ""
                        for i in range(0, len(word), max_chars):
                            chunks.append(word[i : i + max_chars])
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
    return [
        c for c in chunks if c and len(c.strip()) >= 2 and any(ch.isalnum() for ch in c)
    ]
