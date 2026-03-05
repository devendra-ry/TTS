"""
tts/streaming.py
Async generator that streams WAV chunks to the HTTP response.
A background thread (producer) runs blocking inference and pushes chunks
into a bounded queue.  The async consumer yields NDJSON lines to the client.
"""
import asyncio
import base64
import gc
import json
import queue as stdlib_queue
import threading
import time
from typing import Callable, Optional

import torch

from .audio_utils import numpy_to_wav, silence_wav
from .config import CHUNK_JOIN_SILENCE_MS, PRE_ROLL_CHUNKS, SAMPLE_RATE

# Sentinel objects used to signal done / error across the queue boundary.
_QUEUE_DONE = object()
_QUEUE_ERROR = object()


async def stream_audio(
    iter_fns: list[Callable], cleanup: Optional[Callable] = None
):
    """
    Yield NDJSON lines for a streaming TTS response.

    Each *iter_fn* in *iter_fns* must be a zero-argument callable that
    returns an iterable of ``(audio_np, sample_rate, _)`` tuples.
    Optional inter-segment silence is inserted between segments when
    ``CHUNK_JOIN_SILENCE_MS > 0``.

    Yields:
        str — JSON line ending with ``\\n``.  The last line has ``done: true``.
    """
    q: stdlib_queue.Queue = stdlib_queue.Queue(maxsize=max(PRE_ROLL_CHUNKS, 0) + 2)
    cancel_event = threading.Event()

    def _producer() -> None:
        try:
            for seg_idx, iter_fn in enumerate(iter_fns):
                if seg_idx > 0 and CHUNK_JOIN_SILENCE_MS > 0:
                    if cancel_event.is_set():
                        return
                    sil = silence_wav(CHUNK_JOIN_SILENCE_MS, SAMPLE_RATE)
                    if sil:
                        q.put(("wav", sil, SAMPLE_RATE, 0))

                for audio_np, chunk_sr, _ in iter_fn():
                    if cancel_event.is_set():
                        return
                    q.put(("wav", numpy_to_wav(audio_np, chunk_sr), chunk_sr, len(audio_np)))

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
        import logging
        logging.getLogger("tts").exception("Unexpected error in audio stream")
        yield json.dumps({"error": str(exc), "done": True}) + "\n"
    finally:
        cancel_event.set()
        # Drain the queue so the producer thread can exit cleanly.
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
