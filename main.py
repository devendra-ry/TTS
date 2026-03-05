"""
main.py
Application entry point.

Responsibilities:
  - Configure logging
  - Create the FastAPI app with a lifespan handler
  - Register CORS middleware
  - Mount the TTS router

Run with:
    uvicorn main:app --reload --port 8000
"""
import gc
import logging

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tts.model import get_model, _cache
from tts.routes import router
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tts")


# ---------------------------------------------------------------------------
# Lifespan — warm up the model at startup, clean up on shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await get_model()
        log.info("Startup complete - Kokoro ready.")
    except Exception as exc:
        log.error("Failed to load model at startup: %s", exc)
    yield
    log.info("Shutting down ...")
    _cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(title="Kokoro TTS API", version="1.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
