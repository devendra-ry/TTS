# Kokoro TTS

A local Text-to-Speech API powered by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) with a React frontend. Supports both one-shot and real-time streaming audio generation.

---

## Features

- 🎙️ **10 preset voices** — US/UK male & female speakers
- ⚡ **Streaming mode** — audio playback begins before generation finishes
- 📥 **Download** — every generation produces a downloadable WAV file
- 🖥️ **GPU-accelerated** — uses CUDA when available, falls back to CPU
- 📊 **Metrics** — Time-to-first-audio (TTFA) and real-time factor (RTF) reported

---

## Project Structure

```
TTS/
├── main.py                        # FastAPI app entry point
├── tts/                           # Backend package
│   ├── config.py                  # Env vars, constants, voice maps, HF login
│   ├── audio_utils.py             # numpy/WAV helpers
│   ├── text_utils.py              # Text splitting & HTTP validation
│   ├── model.py                   # Model loading, cache, inference
│   ├── streaming.py               # Async streaming generator
│   └── routes.py                  # API endpoints
└── qwen-tts-ui/                   # React frontend (Vite)
    └── src/
        ├── api.js                 # All fetch calls
        ├── audioUtils.js          # Audio processing utilities
        ├── styles.js              # Shared style objects
        ├── KokoroTTS.jsx          # Root component
        ├── hooks/
        │   ├── useHealthPoller.js # Polls /health every 8s
        │   └── useAudioPlayer.js  # WebAudio scheduling
        └── components/
            ├── TopBar.jsx         # Logo + health indicator
            ├── VoiceSelector.jsx  # Speaker dropdown
            ├── StreamToggle.jsx   # Streaming on/off toggle
            └── AudioOutput.jsx    # Audio player + metrics
```

---

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Node.js 18+
- CUDA-capable GPU (optional but recommended)

---

## Setup

### 1. Clone & create environment

```bash
git clone https://github.com/devendra-ry/TTS.git
cd TTS
uv venv
```

### 2. Install Python dependencies

```bash
uv pip install fastapi uvicorn soundfile numpy torch kokoro huggingface_hub python-dotenv
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_token_here
```

> A HuggingFace token is required only if the model repository is gated.

### 4. Install frontend dependencies

```bash
cd qwen-tts-ui
npm install
```

---

## Running

### Backend

```bash
# From the project root
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd qwen-tts-ui
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Model status, config, VRAM info |
| `POST` | `/tts/generate` | Generate a complete WAV file |
| `POST` | `/tts/generate/stream` | Stream WAV chunks as NDJSON |

### POST `/tts/generate`

**Form fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | required | Input text (max 300,000 chars) |
| `speaker_id` | `int` | `0` | Voice ID (0–9) |

**Response:** `audio/wav` binary

### POST `/tts/generate/stream`

Same form fields. Returns `application/x-ndjson` — one JSON object per line:

```json
{"chunk": "<base64-wav>", "sample_rate": 24000, "done": false}
{"chunk": "", "sample_rate": 24000, "done": true, "ttfa_ms": 312, "rtf": 0.08}
```

---

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `hexgrad/Kokoro-82M` | HuggingFace model ID |
| `KOKORO_LANG_CODE` | `a` | Language code (`a` = English) |
| `MAX_TEXT_LEN` | `300000` | Maximum input text length |
| `MODEL_CHUNK_TEXT_LEN` | `220` | Characters per inference chunk |
| `CHUNK_JOIN_SILENCE_MS` | `0` | Silence between chunks (ms) |
| `PRE_ROLL_CHUNKS` | `4` | Queue depth before streaming starts |
| `SAMPLE_RATE` | `24000` | Output sample rate (Hz) |
| `HF_TOKEN` | — | HuggingFace auth token |

---

## Voices

| ID | Voice | Description |
|----|-------|-------------|
| 0 | af_bella | US Female (A-) |
| 1 | af_nicole | US Female (B-) |
| 2 | af_sarah | US Female (C+) |
| 3 | af_sky | US Female (C-) |
| 4 | am_adam | US Male (F+) |
| 5 | am_michael | US Male (C+) |
| 6 | bf_emma | UK Female (B-) |
| 7 | bf_isabella | UK Female (C) |
| 8 | bm_george | UK Male (C) |
| 9 | bm_lewis | UK Male (D+) |
