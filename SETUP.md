# SETUP — Qwen3-TTS Local App

Local TTS stack: **faster-qwen3-tts** · **FastAPI** · **React**  
Target GPU: NVIDIA GTX 1650 (4 GB VRAM)

---

## Prerequisites

| Tool | Minimum version | Notes |
|------|----------------|-------|
| Python | 3.10+ | 3.11 recommended |
| Node.js | 18+ | For the React dev server |
| CUDA Toolkit | 12.x | Must match the PyTorch build |
| `uv` | any | `pip install uv` |

Verify your CUDA version before continuing:
```bash
nvidia-smi
# Look for "CUDA Version: 12.x" in the top-right corner
```

---

## 1 — Python backend

### 1.1 Create the virtual environment

```bash
uv venv --python 3.11
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 1.2 Install PyTorch with CUDA 12.8

```bash
uv pip install torch torchaudio --torch-backend=cu128
```

> If your CUDA version is older (e.g. 12.1) replace `cu128` with `cu121`.  
> Run `nvcc --version` to check.

### 1.3 Install remaining dependencies

```bash
uv pip install faster-qwen3-tts soundfile
uv pip install fastapi "uvicorn[standard]" python-multipart httpx
```

`faster-qwen3-tts` automatically installs `qwen-tts` as a dependency.  
**No bitsandbytes. No transformers from source. No vLLM.**

### 1.4 Place the backend file

Put `tts_backend.py` in the root of your project folder.

### 1.5 Run the backend

```bash
uvicorn tts_backend:app --host 0.0.0.0 --port 8000
```

Expected startup output:
```
INFO:     Loading Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice …
INFO:     Loaded in 8.2s
INFO:     Startup complete — CustomVoice model ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The first synthesis request will trigger CUDA graph capture (5–10 s warm-up).  
Subsequent requests are significantly faster.

**Health check:**
```bash
curl http://localhost:8000/health
```

---

## 2 — React frontend

### 2.1 Scaffold the project

```bash
npm create vite@latest qwen-tts-ui -- --template react
cd qwen-tts-ui
```

### 2.2 Replace the default component

Copy `qwen-tts.jsx` into `src/` and update `src/main.jsx`:

```jsx
// src/main.jsx
import React from "react";
import ReactDOM from "react-dom/client";
import QwenTTS from "./qwen-tts.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <QwenTTS />
  </React.StrictMode>
);
```

Remove `src/App.jsx`, `src/App.css`, `src/index.css` (optional cleanup).

### 2.3 Install & run

```bash
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 3 — Project layout

```
your-project/
├── tts_backend.py        ← FastAPI server
├── qwen-tts-ui/
│   ├── src/
│   │   ├── main.jsx
│   │   └── qwen-tts.jsx  ← React component
│   ├── index.html
│   └── package.json
└── .venv/                ← Python virtual environment
```

---

## 4 — API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Model status, VRAM usage, speaker list |
| `POST` | `/tts/custom` | Preset speaker synthesis (multipart form) |
| `POST` | `/tts/custom/stream` | Streaming preset speaker (NDJSON) |
| `POST` | `/tts/clone` | Voice cloning — upload ref WAV (multipart form) |
| `POST` | `/tts/clone/stream` | Streaming voice clone (NDJSON) |

### `/tts/custom` form fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `text` | string | required | Max `MAX_TEXT_LEN` characters (default 300000) |
| `language` | string | `English` | English, Chinese, French, German, Japanese, Korean, … |
| `speaker` | string | `aiden` | serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan |
| `instruct` | string | — | Optional style instruction, e.g. "Speak warmly" |

### `/tts/clone` form fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `text` | string | required | Max `MAX_TEXT_LEN` characters (default 300000) |
| `language` | string | `English` | |
| `ref_audio` | file | required | WAV file, minimum 3 seconds |
| `ref_text` | string | required | Exact transcript of the reference audio |

---

## 5 — VRAM notes (GTX 1650)

- **CustomVoice 0.6B** uses ~1.5 GB VRAM in fp16 — fits comfortably.
- **Base 0.6B (voice clone)** uses similar VRAM; the backend evicts the other model before loading.
- **Do NOT use 1.7B models** — they exceed 4 GB and will OOM.
- Long requests are automatically split into smaller model chunks (`MODEL_CHUNK_TEXT_LEN`, default 450 chars).
- Use `MAX_TEXT_LEN` to control overall request size (default 300000 chars).
- Use `CHUNK_JOIN_SILENCE_MS` to control pause between generated chunks (default 150 ms).
- After each request the backend calls `torch.cuda.empty_cache()` to return fragmented memory.

---

## 6 — Troubleshooting

### CUDA OOM on first request
Reduce `MODEL_CHUNK_TEXT_LEN` (for example 450 -> 300) and restart the server.

### "Loaded in 8s" but no audio
CUDA graph capture is running. Wait for the warm-up to finish, then retry. The second request will be fast.

### `bf16` errors / `RuntimeError: Expected all tensors to be on the same device`
Do not set `torch_dtype` manually. `faster-qwen3-tts` manages dtype internally (fp16 on 1650). Let it handle device placement.

### Frontend shows "OFFLINE"
Ensure the backend is running on port 8000 and CORS is not blocked. Check:
```bash
curl http://localhost:8000/health
```

### Audio plays choppy in streaming mode
Increase `chunk_size` in the backend streaming calls (e.g. `chunk_size=12`). Larger chunks = more latency but smoother playback.

---

## 7 — Model downloads

Models are fetched from Hugging Face on first use and cached in `~/.cache/huggingface/hub/`.

| Model | Size on disk | First load time |
|-------|-------------|-----------------|
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | ~1.2 GB | ~30s (download) + ~8s (load) |
| `Qwen3-TTS-12Hz-0.6B-Base` | ~1.2 GB | same |

Set `HF_HOME` to change the cache location:
```bash
export HF_HOME=/path/to/your/cache
```
