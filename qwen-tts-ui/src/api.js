/**
 * api.js
 * All network calls to the TTS backend in one place.
 */

function getApiBase() {
    const envUrl = import.meta.env.VITE_API_BASE_URL;
    if (envUrl) return envUrl.replace(/\/$/, "");
    if (typeof window !== "undefined") {
        return `${window.location.protocol}//${window.location.hostname}:8000`;
    }
    return "http://localhost:8000";
}

export const API = getApiBase();

/**
 * Fetch the backend health/config object.
 * Returns null on network failure instead of throwing.
 */
export async function fetchHealth() {
    const res = await fetch(`${API}/health`);
    if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`);
    return res.json();
}

/**
 * Generate a complete WAV file (non-streaming).
 * @returns {Promise<Blob>} WAV blob
 */
export async function generateTTS(text, speakerId, signal) {
    const fd = new FormData();
    fd.append("text", text);
    fd.append("speaker_id", speakerId);

    const res = await fetch(`${API}/tts/generate`, {
        method: "POST",
        body: fd,
        signal,
    });

    if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.detail || `HTTP ${res.status}`);
    }

    return res.blob();
}

/**
 * Start a streaming TTS request.
 * @returns {Promise<Response>} Raw fetch Response for the caller to consume as NDJSON.
 */
export async function generateStream(text, speakerId, signal) {
    const fd = new FormData();
    fd.append("text", text);
    fd.append("speaker_id", speakerId);

    const res = await fetch(`${API}/tts/generate/stream`, {
        method: "POST",
        body: fd,
        signal,
    });

    if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.detail || `HTTP ${res.status}`);
    }

    return res;
}
