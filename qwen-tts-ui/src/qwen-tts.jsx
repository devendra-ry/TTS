import { useState, useRef, useEffect, useCallback } from "react";

/* ─── Constants ─────────────────────────────────────────────────────────── */
const API = "http://localhost:8000";
const SPEAKERS = ["serena","vivian","uncle_fu","ryan","aiden","ono_anna","sohee","eric","dylan"];
const LANGUAGES = ["English","Chinese","French","German","Japanese","Korean","Spanish","Portuguese","Arabic","Russian"];
const DEFAULT_MAX_TEXT = 300000;

/* ─── Audio helpers ──────────────────────────────────────────────────────── */

/*
 * BUG-1 FIX — replaced wavBytesToFloat32() with decodeChunk()
 *
 * The old function hardcoded a 44-byte WAV header skip and manually
 * converted int16 → float32.  Two problems:
 *
 *   a) soundfile may write variable-length headers (e.g. with a LIST
 *      metadata chunk), making the 44-byte offset wrong.  Reading header
 *      bytes as audio samples produces a click/pop at the start of every
 *      chunk and slight timing drift over many chunks.
 *
 *   b) Manual decoding bypasses the browser's native resampler.  If the
 *      AudioContext runs at a device rate different from chunk_sr (e.g.
 *      ctx at 48 000 Hz, buffer at 24 000 Hz), WebAudio's internal
 *      resampler handles it correctly when given a properly-declared
 *      AudioBuffer — but only if the buffer's sampleRate matches the
 *      actual data.  The old code passed msg.sample_rate which was
 *      correct, so this was not the bug here; however, using
 *      decodeAudioData is still more robust because it reads the sample
 *      rate from the WAV header itself rather than trusting the NDJSON
 *      metadata field.
 *
 * Fix: use ctx.decodeAudioData(arrayBuffer).  The browser's native
 * decoder handles any header length, any bit depth, and any sample rate
 * correctly and returns a ready-to-use AudioBuffer.
 */
function base64ToArrayBuffer(b64) {
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) view[i] = raw.charCodeAt(i);
  return buf;
}

/*
 * Reassemble all received chunk WAV blobs into a single WAV for the
 * download / <audio> element.  Strips each per-chunk header and writes
 * one combined header with the correct total data size.
 */
function wavChunksToWavBlob(chunksB64, sampleRate = 24000) {
  const toBytes = (b64) => {
    const raw = atob(b64);
    const out = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
    return out;
  };

  // Read the data-chunk offset from each WAV header rather than
  // hardcoding 44, so variable-length headers are handled correctly.
  const getPcmOffset = (bytes) => {
    const view = new DataView(bytes.buffer);
    let pos = 12; // skip "RIFF????WAVE"
    while (pos + 8 <= bytes.length) {
      const id   = String.fromCharCode(...bytes.subarray(pos, pos + 4));
      const size = view.getUint32(pos + 4, true);
      if (id === "data") return pos + 8;
      pos += 8 + size;
    }
    return 44; // fallback
  };

  const allBytes = chunksB64.map(toBytes);
  const pcmChunks = allBytes
    .filter(b => b.length > 8)
    .map(b => b.subarray(getPcmOffset(b)));

  if (!pcmChunks.length) return null;

  const numChannels    = 1;
  const bytesPerSample = 2;
  const dataSize = pcmChunks.reduce((s, c) => s + c.length, 0);
  const out  = new Uint8Array(44 + dataSize);
  const view = new DataView(out.buffer);

  const ws = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
  ws(0,  "RIFF"); view.setUint32(4, 36 + dataSize, true);
  ws(8,  "WAVE");
  ws(12, "fmt "); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, 16, true);
  ws(36, "data"); view.setUint32(40, dataSize, true);

  let offset = 44;
  pcmChunks.forEach(chunk => { out.set(chunk, offset); offset += chunk.length; });
  return new Blob([out], { type: "audio/wav" });
}

/* ─── Inline styles ──────────────────────────────────────────────────────── */
const css = {
  root: {
    minHeight: "100vh",
    background: "#0c0c0f",
    color: "#c8d0e8",
    fontFamily: "'Fira Mono', 'JetBrains Mono', 'Courier New', monospace",
    padding: "0",
    margin: "0",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },
  topBar: {
    width: "100%",
    borderBottom: "1px solid #1e2030",
    padding: "14px 32px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    background: "#0e0e14",
    boxSizing: "border-box",
  },
  logo: {
    fontSize: "13px",
    letterSpacing: "0.18em",
    color: "#7b9fff",
    textTransform: "uppercase",
    fontWeight: 700,
  },
  healthDot: (ok) => ({
    display: "inline-flex",
    alignItems: "center",
    gap: "6px",
    fontSize: "11px",
    letterSpacing: "0.1em",
    color: ok ? "#5af07e" : "#ff6b6b",
  }),
  dot: (ok) => ({
    width: "7px",
    height: "7px",
    borderRadius: "50%",
    background: ok ? "#5af07e" : "#ff6b6b",
    boxShadow: ok ? "0 0 6px #5af07e88" : "0 0 6px #ff6b6b88",
    animation: ok ? "pulse 2s infinite" : "none",
  }),
  main: {
    width: "100%",
    maxWidth: "760px",
    padding: "40px 24px 80px",
    boxSizing: "border-box",
    display: "flex",
    flexDirection: "column",
    gap: "28px",
  },
  heading: { fontSize: "22px", fontWeight: 700, color: "#e4e8f8", letterSpacing: "0.05em", marginBottom: "-10px" },
  subheading: { fontSize: "11px", color: "#4a5070", letterSpacing: "0.15em", textTransform: "uppercase" },
  panel: {
    background: "#111118", border: "1px solid #1e2035", borderRadius: "8px",
    padding: "20px 22px", display: "flex", flexDirection: "column", gap: "16px",
  },
  label: {
    fontSize: "11px", letterSpacing: "0.12em", textTransform: "uppercase",
    color: "#4a5070", marginBottom: "5px", display: "block",
  },
  textarea: {
    width: "100%", minHeight: "120px", background: "#0c0c0f",
    border: "1px solid #1e2035", borderRadius: "5px", color: "#c8d0e8",
    fontFamily: "inherit", fontSize: "14px", lineHeight: "1.6",
    padding: "12px 14px", resize: "vertical", outline: "none",
    boxSizing: "border-box", transition: "border-color 0.15s",
  },
  textareaFocus: { borderColor: "#3b4a7a" },
  row: { display: "flex", gap: "12px", flexWrap: "wrap" },
  col: { display: "flex", flexDirection: "column", flex: 1, minWidth: "140px" },
  select: {
    background: "#0c0c0f", border: "1px solid #1e2035", borderRadius: "5px",
    color: "#c8d0e8", fontFamily: "inherit", fontSize: "13px", padding: "9px 12px",
    outline: "none", cursor: "pointer", appearance: "none",
    backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%234a5070'/%3E%3C/svg%3E")`,
    backgroundRepeat: "no-repeat", backgroundPosition: "right 12px center", paddingRight: "30px",
  },
  input: {
    background: "#0c0c0f", border: "1px solid #1e2035", borderRadius: "5px",
    color: "#c8d0e8", fontFamily: "inherit", fontSize: "13px", padding: "9px 12px",
    outline: "none", width: "100%", boxSizing: "border-box",
  },
  modeToggle: {
    display: "flex", background: "#0c0c0f", border: "1px solid #1e2035",
    borderRadius: "6px", overflow: "hidden", width: "fit-content",
  },
  modeBtn: (active) => ({
    padding: "8px 22px", fontSize: "12px", letterSpacing: "0.1em",
    textTransform: "uppercase", border: "none", cursor: "pointer",
    fontFamily: "inherit", background: active ? "#7b9fff" : "transparent",
    color: active ? "#0c0c0f" : "#4a5070", fontWeight: active ? 700 : 400, transition: "all 0.15s",
  }),
  streamToggle: { display: "flex", alignItems: "center", gap: "10px", cursor: "pointer", userSelect: "none" },
  toggleTrack: (on) => ({
    width: "36px", height: "20px", borderRadius: "10px",
    background: on ? "#7b9fff44" : "#1e2035", border: `1px solid ${on ? "#7b9fff" : "#2a2f50"}`,
    position: "relative", transition: "all 0.2s", cursor: "pointer", flexShrink: 0,
  }),
  toggleThumb: (on) => ({
    width: "14px", height: "14px", borderRadius: "50%",
    background: on ? "#7b9fff" : "#3a4060", position: "absolute",
    top: "2px", left: on ? "18px" : "2px", transition: "left 0.2s",
    boxShadow: on ? "0 0 6px #7b9fff88" : "none",
  }),
  toggleLabel: { fontSize: "12px", letterSpacing: "0.1em", textTransform: "uppercase", color: "#4a5070" },
  actionRow: { display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" },
  btn: (variant = "primary", disabled = false) => ({
    padding: "10px 24px", fontFamily: "inherit", fontSize: "12px",
    letterSpacing: "0.12em", textTransform: "uppercase", fontWeight: 700,
    border: "none", borderRadius: "5px", cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.4 : 1, transition: "all 0.15s",
    ...(variant === "primary"
      ? { background: "#7b9fff", color: "#0c0c0f" }
      : variant === "danger"
      ? { background: "transparent", color: "#ff6b6b", border: "1px solid #ff6b6b44" }
      : { background: "transparent", color: "#7b9fff", border: "1px solid #7b9fff44" }),
  }),
  charCount: (over) => ({ fontSize: "11px", color: over ? "#ff6b6b" : "#2a3050", textAlign: "right", marginTop: "-8px" }),
  audioWrap: {
    background: "#0c0c0f", border: "1px solid #1e2035", borderRadius: "5px",
    padding: "12px 14px", display: "flex", flexDirection: "column", gap: "10px",
  },
  metricsRow: { display: "flex", gap: "20px", flexWrap: "wrap" },
  metric: { fontSize: "11px", color: "#4a5070", letterSpacing: "0.08em" },
  metricVal: { color: "#7b9fff", fontWeight: 700 },
  statusBar: { fontSize: "12px", color: "#4a5070", letterSpacing: "0.08em", minHeight: "18px", padding: "2px 0" },
  errorMsg: {
    fontSize: "12px", color: "#ff6b6b", background: "#200f0f",
    border: "1px solid #3a1515", borderRadius: "5px", padding: "10px 14px",
  },
  fileZone: {
    border: "1px dashed #2a2f50", borderRadius: "5px", padding: "18px",
    textAlign: "center", fontSize: "12px", color: "#4a5070", cursor: "pointer",
    transition: "border-color 0.15s", background: "#0c0c0f",
  },
};

/* ─── Component ──────────────────────────────────────────────────────────── */
export default function QwenTTS() {
  const [mode, setMode]           = useState("custom");
  const [text, setText]           = useState("");
  const [language, setLanguage]   = useState("English");
  const [speaker, setSpeaker]     = useState("aiden");
  const [instruct, setInstruct]   = useState("");
  const [streaming, setStreaming] = useState(false);

  const [refAudio, setRefAudio]   = useState(null);
  const [refText, setRefText]     = useState("");

  const [status, setStatus]       = useState("");
  const [error, setError]         = useState("");
  const [loading, setLoading]     = useState(false);
  const [health, setHealth]       = useState(null);

  const [audioUrl, setAudioUrl]   = useState(null);
  const [metrics, setMetrics]     = useState(null);
  const audioRef                  = useRef(null);
  const abortRef                  = useRef(null);

  /*
   * BUG-3 FIX — AudioContext refs
   *
   * audioCtxRef     : the single AudioContext for the whole stream.
   * nextPlayTimeRef : absolute AudioContext time at which the next
   *                   chunk should start.  Incremented by buf.duration
   *                   after every scheduled chunk so there is NEVER a
   *                   gap between them.
   * startedRef      : false until the first chunk is scheduled; used
   *                   to set the initial nextPlayTime only once.
   */
  const audioCtxRef     = useRef(null);
  const nextPlayTimeRef = useRef(0);
  const startedRef      = useRef(false);

  const [taFocused, setTaFocused] = useState(false);
  const maxText = health?.max_text_length ?? DEFAULT_MAX_TEXT;

  /* Poll health */
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API}/health`);
        setHealth(await r.json());
      } catch {
        setHealth(null);
      }
    };
    check();
    const id = setInterval(check, 8000);
    return () => clearInterval(id);
  }, []);

  /* Cleanup AudioContext on unmount */
  useEffect(() => () => { audioCtxRef.current?.close(); }, []);

  const clearState = useCallback(() => {
    setError("");
    setStatus("");
    setMetrics(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    nextPlayTimeRef.current = 0;
    startedRef.current      = false;
  }, [audioUrl]);

  /* ── Non-streaming ─────────────────────────────────────────────────── */
  const runNonStream = async (formData, endpoint) => {
    setStatus("Generating…");
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const r = await fetch(`${API}${endpoint}`, { method: "POST", body: formData, signal: ctrl.signal });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.detail || `HTTP ${r.status}`);
    }
    const blob = await r.blob();
    setAudioUrl(URL.createObjectURL(blob));
    setStatus("Done.");
  };

  /* ── Streaming ─────────────────────────────────────────────────────── */
  const runStream = async (formData, endpoint) => {
    setStatus("Connecting…");
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    /*
     * BUG-3 FIX — latencyHint: "interactive"
     *
     * The old code used latencyHint: "playback", which instructs the
     * browser to use a large output buffer (4096–8192 samples, ~85–170 ms
     * at 48 kHz) to maximise throughput.  For streaming TTS playback this
     * means chunk boundaries that land inside a hardware buffer period are
     * played as one block → irregular perceived durations → jitter.
     *
     * "interactive" minimises the output buffer size, giving precise
     * chunk-boundary scheduling at the cost of slightly higher CPU usage —
     * the right trade-off for real-time TTS.
     */
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    audioCtxRef.current = new AudioCtx({ latencyHint: "interactive" });

    /*
     * BUG-5 FIX — AudioContext autoplay policy
     *
     * The old code called resume().catch(() => {}) which silently swallowed
     * any autoplay-policy rejection, leaving the context suspended and
     * causing complete silence.  We now surface the error so the user
     * knows they need to interact with the page first.
     */
    if (audioCtxRef.current.state === "suspended") {
      await audioCtxRef.current.resume();
      // If still suspended after resume(), the browser blocked autoplay.
      if (audioCtxRef.current.state === "suspended") {
        throw new Error("Browser blocked audio autoplay. Click anywhere on the page and try again.");
      }
    }

    nextPlayTimeRef.current = 0;
    startedRef.current      = false;

    const allChunks = [];
    let outputSampleRate = 24000;

    const r = await fetch(`${API}${endpoint}`, { method: "POST", body: formData, signal: ctrl.signal });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.detail || `HTTP ${r.status}`);
    }

    const reader  = r.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = "";

    setStatus("Streaming…");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (msg.error) throw new Error(msg.error);

        if (msg.done) {
          setMetrics({ ttfa_ms: msg.ttfa_ms, rtf: msg.rtf });
          setStatus("Done.");
          continue;
        }

        if (!msg.chunk) continue;

        allChunks.push(msg.chunk);
        outputSampleRate = msg.sample_rate || outputSampleRate;

        /*
         * BUG-1 FIX + BUG-2 FIX — use decodeAudioData for chunk playback
         *
         * OLD CODE problems:
         *   1. wavBytesToFloat32 skipped a hardcoded 44 bytes for the WAV
         *      header.  If soundfile writes a longer header (e.g. with a
         *      LIST metadata chunk) this reads header bytes as audio data,
         *      producing a click/pop at the start of every chunk.
         *   2. After decoding, the scheduling used a fixed minLead guard:
         *
         *        if (nextPlayTimeRef.current < ctx.currentTime + 0.12) {
         *          nextPlayTimeRef.current = ctx.currentTime + 0.12;
         *        }
         *
         *      This guard was meant to prevent scheduling in the past, but
         *      it has a fatal side-effect: any time a chunk arrives more
         *      than 120 ms later than expected (e.g. during GPU prefill or
         *      a slow inference frame), nextPlayTime is bumped FORWARD by
         *      120 ms from the current moment, creating a 120 ms silence
         *      gap before that chunk.  With many such events across a long
         *      stream → continuous stutter.
         *
         * NEW CODE:
         *   1. ctx.decodeAudioData() handles any WAV header length and any
         *      sample rate correctly — no manual header parsing needed.
         *
         *   2. Scheduling:
         *      - On the FIRST chunk, set nextPlayTime = ctx.currentTime +
         *        a small fixed lookahead (50 ms).  This single lookahead is
         *        enough to absorb decode latency; after the first chunk the
         *        buffer is ahead of the cursor.
         *      - On every chunk: start = nextPlayTime (or currentTime+50ms
         *        if somehow we fell behind), then advance nextPlayTime by
         *        exactly buf.duration.  Because nextPlayTime is incremented
         *        by the actual decoded duration, chunks tile perfectly with
         *        ZERO gap between them regardless of when they arrive over
         *        the network.
         */
        const ctx = audioCtxRef.current;
        if (!ctx) continue;    // context was closed (stop was pressed)

        try {
          const arrayBuf   = base64ToArrayBuffer(msg.chunk);
          const audioBuf   = await ctx.decodeAudioData(arrayBuf);

          // Initialise cursor on first chunk, or catch up if we fell behind.
          const LOOKAHEAD = 0.05; // 50 ms — enough for one decode round-trip
          if (!startedRef.current) {
            nextPlayTimeRef.current = ctx.currentTime + LOOKAHEAD;
            startedRef.current      = true;
          } else if (nextPlayTimeRef.current < ctx.currentTime) {
            // We are behind (extremely late chunk). Catch up without a gap.
            nextPlayTimeRef.current = ctx.currentTime + LOOKAHEAD;
          }

          const src = ctx.createBufferSource();
          src.buffer = audioBuf;
          src.connect(ctx.destination);
          src.start(nextPlayTimeRef.current);

          // Advance cursor by the EXACT decoded duration.
          // This is the key to gapless playback — no minLead bump here.
          nextPlayTimeRef.current += audioBuf.duration;
        } catch (decodeErr) {
          // A single bad chunk should not kill the stream.
          console.warn("decodeAudioData failed for chunk:", decodeErr);
        }
      }
    }

    if (allChunks.length) {
      const wavBlob = wavChunksToWavBlob(allChunks, outputSampleRate);
      if (wavBlob) setAudioUrl(URL.createObjectURL(wavBlob));
    }
  };

  /* ── Generate handler ──────────────────────────────────────────────── */
  const generate = async () => {
    clearState();
    setLoading(true);

    const fd = new FormData();
    fd.append("text", text.trim());
    fd.append("language", language);

    try {
      if (mode === "custom") {
        fd.append("speaker", speaker);
        if (instruct.trim()) fd.append("instruct", instruct.trim());
        await (streaming ? runStream(fd, "/tts/custom/stream") : runNonStream(fd, "/tts/custom"));
      } else {
        if (!refAudio)        throw new Error("Please upload a reference WAV file.");
        if (!refText.trim())  throw new Error("Please provide the reference audio transcript.");
        fd.append("ref_audio", refAudio);
        fd.append("ref_text",  refText.trim());
        await (streaming ? runStream(fd, "/tts/clone/stream") : runNonStream(fd, "/tts/clone"));
      }
    } catch (e) {
      if (e.name === "AbortError") {
        setStatus("Stopped.");
      } else {
        setError(e.message || "Unknown error");
        setStatus("");
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  };

  const stop = () => {
    abortRef.current?.abort();
    audioCtxRef.current?.suspend();
  };

  const charOver    = text.length > maxText;
  const canGenerate = !loading && text.trim().length > 0 && !charOver &&
    (mode === "custom" || (refAudio && refText.trim()));
  const showOutput  = !!audioUrl;

  return (
    <div style={css.root}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fira+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; background: #0c0c0f; }
        select option { background: #111118; }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0c0c0f; }
        ::-webkit-scrollbar-thumb { background: #1e2035; border-radius: 3px; }
        audio { width: 100%; accent-color: #7b9fff; }
        audio::-webkit-media-controls-panel { background: #111118; }
      `}</style>

      {/* Top Bar */}
      <div style={css.topBar}>
        <span style={css.logo}>◈ Qwen3-TTS</span>
        <span style={css.healthDot(!!health)}>
          <span style={css.dot(!!health)} />
          {health ? `READY · ${health.vram?.device ?? "GPU"}` : "OFFLINE"}
        </span>
      </div>

      {/* Main */}
      <div style={css.main}>
        <div>
          <div style={css.heading}>Text-to-Speech</div>
          <div style={{ ...css.subheading, marginTop: "6px" }}>
            Local inference · GTX 1650 · {maxText} char request limit
          </div>
        </div>

        {/* Mode toggle */}
        <div>
          <div style={css.modeToggle}>
            <button style={css.modeBtn(mode === "custom")} onClick={() => setMode("custom")}>Custom Voice</button>
            <button style={css.modeBtn(mode === "clone")}  onClick={() => setMode("clone")}>Voice Clone</button>
          </div>
        </div>

        {/* Input panel */}
        <div style={css.panel}>
          <div>
            <label style={css.label}>Input text</label>
            <textarea
              style={{ ...css.textarea, ...(taFocused ? css.textareaFocus : {}) }}
              value={text}
              onChange={e => setText(e.target.value)}
              onFocus={() => setTaFocused(true)}
              onBlur={() => setTaFocused(false)}
              placeholder="Enter the text you want to synthesise…"
              maxLength={maxText}
              spellCheck={false}
            />
            <div style={css.charCount(charOver)}>{text.length} / {maxText}</div>
          </div>

          <div style={css.row}>
            <div style={css.col}>
              <label style={css.label}>Language</label>
              <select style={css.select} value={language} onChange={e => setLanguage(e.target.value)}>
                {LANGUAGES.map(l => <option key={l}>{l}</option>)}
              </select>
            </div>
            {mode === "custom" && (
              <div style={css.col}>
                <label style={css.label}>Speaker</label>
                <select style={css.select} value={speaker} onChange={e => setSpeaker(e.target.value)}>
                  {SPEAKERS.map(s => <option key={s}>{s}</option>)}
                </select>
              </div>
            )}
          </div>

          {mode === "custom" && (
            <div>
              <label style={css.label}>Style instruction <span style={{ color: "#2a3050" }}>(optional)</span></label>
              <input
                style={css.input}
                value={instruct}
                onChange={e => setInstruct(e.target.value)}
                placeholder="e.g. Speak warmly and slowly"
              />
            </div>
          )}

          {mode === "clone" && (
            <>
              <div>
                <label style={css.label}>Reference audio <span style={{ color: "#2a3050" }}>(WAV, 3+ sec)</span></label>
                <div
                  style={{ ...css.fileZone, ...(refAudio ? { borderColor: "#3b4a7a", color: "#7b9fff" } : {}) }}
                  onClick={() => document.getElementById("refAudioInput").click()}
                >
                  {refAudio ? `✓ ${refAudio.name}` : "Click to upload WAV file"}
                  <input
                    id="refAudioInput" type="file" accept="audio/wav,audio/*"
                    style={{ display: "none" }}
                    onChange={e => setRefAudio(e.target.files[0] || null)}
                  />
                </div>
              </div>
              <div>
                <label style={css.label}>Reference transcript</label>
                <textarea
                  style={{ ...css.textarea, minHeight: "60px" }}
                  value={refText}
                  onChange={e => setRefText(e.target.value)}
                  placeholder="Exact transcript of the reference audio…"
                  spellCheck={false}
                />
              </div>
            </>
          )}

          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "10px" }}>
            <div style={css.streamToggle} onClick={() => setStreaming(s => !s)}>
              <div style={css.toggleTrack(streaming)}>
                <div style={css.toggleThumb(streaming)} />
              </div>
              <span style={css.toggleLabel}>Streaming</span>
            </div>
            {health && (
              <span style={{ fontSize: "11px", color: "#2a3050" }}>
                VRAM free: <span style={{ color: "#4a5070" }}>{health.vram?.free_mb ?? "?"}MB</span>
              </span>
            )}
          </div>

          <div style={css.actionRow}>
            <button style={css.btn("primary", !canGenerate)} disabled={!canGenerate} onClick={generate}>
              {loading ? "Generating…" : "Generate"}
            </button>
            {loading && (
              <button style={css.btn("danger")} onClick={stop}>Stop</button>
            )}
          </div>
        </div>

        {/* Status / error */}
        {(status || error) && (
          <div>
            {error  && <div style={css.errorMsg}>⚠ {error}</div>}
            {status && !error && <div style={css.statusBar}>› {status}</div>}
          </div>
        )}

        {/* Audio output */}
        {showOutput && (
          <div style={css.panel}>
            <label style={css.label}>Output</label>
            <div style={css.audioWrap}>
              <audio ref={audioRef} src={audioUrl} controls preload="metadata" />
            </div>
            {metrics && (
              <div style={css.metricsRow}>
                <span style={css.metric}>TTFA: <span style={css.metricVal}>{metrics.ttfa_ms}ms</span></span>
                <span style={css.metric}>RTF:  <span style={css.metricVal}>{metrics.rtf}×</span></span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}