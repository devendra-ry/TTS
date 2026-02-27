import { useState, useRef, useEffect, useCallback } from "react";

/* ─── Constants ─────────────────────────────────────────────────────────── */
const API = "http://localhost:8000";
const SPEAKERS = ["serena","vivian","uncle_fu","ryan","aiden","ono_anna","sohee","eric","dylan"];
const LANGUAGES = ["English","Chinese","French","German","Japanese","Korean","Spanish","Portuguese","Arabic","Russian"];
const DEFAULT_MAX_TEXT = 300000;

/* ─── Base64 → Float32 helper ────────────────────────────────────────────── */
function wavBytesToFloat32(b64) {
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  // Skip WAV header (44 bytes), read PCM as Int16
  const view = new DataView(bytes.buffer, 44);
  const samples = new Float32Array(view.byteLength / 2);
  for (let i = 0; i < samples.length; i++) {
    samples[i] = view.getInt16(i * 2, true) / 32768.0;
  }
  return samples;
}

function float32ToWavBlob(samples, sampleRate = 24000) {
  const bytesPerSample = 2;
  const numChannels = 1;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
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
  heading: {
    fontSize: "22px",
    fontWeight: 700,
    color: "#e4e8f8",
    letterSpacing: "0.05em",
    marginBottom: "-10px",
  },
  subheading: {
    fontSize: "11px",
    color: "#4a5070",
    letterSpacing: "0.15em",
    textTransform: "uppercase",
  },
  panel: {
    background: "#111118",
    border: "1px solid #1e2035",
    borderRadius: "8px",
    padding: "20px 22px",
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  },
  label: {
    fontSize: "11px",
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    color: "#4a5070",
    marginBottom: "5px",
    display: "block",
  },
  textarea: {
    width: "100%",
    minHeight: "120px",
    background: "#0c0c0f",
    border: "1px solid #1e2035",
    borderRadius: "5px",
    color: "#c8d0e8",
    fontFamily: "inherit",
    fontSize: "14px",
    lineHeight: "1.6",
    padding: "12px 14px",
    resize: "vertical",
    outline: "none",
    boxSizing: "border-box",
    transition: "border-color 0.15s",
  },
  textareaFocus: {
    borderColor: "#3b4a7a",
  },
  row: {
    display: "flex",
    gap: "12px",
    flexWrap: "wrap",
  },
  col: {
    display: "flex",
    flexDirection: "column",
    flex: 1,
    minWidth: "140px",
  },
  select: {
    background: "#0c0c0f",
    border: "1px solid #1e2035",
    borderRadius: "5px",
    color: "#c8d0e8",
    fontFamily: "inherit",
    fontSize: "13px",
    padding: "9px 12px",
    outline: "none",
    cursor: "pointer",
    appearance: "none",
    backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%234a5070'/%3E%3C/svg%3E")`,
    backgroundRepeat: "no-repeat",
    backgroundPosition: "right 12px center",
    paddingRight: "30px",
  },
  input: {
    background: "#0c0c0f",
    border: "1px solid #1e2035",
    borderRadius: "5px",
    color: "#c8d0e8",
    fontFamily: "inherit",
    fontSize: "13px",
    padding: "9px 12px",
    outline: "none",
    width: "100%",
    boxSizing: "border-box",
  },
  modeToggle: {
    display: "flex",
    background: "#0c0c0f",
    border: "1px solid #1e2035",
    borderRadius: "6px",
    overflow: "hidden",
    width: "fit-content",
  },
  modeBtn: (active) => ({
    padding: "8px 22px",
    fontSize: "12px",
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    border: "none",
    cursor: "pointer",
    fontFamily: "inherit",
    background: active ? "#7b9fff" : "transparent",
    color: active ? "#0c0c0f" : "#4a5070",
    fontWeight: active ? 700 : 400,
    transition: "all 0.15s",
  }),
  streamToggle: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    cursor: "pointer",
    userSelect: "none",
  },
  toggleTrack: (on) => ({
    width: "36px",
    height: "20px",
    borderRadius: "10px",
    background: on ? "#7b9fff44" : "#1e2035",
    border: `1px solid ${on ? "#7b9fff" : "#2a2f50"}`,
    position: "relative",
    transition: "all 0.2s",
    cursor: "pointer",
    flexShrink: 0,
  }),
  toggleThumb: (on) => ({
    width: "14px",
    height: "14px",
    borderRadius: "50%",
    background: on ? "#7b9fff" : "#3a4060",
    position: "absolute",
    top: "2px",
    left: on ? "18px" : "2px",
    transition: "left 0.2s",
    boxShadow: on ? "0 0 6px #7b9fff88" : "none",
  }),
  toggleLabel: {
    fontSize: "12px",
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    color: "#4a5070",
  },
  actionRow: {
    display: "flex",
    gap: "10px",
    alignItems: "center",
    flexWrap: "wrap",
  },
  btn: (variant = "primary", disabled = false) => ({
    padding: "10px 24px",
    fontFamily: "inherit",
    fontSize: "12px",
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    fontWeight: 700,
    border: "none",
    borderRadius: "5px",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.4 : 1,
    transition: "all 0.15s",
    ...(variant === "primary" ? {
      background: "#7b9fff",
      color: "#0c0c0f",
    } : variant === "danger" ? {
      background: "transparent",
      color: "#ff6b6b",
      border: "1px solid #ff6b6b44",
    } : {
      background: "transparent",
      color: "#7b9fff",
      border: "1px solid #7b9fff44",
    }),
  }),
  charCount: (over) => ({
    fontSize: "11px",
    color: over ? "#ff6b6b" : "#2a3050",
    textAlign: "right",
    marginTop: "-8px",
  }),
  audioWrap: {
    background: "#0c0c0f",
    border: "1px solid #1e2035",
    borderRadius: "5px",
    padding: "12px 14px",
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },
  metricsRow: {
    display: "flex",
    gap: "20px",
    flexWrap: "wrap",
  },
  metric: {
    fontSize: "11px",
    color: "#4a5070",
    letterSpacing: "0.08em",
  },
  metricVal: {
    color: "#7b9fff",
    fontWeight: 700,
  },
  statusBar: {
    fontSize: "12px",
    color: "#4a5070",
    letterSpacing: "0.08em",
    minHeight: "18px",
    padding: "2px 0",
  },
  errorMsg: {
    fontSize: "12px",
    color: "#ff6b6b",
    background: "#200f0f",
    border: "1px solid #3a1515",
    borderRadius: "5px",
    padding: "10px 14px",
  },
  fileZone: {
    border: "1px dashed #2a2f50",
    borderRadius: "5px",
    padding: "18px",
    textAlign: "center",
    fontSize: "12px",
    color: "#4a5070",
    cursor: "pointer",
    transition: "border-color 0.15s",
    background: "#0c0c0f",
  },
};

/* ─── Component ──────────────────────────────────────────────────────────── */
export default function QwenTTS() {
  const [mode, setMode]             = useState("custom"); // "custom" | "clone"
  const [text, setText]             = useState("");
  const [language, setLanguage]     = useState("English");
  const [speaker, setSpeaker]       = useState("aiden");
  const [instruct, setInstruct]     = useState("");
  const [streaming, setStreaming]   = useState(false);

  // Clone mode
  const [refAudio, setRefAudio]     = useState(null);   // File
  const [refText, setRefText]       = useState("");

  // UI state
  const [status, setStatus]         = useState("");
  const [error, setError]           = useState("");
  const [loading, setLoading]       = useState(false);
  const [health, setHealth]         = useState(null);

  // Audio
  const [audioUrl, setAudioUrl]     = useState(null);
  const [metrics, setMetrics]       = useState(null); // {ttfa_ms, rtf}
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const audioRef                    = useRef(null);
  const abortRef                    = useRef(null);
  const audioCtxRef                 = useRef(null);
  const nextPlayTimeRef             = useRef(0);

  // Textarea focus
  const [taFocused, setTaFocused]   = useState(false);
  const maxText = health?.max_text_length ?? DEFAULT_MAX_TEXT;

  /* Poll health on mount */
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API}/health`);
        const d = await r.json();
        setHealth(d);
      } catch {
        setHealth(null);
      }
    };
    check();
    const id = setInterval(check, 8000);
    return () => clearInterval(id);
  }, []);

  /* Cleanup AudioContext on unmount */
  useEffect(() => {
    return () => { audioCtxRef.current?.close(); };
  }, []);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return undefined;

    const sync = () => setIsAudioPlaying(!audio.paused);
    const onEnded = () => setIsAudioPlaying(false);
    audio.addEventListener("play", sync);
    audio.addEventListener("pause", sync);
    audio.addEventListener("ended", onEnded);
    setIsAudioPlaying(!audio.paused);

    return () => {
      audio.removeEventListener("play", sync);
      audio.removeEventListener("pause", sync);
      audio.removeEventListener("ended", onEnded);
    };
  }, [audioUrl]);

  const clearState = useCallback(() => {
    setError("");
    setStatus("");
    setMetrics(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
      nextPlayTimeRef.current = 0;
    }
  }, [audioUrl]);

  /* ── Non-streaming request ─ */
  const runNonStream = async (formData, endpoint) => {
    setStatus("Generating…");
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const r = await fetch(`${API}${endpoint}`, {
      method: "POST",
      body: formData,
      signal: ctrl.signal,
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.detail || `HTTP ${r.status}`);
    }
    const blob = await r.blob();
    const url  = URL.createObjectURL(blob);
    setAudioUrl(url);
    setStatus("Done.");
  };

  /* ── Streaming request ─ */
  const runStream = async (formData, endpoint) => {
    setStatus("Connecting…");
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    nextPlayTimeRef.current = audioCtxRef.current.currentTime + 0.05;

    const allPcmChunks = [];

    const r = await fetch(`${API}${endpoint}`, {
      method: "POST",
      body: formData,
      signal: ctrl.signal,
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.detail || `HTTP ${r.status}`);
    }

    const reader  = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer    = "";

    setStatus("Streaming…");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (msg.error) throw new Error(msg.error);

        if (msg.done) {
          setMetrics({ ttfa_ms: msg.ttfa_ms, rtf: msg.rtf });
          setStatus("Done.");
        } else if (msg.chunk) {
          // Decode once per chunk, then use for both playback and final WAV.
          try {
            const pcm  = wavBytesToFloat32(msg.chunk);
            allPcmChunks.push(pcm);
            const ctx  = audioCtxRef.current;
            const buf  = ctx.createBuffer(1, pcm.length, 24000);
            buf.copyToChannel(pcm, 0);
            const src  = ctx.createBufferSource();
            src.buffer = buf;
            src.connect(ctx.destination);
            const startAt = Math.max(nextPlayTimeRef.current, ctx.currentTime);
            src.start(startAt);
            nextPlayTimeRef.current = startAt + buf.duration;
          } catch {/* non-fatal */ }
        }
      }
    }

    if (allPcmChunks.length) {
      const totalSamples = allPcmChunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const merged = new Float32Array(totalSamples);
      let offset = 0;
      allPcmChunks.forEach((chunk) => {
        merged.set(chunk, offset);
        offset += chunk.length;
      });
      const wavBlob = float32ToWavBlob(merged, 24000);
      const url = URL.createObjectURL(wavBlob);
      setAudioUrl(url);
    }
  };

  /* ── Main generate handler ─ */
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

        if (streaming) {
          await runStream(fd, "/tts/custom/stream");
        } else {
          await runNonStream(fd, "/tts/custom");
        }
      } else {
        if (!refAudio) throw new Error("Please upload a reference WAV file.");
        if (!refText.trim()) throw new Error("Please provide the reference audio transcript.");
        fd.append("ref_audio", refAudio);
        fd.append("ref_text", refText.trim());

        if (streaming) {
          await runStream(fd, "/tts/clone/stream");
        } else {
          await runNonStream(fd, "/tts/clone");
        }
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

  const download = () => {
    if (!audioUrl) return;
    const a = document.createElement("a");
    a.href = audioUrl;
    a.download = "qwen_tts_output.wav";
    a.click();
  };

  const togglePlayPause = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      audio.play().catch(() => {});
    } else {
      audio.pause();
    }
  };

  const seekBy = (seconds) => {
    const audio = audioRef.current;
    if (!audio) return;
    const duration = Number.isFinite(audio.duration) ? audio.duration : Number.MAX_SAFE_INTEGER;
    const next = Math.max(0, Math.min(duration, audio.currentTime + seconds));
    audio.currentTime = next;
  };

  const restart = () => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = 0;
    audio.play().catch(() => {});
  };

  const charOver = text.length > maxText;
  const canGenerate = !loading && text.trim().length > 0 && !charOver &&
    (mode === "custom" || (refAudio && refText.trim()));

  return (
    <div style={css.root}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fira+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; background: #0c0c0f; }
        select option { background: #111118; }
        @keyframes pulse {
          0%,100% { opacity:1; }
          50% { opacity:0.4; }
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0c0c0f; }
        ::-webkit-scrollbar-thumb { background: #1e2035; border-radius: 3px; }
        audio { width: 100%; accent-color: #7b9fff; filter: invert(0); }
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
          <div style={{...css.subheading, marginTop:"6px"}}>
            Local inference · GTX 1650 · {maxText} char request limit
          </div>
        </div>

        {/* Mode toggle */}
        <div>
          <div style={css.modeToggle}>
            <button style={css.modeBtn(mode==="custom")} onClick={()=>setMode("custom")}>Custom Voice</button>
            <button style={css.modeBtn(mode==="clone")} onClick={()=>setMode("clone")}>Voice Clone</button>
          </div>
        </div>

        {/* Input panel */}
        <div style={css.panel}>
          {/* Text */}
          <div>
            <label style={css.label}>Input text</label>
            <textarea
              style={{
                ...css.textarea,
                ...(taFocused ? css.textareaFocus : {}),
              }}
              value={text}
              onChange={e => setText(e.target.value)}
              onFocus={() => setTaFocused(true)}
              onBlur={() => setTaFocused(false)}
              placeholder="Enter the text you want to synthesise…"
              maxLength={maxText}
              spellCheck={false}
            />
            <div style={css.charCount(charOver)}>
              {text.length} / {maxText}
            </div>
          </div>

          {/* Language + Speaker (custom) */}
          <div style={css.row}>
            <div style={css.col}>
              <label style={css.label}>Language</label>
              <select style={css.select} value={language} onChange={e=>setLanguage(e.target.value)}>
                {LANGUAGES.map(l => <option key={l}>{l}</option>)}
              </select>
            </div>

            {mode === "custom" && (
              <div style={css.col}>
                <label style={css.label}>Speaker</label>
                <select style={css.select} value={speaker} onChange={e=>setSpeaker(e.target.value)}>
                  {SPEAKERS.map(s => <option key={s}>{s}</option>)}
                </select>
              </div>
            )}
          </div>

          {/* Style instruction (custom) */}
          {mode === "custom" && (
            <div>
              <label style={css.label}>Style instruction <span style={{color:"#2a3050"}}>(optional)</span></label>
              <input
                style={css.input}
                value={instruct}
                onChange={e=>setInstruct(e.target.value)}
                placeholder="e.g. Speak warmly and slowly"
              />
            </div>
          )}

          {/* Reference audio (clone) */}
          {mode === "clone" && (
            <>
              <div>
                <label style={css.label}>Reference audio <span style={{color:"#2a3050"}}>(WAV, 3+ sec)</span></label>
                <div
                  style={{...css.fileZone, ...(refAudio ? {borderColor:"#3b4a7a", color:"#7b9fff"} : {})}}
                  onClick={() => document.getElementById("refAudioInput").click()}
                >
                  {refAudio ? `✓ ${refAudio.name}` : "Click to upload WAV file"}
                  <input
                    id="refAudioInput"
                    type="file"
                    accept="audio/wav,audio/*"
                    style={{display:"none"}}
                    onChange={e => setRefAudio(e.target.files[0] || null)}
                  />
                </div>
              </div>
              <div>
                <label style={css.label}>Reference transcript</label>
                <textarea
                  style={{...css.textarea, minHeight:"60px"}}
                  value={refText}
                  onChange={e=>setRefText(e.target.value)}
                  placeholder="Exact transcript of the reference audio…"
                  spellCheck={false}
                />
              </div>
            </>
          )}

          {/* Stream toggle */}
          <div style={{display:"flex", alignItems:"center", justifyContent:"space-between", flexWrap:"wrap", gap:"10px"}}>
            <div style={css.streamToggle} onClick={()=>setStreaming(s=>!s)}>
              <div style={css.toggleTrack(streaming)}>
                <div style={css.toggleThumb(streaming)} />
              </div>
              <span style={css.toggleLabel}>Streaming</span>
            </div>
            {health && (
              <span style={{fontSize:"11px", color:"#2a3050"}}>
                VRAM free: <span style={{color:"#4a5070"}}>{health.vram?.free_mb ?? "?"}MB</span>
              </span>
            )}
          </div>

          {/* Buttons */}
          <div style={css.actionRow}>
            <button
              style={css.btn("primary", !canGenerate)}
              disabled={!canGenerate}
              onClick={generate}
            >
              {loading ? "Generating…" : "Generate"}
            </button>
            {loading && (
              <button style={css.btn("danger")} onClick={stop}>
                Stop
              </button>
            )}
          </div>
        </div>

        {/* Status */}
        {(status || error) && (
          <div>
            {error && <div style={css.errorMsg}>⚠ {error}</div>}
            {status && !error && <div style={css.statusBar}>› {status}</div>}
          </div>
        )}

        {/* Audio output */}
        {audioUrl && (
          <div style={css.panel}>
            <label style={css.label}>Output</label>
            <div style={css.audioWrap}>
              <audio ref={audioRef} src={audioUrl} controls preload="metadata" />
            </div>

            {metrics && (
              <div style={css.metricsRow}>
                <span style={css.metric}>
                  TTFA: <span style={css.metricVal}>{metrics.ttfa_ms}ms</span>
                </span>
                <span style={css.metric}>
                  RTF: <span style={css.metricVal}>{metrics.rtf}×</span>
                </span>
              </div>
            )}

            <div style={css.actionRow}>
              <button style={css.btn("secondary")} onClick={togglePlayPause}>
                {isAudioPlaying ? "Pause" : "Play"}
              </button>
              <button style={css.btn("secondary")} onClick={() => seekBy(-10)}>
                Rewind 10s
              </button>
              <button style={css.btn("secondary")} onClick={() => seekBy(10)}>
                Forward 10s
              </button>
              <button style={css.btn("secondary")} onClick={restart}>
                Restart
              </button>
              <button style={css.btn("secondary")} onClick={download}>
                Download WAV
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
