import { useState, useRef, useEffect, useCallback } from "react";

/* ─── Constants ─────────────────────────────────────────────────────────── */
const API = "http://localhost:8000";
const DEFAULT_MAX = 300000;
const SPEAKER_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/* ─── Audio helpers ──────────────────────────────────────────────────────── */
function base64ToArrayBuffer(b64) {
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) view[i] = raw.charCodeAt(i);
  return buf;
}

// Reassemble all streamed per-chunk WAVs into one downloadable WAV.
// Finds the PCM data offset dynamically so variable-length headers are safe.
function wavChunksToBlob(chunksB64, sr = 24000) {
  const toBytes = b64 => {
    const raw = atob(b64), out = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
    return out;
  };
  const getPcmOffset = bytes => {
    const view = new DataView(bytes.buffer);
    let pos = 12;
    while (pos + 8 <= bytes.length) {
      if (String.fromCharCode(...bytes.subarray(pos, pos + 4)) === "data") return pos + 8;
      pos += 8 + view.getUint32(pos + 4, true);
    }
    return 44;
  };
  const pcm = chunksB64.map(toBytes).filter(b => b.length > 8).map(b => b.subarray(getPcmOffset(b)));
  if (!pcm.length) return null;
  const dataSize = pcm.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(44 + dataSize);
  const view = new DataView(out.buffer);
  const ws = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); view.setUint32(4, 36 + dataSize, true);
  ws(8, "WAVE"); ws(12, "fmt "); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sr, true); view.setUint32(28, sr * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  ws(36, "data"); view.setUint32(40, dataSize, true);
  let off = 44;
  pcm.forEach(c => { out.set(c, off); off += c.length; });
  return new Blob([out], { type: "audio/wav" });
}

/* ─── Styles ─────────────────────────────────────────────────────────────── */
const S = {
  root: { minHeight: "100vh", background: "#0c0c0f", color: "#c8d0e8", fontFamily: "'Fira Mono','JetBrains Mono','Courier New',monospace", margin: 0, padding: 0, display: "flex", flexDirection: "column", alignItems: "center" },
  topBar: { width: "100%", borderBottom: "1px solid #1e2030", padding: "14px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", background: "#0e0e14", boxSizing: "border-box" },
  logo: { fontSize: "13px", letterSpacing: "0.18em", color: "#7b9fff", textTransform: "uppercase", fontWeight: 700 },
  health: ok => ({ display: "inline-flex", alignItems: "center", gap: "6px", fontSize: "11px", color: ok ? "#5af07e" : "#ff6b6b" }),
  dot: ok => ({ width: 7, height: 7, borderRadius: "50%", background: ok ? "#5af07e" : "#ff6b6b", boxShadow: ok ? "0 0 6px #5af07e88" : "0 0 6px #ff6b6b88", animation: ok ? "pulse 2s infinite" : "none" }),
  main: { width: "100%", maxWidth: 760, padding: "40px 24px 80px", boxSizing: "border-box", display: "flex", flexDirection: "column", gap: 28 },
  heading: { fontSize: 22, fontWeight: 700, color: "#e4e8f8", letterSpacing: "0.05em", marginBottom: -10 },
  sub: { fontSize: 11, color: "#4a5070", letterSpacing: "0.15em", textTransform: "uppercase" },
  panel: { background: "#111118", border: "1px solid #1e2035", borderRadius: 8, padding: "20px 22px", display: "flex", flexDirection: "column", gap: 16 },
  label: { fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "#4a5070", marginBottom: 5, display: "block" },
  textarea: { width: "100%", minHeight: 120, background: "#0c0c0f", border: "1px solid #1e2035", borderRadius: 5, color: "#c8d0e8", fontFamily: "inherit", fontSize: 14, lineHeight: 1.6, padding: "12px 14px", resize: "vertical", outline: "none", boxSizing: "border-box", transition: "border-color 0.15s" },
  taFocus: { borderColor: "#3b4a7a" },
  row: { display: "flex", gap: 12, flexWrap: "wrap" },
  col: { display: "flex", flexDirection: "column", flex: 1, minWidth: 140 },
  select: { background: "#0c0c0f", border: "1px solid #1e2035", borderRadius: 5, color: "#c8d0e8", fontFamily: "inherit", fontSize: 13, padding: "9px 12px", outline: "none", cursor: "pointer", appearance: "none", backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%234a5070'/%3E%3C/svg%3E")`, backgroundRepeat: "no-repeat", backgroundPosition: "right 12px center", paddingRight: 30 },
  toggleWrap: { display: "flex", alignItems: "center", gap: 10, cursor: "pointer", userSelect: "none" },
  track: on => ({ width: 36, height: 20, borderRadius: 10, background: on ? "#7b9fff44" : "#1e2035", border: `1px solid ${on ? "#7b9fff" : "#2a2f50"}`, position: "relative", transition: "all 0.2s", flexShrink: 0 }),
  thumb: on => ({ width: 14, height: 14, borderRadius: "50%", background: on ? "#7b9fff" : "#3a4060", position: "absolute", top: 2, left: on ? 18 : 2, transition: "left 0.2s", boxShadow: on ? "0 0 6px #7b9fff88" : "none" }),
  toggleLbl: { fontSize: 12, letterSpacing: "0.1em", textTransform: "uppercase", color: "#4a5070" },
  actionRow: { display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" },
  btn: (v = "primary", d = false) => ({ padding: "10px 24px", fontFamily: "inherit", fontSize: 12, letterSpacing: "0.12em", textTransform: "uppercase", fontWeight: 700, border: "none", borderRadius: 5, cursor: d ? "not-allowed" : "pointer", opacity: d ? 0.4 : 1, transition: "all 0.15s", ...(v === "primary" ? { background: "#7b9fff", color: "#0c0c0f" } : v === "danger" ? { background: "transparent", color: "#ff6b6b", border: "1px solid #ff6b6b44" } : { background: "transparent", color: "#7b9fff", border: "1px solid #7b9fff44" }) }),
  charCount: o => ({ fontSize: 11, color: o ? "#ff6b6b" : "#2a3050", textAlign: "right", marginTop: -8 }),
  audioWrap: { background: "#0c0c0f", border: "1px solid #1e2035", borderRadius: 5, padding: "12px 14px" },
  metrics: { display: "flex", gap: 20, flexWrap: "wrap" },
  metric: { fontSize: 11, color: "#4a5070", letterSpacing: "0.08em" },
  mVal: { color: "#7b9fff", fontWeight: 700 },
  status: { fontSize: 12, color: "#4a5070", letterSpacing: "0.08em", minHeight: 18, padding: "2px 0" },
  error: { fontSize: 12, color: "#ff6b6b", background: "#200f0f", border: "1px solid #3a1515", borderRadius: 5, padding: "10px 14px" },
  infoBox: { fontSize: 11, color: "#4a5070", background: "#0e0e14", border: "1px solid #1e2035", borderRadius: 5, padding: "10px 14px", lineHeight: 1.7 },
};

/* ─── Component ──────────────────────────────────────────────────────────── */
export default function CSMTTS() {
  const [text, setText] = useState("");
  const [speakerId, setSpeakerId] = useState(0);
  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState("");
  const [errMsg, setErrMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [taFocus, setTaFocus] = useState(false);

  const audioRef = useRef(null);
  const abortRef = useRef(null);
  const audioCtxRef = useRef(null);
  const nextPlayRef = useRef(0);
  const startedRef = useRef(false);

  const maxText = health?.max_text_length ?? DEFAULT_MAX;

  useEffect(() => {
    const poll = async () => {
      try { setHealth(await (await fetch(`${API}/health`)).json()); }
      catch { setHealth(null); }
    };
    poll();
    const id = setInterval(poll, 8000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => () => { audioCtxRef.current?.close(); }, []);

  const clearState = useCallback(() => {
    setErrMsg(""); setStatus(""); setMetrics(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    nextPlayRef.current = 0;
    startedRef.current = false;
  }, [audioUrl]);

  /* ── Non-streaming ──────────────────────────────────────────────────────── */
  const runNonStream = async (fd, endpoint) => {
    setStatus("Generating…");
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const r = await fetch(`${API}${endpoint}`, { method: "POST", body: fd, signal: ctrl.signal });
    if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error(j.detail || `HTTP ${r.status}`); }
    setAudioUrl(URL.createObjectURL(await r.blob()));
    setStatus("Done.");
  };

  /* ── Streaming ──────────────────────────────────────────────────────────── */
  const runStream = async (fd, endpoint) => {
    setStatus("Connecting…");
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    // latencyHint "interactive" minimises output-buffer size for precise scheduling.
    const Ctx = window.AudioContext || window.webkitAudioContext;
    audioCtxRef.current = new Ctx({ latencyHint: "interactive" });
    if (audioCtxRef.current.state === "suspended") {
      await audioCtxRef.current.resume();
      if (audioCtxRef.current.state === "suspended")
        throw new Error("Browser blocked audio autoplay. Click anywhere on the page, then try again.");
    }
    nextPlayRef.current = 0;
    startedRef.current = false;

    const allChunks = [];
    let sr = 24000;

    const r = await fetch(`${API}${endpoint}`, { method: "POST", body: fd, signal: ctrl.signal });
    if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error(j.detail || `HTTP ${r.status}`); }

    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    setStatus("Streaming…");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (msg.error) throw new Error(msg.error);
        if (msg.done) { setMetrics({ ttfa_ms: msg.ttfa_ms, rtf: msg.rtf }); setStatus("Done."); continue; }
        if (!msg.chunk) continue;

        allChunks.push(msg.chunk);
        sr = msg.sample_rate || sr;

        const ctx = audioCtxRef.current;
        if (!ctx) continue;
        try {
          // decodeAudioData handles any WAV header length and sample rate correctly —
          // no manual header parsing or int16 conversion needed.
          const audioBuf = await ctx.decodeAudioData(base64ToArrayBuffer(msg.chunk));
          const LOOKAHEAD = 0.05; // 50 ms: absorbs one decode round-trip without audible gap

          if (!startedRef.current) {
            nextPlayRef.current = ctx.currentTime + LOOKAHEAD;
            startedRef.current = true;
          } else if (nextPlayRef.current < ctx.currentTime) {
            // Fell behind (e.g. very slow GPU frame). Catch up without inserting silence.
            nextPlayRef.current = ctx.currentTime + LOOKAHEAD;
          }

          const src = ctx.createBufferSource();
          src.buffer = audioBuf;
          src.connect(ctx.destination);
          src.start(nextPlayRef.current);
          // Advance by exact decoded duration → chunks tile with zero gap.
          nextPlayRef.current += audioBuf.duration;
        } catch (e) {
          console.warn("decodeAudioData failed:", e);
        }
      }
    }

    if (allChunks.length) {
      const blob = wavChunksToBlob(allChunks, sr);
      if (blob) setAudioUrl(URL.createObjectURL(blob));
    }
  };

  /* ── Generate handler ───────────────────────────────────────────────────── */
  const generate = async () => {
    clearState();
    setLoading(true);
    const fd = new FormData();
    fd.append("text", text.trim());
    fd.append("speaker_id", speakerId);
    try {
      await (streaming ? runStream(fd, "/tts/generate/stream") : runNonStream(fd, "/tts/generate"));
    } catch (e) {
      if (e.name === "AbortError") setStatus("Stopped.");
      else { setErrMsg(e.message || "Unknown error"); setStatus(""); }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  };

  const stop = () => { abortRef.current?.abort(); audioCtxRef.current?.suspend(); };

  const charOver = text.length > maxText;
  const canGenerate = !loading && text.trim().length > 0 && !charOver;

  return (
    <div style={S.root}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fira+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; } body { margin: 0; background: #0c0c0f; }
        select option { background: #111118; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
        ::-webkit-scrollbar{width:6px} ::-webkit-scrollbar-track{background:#0c0c0f}
        ::-webkit-scrollbar-thumb{background:#1e2035;border-radius:3px}
        audio{width:100%;accent-color:#7b9fff}
        audio::-webkit-media-controls-panel{background:#111118}
      `}</style>

      {/* Top bar */}
      <div style={S.topBar}>
        <span style={S.logo}>◈ CSM-1B TTS</span>
        <span style={S.health(!!health)}>
          <span style={S.dot(!!health)} />
          {health
            ? `READY · ${health.vram?.device ?? "GPU"} · ${health.vram?.free_mb ?? "?"}MB free`
            : "OFFLINE"}
        </span>
      </div>

      <div style={S.main}>
        <div>
          <div style={S.heading}>Text-to-Speech</div>
          <div style={{ ...S.sub, marginTop: 6 }}>
            sesame/csm-1b · English · {maxText.toLocaleString()} char limit
          </div>
        </div>

        {/* Model info */}
        <div style={S.infoBox}>
          <strong style={{ color: "#7b9fff" }}>CSM-1B</strong> by Sesame — Llama backbone + Mimi vocoder.
          &nbsp;<strong style={{ color: "#7b9fff" }}>English only.</strong>
          &nbsp;Speaker IDs 0–9 select different built-in voice identities (0 = default neutral voice).
        </div>

        {/* Input panel */}
        <div style={S.panel}>
          {/* Text */}
          <div>
            <label style={S.label}>Input text</label>
            <textarea
              style={{ ...S.textarea, ...(taFocus ? S.taFocus : {}) }}
              value={text} onChange={e => setText(e.target.value)}
              onFocus={() => setTaFocus(true)} onBlur={() => setTaFocus(false)}
              placeholder="Enter the text you want to synthesise…"
              maxLength={maxText} spellCheck={false}
            />
            <div style={S.charCount(charOver)}>{text.length.toLocaleString()} / {maxText.toLocaleString()}</div>
          </div>

          {/* Speaker ID */}
          <div style={{ ...S.col, maxWidth: 220 }}>
            <label style={S.label}>
              Speaker ID
              <span style={{ color: "#2a3050", marginLeft: 6 }}>0–9</span>
            </label>
            <select style={S.select} value={speakerId} onChange={e => setSpeakerId(Number(e.target.value))}>
              {SPEAKER_IDS.map(id => (
                <option key={id} value={id}>{id === 0 ? `${id} (default)` : id}</option>
              ))}
            </select>
          </div>

          {/* Stream toggle */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
            <div style={S.toggleWrap} onClick={() => setStreaming(s => !s)}>
              <div style={S.track(streaming)}><div style={S.thumb(streaming)} /></div>
              <span style={S.toggleLbl}>Streaming</span>
            </div>
            {health && (
              <span style={{ fontSize: 11, color: "#2a3050" }}>
                chunk: <span style={{ color: "#4a5070" }}>{health.model_chunk_text_len}c</span>
                &nbsp;·&nbsp;pre-roll: <span style={{ color: "#4a5070" }}>{health.pre_roll_chunks}</span>
              </span>
            )}
          </div>

          {/* Buttons */}
          <div style={S.actionRow}>
            <button style={S.btn("primary", !canGenerate)} disabled={!canGenerate} onClick={generate}>
              {loading ? "Generating…" : "Generate"}
            </button>
            {loading && <button style={S.btn("danger")} onClick={stop}>Stop</button>}
          </div>
        </div>

        {/* Status / error */}
        {(status || errMsg) && (
          <div>
            {errMsg && <div style={S.error}>⚠ {errMsg}</div>}
            {status && !errMsg && <div style={S.status}>› {status}</div>}
          </div>
        )}

        {/* Output */}
        {audioUrl && (
          <div style={S.panel}>
            <label style={S.label}>Output</label>
            <div style={S.audioWrap}>
              <audio ref={audioRef} src={audioUrl} controls preload="metadata" />
            </div>
            {metrics && (
              <div style={S.metrics}>
                <span style={S.metric}>TTFA: <span style={S.mVal}>{metrics.ttfa_ms}ms</span></span>
                <span style={S.metric}>RTF:  <span style={S.mVal}>{metrics.rtf}×</span></span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}