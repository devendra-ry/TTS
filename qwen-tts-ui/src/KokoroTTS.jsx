/**
 * KokoroTTS.jsx
 * Root component — owns only top-level UI state and the generate/stop handlers.
 * All heavy logic lives in custom hooks and helper modules.
 */
import { useState, useRef, useCallback } from "react";

import { generateTTS, generateStream } from "./api";
import { wavChunksToBlob } from "./audioUtils";
import { S } from "./styles";
import { useHealthPoller } from "./hooks/useHealthPoller";
import { useAudioPlayer } from "./hooks/useAudioPlayer";
import { TopBar } from "./components/TopBar";
import { VoiceSelector } from "./components/VoiceSelector";
import { StreamToggle } from "./components/StreamToggle";
import { AudioOutput } from "./components/AudioOutput";

const DEFAULT_MAX = 300_000;

export default function KokoroTTS() {
    // ── UI state ──────────────────────────────────────────────────────────────
    const [text, setText] = useState("");
    const [speakerId, setSpeakerId] = useState(0);
    const [streaming, setStreaming] = useState(false);
    const [status, setStatus] = useState("");
    const [errMsg, setErrMsg] = useState("");
    const [loading, setLoading] = useState(false);
    const [metrics, setMetrics] = useState(null);
    const [taFocus, setTaFocus] = useState(false);

    const abortRef = useRef(null);

    // ── Custom hooks ──────────────────────────────────────────────────────────
    const health = useHealthPoller();
    const {
        audioRef,
        audioUrl,
        clearAudio,
        setAudioAndAutoplay,
        initAudioContext,
        resetScheduler,
        scheduleChunk,
    } = useAudioPlayer();

    // ── Derived values ────────────────────────────────────────────────────────
    const maxText = health?.max_text_length ?? DEFAULT_MAX;
    const charOver = text.length > maxText;
    const canGenerate = !loading && text.trim().length > 0 && !charOver;

    // ── Helpers ───────────────────────────────────────────────────────────────
    const resetState = useCallback(() => {
        setErrMsg("");
        setStatus("");
        setMetrics(null);
        clearAudio();
    }, [clearAudio]);

    // ── Non-streaming generation ──────────────────────────────────────────────
    const runNonStream = async (signal) => {
        setStatus("Generating…");
        const blob = await generateTTS(text.trim(), speakerId, signal);
        if (blob.size <= 44) throw new Error("Model returned an empty WAV (no audio samples).");
        setAudioAndAutoplay(URL.createObjectURL(blob));
        setStatus("Done.");
    };

    // ── Streaming generation ──────────────────────────────────────────────────
    const runStream = async (signal) => {
        setStatus("Connecting…");
        await initAudioContext();
        resetScheduler();

        const allChunks = [];
        let decodedChunks = 0;
        let decodeFailures = 0;
        let sr = 24000;

        const res = await generateStream(text.trim(), speakerId, signal);
        const reader = res.body.getReader();
        const dec = new TextDecoder();
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
                if (msg.done) {
                    setMetrics({ ttfa_ms: msg.ttfa_ms, rtf: msg.rtf });
                    setStatus("Done.");
                    continue;
                }
                if (!msg.chunk) continue;

                allChunks.push(msg.chunk);
                sr = msg.sample_rate || sr;

                const decoded = await scheduleChunk(msg.chunk);
                decoded ? (decodedChunks += 1) : (decodeFailures += 1);
            }
        }

        if (allChunks.length) {
            const blob = wavChunksToBlob(allChunks, sr);
            if (blob) setAudioAndAutoplay(URL.createObjectURL(blob));
        }
        if (!allChunks.length) throw new Error("Stream ended without any audio chunks.");
        if (!decodedChunks && decodeFailures > 0) {
            throw new Error("Received chunks but browser failed to decode audio. Try non-stream mode.");
        }
    };

    // ── Generate / Stop ───────────────────────────────────────────────────────
    const generate = async () => {
        resetState();
        setLoading(true);
        const ctrl = new AbortController();
        abortRef.current = ctrl;
        try {
            await (streaming ? runStream(ctrl.signal) : runNonStream(ctrl.signal));
        } catch (e) {
            if (e.name === "AbortError") setStatus("Stopped.");
            else { setErrMsg(e.message || "Unknown error"); setStatus(""); }
        } finally {
            setLoading(false);
            abortRef.current = null;
        }
    };

    const stop = () => {
        abortRef.current?.abort();
    };

    // ── Render ────────────────────────────────────────────────────────────────
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

            <TopBar health={health} />

            <div style={S.main}>
                {/* Heading */}
                <div>
                    <div style={S.heading}>Text-to-Speech</div>
                    <div style={{ ...S.sub, marginTop: 6 }}>
                        hexgrad/Kokoro-82M · English · {maxText.toLocaleString()} char limit
                    </div>
                </div>

                {/* Model info */}
                <div style={S.infoBox}>
                    <strong style={{ color: "#7b9fff" }}>Kokoro-82M</strong> by hexgrad.
                    &nbsp;<strong style={{ color: "#7b9fff" }}>English only.</strong>
                    &nbsp;Speaker IDs 0–9 map to preset Kokoro voices.
                </div>

                {/* Input panel */}
                <div style={S.panel}>
                    {/* Text area */}
                    <div>
                        <label style={S.label}>Input text</label>
                        <textarea
                            style={{ ...S.textarea, ...(taFocus ? S.taFocus : {}) }}
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            onFocus={() => setTaFocus(true)}
                            onBlur={() => setTaFocus(false)}
                            placeholder="Enter the text you want to synthesise…"
                            maxLength={maxText}
                            spellCheck={false}
                        />
                        <div style={S.charCount(charOver)}>
                            {text.length.toLocaleString()} / {maxText.toLocaleString()}
                        </div>
                    </div>

                    <VoiceSelector health={health} speakerId={speakerId} onChange={setSpeakerId} />

                    <StreamToggle
                        streaming={streaming}
                        onToggle={() => setStreaming((s) => !s)}
                        health={health}
                    />

                    {/* Action buttons */}
                    <div style={S.actionRow}>
                        <button
                            style={S.btn("primary", !canGenerate)}
                            disabled={!canGenerate}
                            onClick={generate}
                        >
                            {loading ? "Generating…" : "Generate"}
                        </button>
                        {loading && (
                            <button style={S.btn("danger")} onClick={stop}>
                                Stop
                            </button>
                        )}
                    </div>
                </div>

                {/* Status / error */}
                {(status || errMsg) && (
                    <div>
                        {errMsg && <div style={S.error}>⚠ {errMsg}</div>}
                        {status && !errMsg && <div style={S.status}>› {status}</div>}
                    </div>
                )}

                {/* Audio output */}
                <AudioOutput audioRef={audioRef} audioUrl={audioUrl} metrics={metrics} />
            </div>
        </div>
    );
}
