/**
 * hooks/useAudioPlayer.js
 * Custom hook — manages audio playback state for both streaming and
 * non-streaming modes.
 *
 * Returns:
 *   audioRef         – ref to attach to <audio>
 *   audioUrl         – object URL string for the <audio src>
 *   clearAudio()     – reset all playback state and revoke the object URL
 *   setAudioAndAutoplay(url) – set url and attempt autoplay
 *   scheduleChunk(b64)       – decode a base-64 WAV chunk and schedule it
 *                              via the Web Audio API (streaming mode)
 *   initAudioContext()       – create/resume AudioContext (streaming mode)
 *   resetScheduler()         – reset timing state between streams
 */
import { useState, useRef, useEffect, useCallback } from "react";
import { base64ToArrayBuffer } from "../audioUtils";

export function useAudioPlayer() {
    const [audioUrl, setAudioUrl] = useState(null);

    const audioRef = useRef(null);
    const audioCtxRef = useRef(null);
    const nextPlayRef = useRef(0);
    const startedRef = useRef(false);

    // Close AudioContext on unmount.
    useEffect(() => () => { audioCtxRef.current?.close(); }, []);

    const clearAudio = useCallback(() => {
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
        audioCtxRef.current?.close();
        audioCtxRef.current = null;
        nextPlayRef.current = 0;
        startedRef.current = false;
    }, [audioUrl]);

    const setAudioAndAutoplay = useCallback((url) => {
        setAudioUrl(url);
        requestAnimationFrame(() => {
            audioRef.current?.play().catch(() => {
                // Autoplay was blocked — the UI status message will guide the user.
            });
        });
    }, []);

    /**
     * Create (or reuse) an AudioContext and ensure it is running.
     * Throws if the browser blocks audio autoplay.
     */
    const initAudioContext = useCallback(async () => {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        audioCtxRef.current = new Ctx({ latencyHint: "interactive" });
        if (audioCtxRef.current.state === "suspended") {
            await audioCtxRef.current.resume();
            if (audioCtxRef.current.state === "suspended") {
                throw new Error(
                    "Browser blocked audio autoplay. Click anywhere on the page, then try again."
                );
            }
        }
    }, []);

    const resetScheduler = useCallback(() => {
        nextPlayRef.current = 0;
        startedRef.current = false;
    }, []);

    /**
     * Decode a base-64 WAV chunk and schedule it for gapless playback.
     * Returns the number of decoded chunks on success, or 0 on decode failure.
     */
    const scheduleChunk = useCallback(async (b64) => {
        const ctx = audioCtxRef.current;
        if (!ctx) return 0;

        try {
            const audioBuf = await ctx.decodeAudioData(base64ToArrayBuffer(b64));
            const LOOKAHEAD = 0.05; // 50 ms: absorbs one decode round-trip

            if (!startedRef.current) {
                nextPlayRef.current = ctx.currentTime + LOOKAHEAD;
                startedRef.current = true;
            } else if (nextPlayRef.current < ctx.currentTime) {
                // Fell behind — catch up without inserting silence.
                nextPlayRef.current = ctx.currentTime + LOOKAHEAD;
            }

            const src = ctx.createBufferSource();
            src.buffer = audioBuf;
            src.connect(ctx.destination);
            src.start(nextPlayRef.current);
            nextPlayRef.current += audioBuf.duration;
            return 1;
        } catch (e) {
            console.warn("decodeAudioData failed:", e);
            return 0;
        }
    }, []);

    return {
        audioRef,
        audioUrl,
        clearAudio,
        setAudioAndAutoplay,
        initAudioContext,
        resetScheduler,
        scheduleChunk,
    };
}
