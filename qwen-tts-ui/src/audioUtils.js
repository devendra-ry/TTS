/**
 * audioUtils.js
 * Pure audio-processing utilities — no React or API dependencies.
 */

/**
 * Decode a base-64 string into an ArrayBuffer suitable for Web Audio API.
 */
export function base64ToArrayBuffer(b64) {
    const raw = atob(b64);
    const buf = new ArrayBuffer(raw.length);
    const view = new Uint8Array(buf);
    for (let i = 0; i < raw.length; i++) view[i] = raw.charCodeAt(i);
    return buf;
}

/**
 * Reassemble all streamed per-chunk WAVs into one downloadable WAV Blob.
 * Finds the PCM data offset dynamically so variable-length headers are safe.
 *
 * @param {string[]} chunksB64 - Array of base-64 encoded WAV chunks
 * @param {number}   sr        - Sample rate for the output header
 * @returns {Blob|null}
 */
export function wavChunksToBlob(chunksB64, sr = 24000) {
    const toBytes = (b64) => {
        const raw = atob(b64);
        const out = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
        return out;
    };

    const getPcmOffset = (bytes) => {
        const view = new DataView(bytes.buffer);
        let pos = 12;
        while (pos + 8 <= bytes.length) {
            if (String.fromCharCode(...bytes.subarray(pos, pos + 4)) === "data")
                return pos + 8;
            pos += 8 + view.getUint32(pos + 4, true);
        }
        return 44; // safe fallback
    };

    const pcm = chunksB64
        .map(toBytes)
        .filter((b) => b.length > 8)
        .map((b) => b.subarray(getPcmOffset(b)));

    if (!pcm.length) return null;

    const dataSize = pcm.reduce((s, c) => s + c.length, 0);
    const out = new Uint8Array(44 + dataSize);
    const view = new DataView(out.buffer);

    const ws = (off, s) => {
        for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
    };

    ws(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    ws(8, "WAVE");
    ws(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);   // PCM
    view.setUint16(22, 1, true);   // mono
    view.setUint32(24, sr, true);
    view.setUint32(28, sr * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    ws(36, "data");
    view.setUint32(40, dataSize, true);

    let off = 44;
    pcm.forEach((c) => {
        out.set(c, off);
        off += c.length;
    });

    return new Blob([out], { type: "audio/wav" });
}
