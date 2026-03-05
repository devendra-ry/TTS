/**
 * components/AudioOutput.jsx
 * Renders the <audio> player and TTFA/RTF performance metrics
 * once audio is available.
 */
import { S } from "../styles";

export function AudioOutput({ audioRef, audioUrl, metrics }) {
    if (!audioUrl) return null;

    return (
        <div style={S.panel}>
            <label style={S.label}>Output</label>
            <div style={S.audioWrap}>
                <audio ref={audioRef} src={audioUrl} controls preload="metadata" />
            </div>

            {metrics && (
                <div style={S.metrics}>
                    <span style={S.metric}>
                        TTFA: <span style={S.mVal}>{metrics.ttfa_ms}ms</span>
                    </span>
                    <span style={S.metric}>
                        RTF: <span style={S.mVal}>{metrics.rtf}×</span>
                    </span>
                </div>
            )}
        </div>
    );
}
