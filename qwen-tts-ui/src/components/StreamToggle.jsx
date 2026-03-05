/**
 * components/StreamToggle.jsx
 * CSS pill toggle for enabling/disabling streaming mode.
 */
import { S } from "../styles";

export function StreamToggle({ streaming, onToggle, health }) {
    return (
        <div
            style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap",
                gap: 10,
            }}
        >
            <div style={S.toggleWrap} onClick={onToggle}>
                <div style={S.track(streaming)}>
                    <div style={S.thumb(streaming)} />
                </div>
                <span style={S.toggleLbl}>Streaming</span>
            </div>

            {health && (
                <span style={{ fontSize: 11, color: "#2a3050" }}>
                    chunk:{" "}
                    <span style={{ color: "#4a5070" }}>{health.model_chunk_text_len}c</span>
                    &nbsp;·&nbsp;pre-roll:{" "}
                    <span style={{ color: "#4a5070" }}>{health.pre_roll_chunks}</span>
                </span>
            )}
        </div>
    );
}
