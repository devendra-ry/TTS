/**
 * components/TopBar.jsx
 * Top navigation bar — logo and backend health indicator.
 */
import { S } from "../styles";

export function TopBar({ health }) {
    const online = !!health;
    return (
        <div style={S.topBar}>
            <span style={S.logo}>◈ Kokoro-82M TTS</span>
            <span style={S.health(online)}>
                <span style={S.dot(online)} />
                {online
                    ? `READY · ${health.vram?.device ?? "GPU"} · ${health.vram?.free_mb ?? "?"}MB free`
                    : "OFFLINE"}
            </span>
        </div>
    );
}
