/**
 * components/VoiceSelector.jsx
 * Controlled <select> for choosing a Kokoro speaker voice.
 */
import { S } from "../styles";

const DEFAULT_SPEAKER_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

export function VoiceSelector({ health, speakerId, onChange }) {
    const speakerRange = health?.speaker_id_range;
    const speakerIds =
        Array.isArray(speakerRange) && speakerRange.length === 2
            ? Array.from(
                { length: speakerRange[1] - speakerRange[0] + 1 },
                (_, i) => speakerRange[0] + i
            )
            : DEFAULT_SPEAKER_IDS;

    const options = speakerIds.map((id) => ({
        id,
        label: health?.voice_labels?.[String(id)] ?? `Voice ${id}`,
    }));

    return (
        <div style={{ ...S.col, maxWidth: 220 }}>
            <label style={S.label}>
                Speaker ID
                <span style={{ color: "#2a3050", marginLeft: 6 }}>0–9</span>
            </label>
            <select
                style={S.select}
                value={speakerId}
                onChange={(e) => onChange(Number(e.target.value))}
            >
                {options.map(({ id, label }) => (
                    <option key={id} value={id}>
                        {`${id} · ${label}`}
                    </option>
                ))}
            </select>
        </div>
    );
}
