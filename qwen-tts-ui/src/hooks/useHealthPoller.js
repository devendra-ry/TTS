/**
 * hooks/useHealthPoller.js
 * Custom hook — polls the /health endpoint every 8 seconds.
 * Returns the latest health object (or null if unreachable).
 */
import { useState, useEffect } from "react";
import { fetchHealth } from "../api";

export function useHealthPoller(intervalMs = 8000) {
    const [health, setHealth] = useState(null);

    useEffect(() => {
        const poll = async () => {
            try {
                setHealth(await fetchHealth());
            } catch {
                setHealth(null);
            }
        };

        poll();
        const id = setInterval(poll, intervalMs);
        return () => clearInterval(id);
    }, [intervalMs]);

    return health;
}
