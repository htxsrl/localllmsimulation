import { useState, useCallback } from 'react';
import { simulateOcr } from '../api/client';

export function useOcrSimulation() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);

    const runOcrSimulation = useCallback(async (params) => {
        setLoading(true);
        setError(null);

        try {
            const data = await simulateOcr(params);
            setResult(data);
        } catch (err) {
            setError(err.message);
            setResult(null);
        } finally {
            setLoading(false);
        }
    }, []);

    const reset = useCallback(() => {
        setResult(null);
        setError(null);
    }, []);

    return { loading, error, result, runOcrSimulation, reset };
}
