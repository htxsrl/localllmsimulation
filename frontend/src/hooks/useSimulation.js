import { useState, useCallback } from 'react';
import { simulate } from '../api/client';

export function useSimulation() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);

    const runSimulation = useCallback(async (params) => {
        setLoading(true);
        setError(null);

        try {
            const data = await simulate(params);
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

    return { loading, error, result, runSimulation, reset };
}
