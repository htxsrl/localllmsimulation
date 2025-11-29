import { useState, useCallback } from 'react';
import { searchHardwareForModel } from '../api/client';

export function useHardwareSearch() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);

    const search = useCallback(async (quantizationId, minTps) => {
        setLoading(true);
        setError(null);

        try {
            const data = await searchHardwareForModel(quantizationId, minTps);
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

    return { loading, error, result, search, reset };
}
