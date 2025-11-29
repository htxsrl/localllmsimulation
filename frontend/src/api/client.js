const API_BASE = '/api/v1';

async function fetchApi(endpoint, options = {}) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
        ...options,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `API error: ${response.status}`);
    }

    return response.json();
}

export async function getHardware(params = {}) {
    const query = new URLSearchParams(params).toString();
    return fetchApi(`/hardware${query ? `?${query}` : ''}`);
}

export async function getModels(params = {}) {
    const query = new URLSearchParams(params).toString();
    return fetchApi(`/models${query ? `?${query}` : ''}`);
}

export async function getModelDetail(modelId) {
    return fetchApi(`/models/${modelId}`);
}

export async function simulate(data) {
    return fetchApi('/simulate', {
        method: 'POST',
        body: JSON.stringify(data),
    });
}

export async function searchHardwareForModel(quantizationId, minTps) {
    return fetchApi(`/search/hardware-for-model?quantization_id=${quantizationId}&min_tps=${minTps}`);
}

export async function searchModelsForHardware(params) {
    const query = new URLSearchParams(params).toString();
    return fetchApi(`/search/models-for-hardware?${query}`);
}

export async function getOcrTools() {
    return fetchApi('/ocr-tools');
}

export async function simulateOcr(data) {
    return fetchApi('/simulate-ocr', {
        method: 'POST',
        body: JSON.stringify(data),
    });
}

export async function getBenchmarks() {
    return fetchApi('/benchmarks');
}
