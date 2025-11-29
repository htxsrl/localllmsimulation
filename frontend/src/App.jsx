import { useState, useEffect, useRef } from 'react';
import { getHardware, getModels, getModelDetail, getOcrTools } from './api/client';
import { useSimulation } from './hooks/useSimulation';
import { useHardwareSearch } from './hooks/useHardwareSearch';
import { useTokenAnimation } from './hooks/useTokenAnimation';
import { useOcrSimulation } from './hooks/useOcrSimulation';
import ConfidenceExplainer from './components/ConfidenceExplainer';
import SourcesModal from './components/SourcesModal';

function App() {
    const [mode, setMode] = useState('hardware');
    const [hardware, setHardware] = useState([]);
    const [models, setModels] = useState([]);
    const [ocrTools, setOcrTools] = useState([]);
    const [loadingData, setLoadingData] = useState(true);

    // Hardware mode state
    const [selectedHardware, setSelectedHardware] = useState('');
    const [isCustomHardware, setIsCustomHardware] = useState(false);
    const [customHardware, setCustomHardware] = useState({
        name: '',
        memory_gb: 64,
        memory_bandwidth_gbs: 400,
        gpu_cores: '',
        memory_type: 'unified'
    });
    const [selectedFamily, setSelectedFamily] = useState('');
    const [selectedSize, setSelectedSize] = useState('');
    const [selectedQuant, setSelectedQuant] = useState('');
    const [quantizations, setQuantizations] = useState([]);
    const [promptTokens, setPromptTokens] = useState(8000);
    const [showCountdown, setShowCountdown] = useState(false);
    const [showConfidenceExplainer, setShowConfidenceExplainer] = useState(false);
    const [showSourcesModal, setShowSourcesModal] = useState(false);
    const [isCustomModel, setIsCustomModel] = useState(false);
    const [customModel, setCustomModel] = useState({
        name: '',
        size_b: 7,
        size_gb: 4,
        is_moe: false
    });

    // Model mode state
    const [targetFamily, setTargetFamily] = useState('');
    const [targetSize, setTargetSize] = useState('');
    const [targetQuant, setTargetQuant] = useState('');
    const [targetQuantizations, setTargetQuantizations] = useState([]);
    const [targetTps, setTargetTps] = useState(30);

    // OCR mode state
    const [ocrHardware, setOcrHardware] = useState('');
    const [selectedOcrTool, setSelectedOcrTool] = useState('');
    const [pageCount, setPageCount] = useState(12);
    const [documentType, setDocumentType] = useState('mixed');

    // Track if defaults have been set
    const [defaultsLoaded, setDefaultsLoaded] = useState(false);

    // Animation restart key - increments each time simulate is clicked
    const [animationKey, setAnimationKey] = useState(0);

    const { loading: simLoading, error: simError, result: simResult, runSimulation } = useSimulation();
    const { loading: searchLoading, error: searchError, result: searchResult, search } = useHardwareSearch();
    const { loading: ocrLoading, error: ocrError, result: ocrResult, runOcrSimulation } = useOcrSimulation();
    const { phase, displayed, ttftRemaining, stop } = useTokenAnimation(
        simResult?.generation_tps,
        simResult?.ttft_ms,
        simResult?.can_run,
        animationKey
    );

    // Ref for hardware list scrolling
    const hardwareListRef = useRef(null);
    const bestMatchRef = useRef(null);
    const resultCardRef = useRef(null);
    const tokenStreamRef = useRef(null);

    // Auto-scroll to best match when search results change
    useEffect(() => {
        if (searchResult && bestMatchRef.current && hardwareListRef.current) {
            setTimeout(() => {
                bestMatchRef.current?.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center'
                });
            }, 100);
        }
    }, [searchResult]);

    // Auto-scroll to result card when simulation completes
    useEffect(() => {
        if (simResult && resultCardRef.current && animationKey > 0) {
            setTimeout(() => {
                resultCardRef.current?.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }, 100);
        }
    }, [simResult, animationKey]);

    // Auto-scroll token stream to bottom as text grows
    useEffect(() => {
        if (tokenStreamRef.current) {
            tokenStreamRef.current.scrollTop = tokenStreamRef.current.scrollHeight;
        }
    }, [displayed]);

    useEffect(() => {
        Promise.all([getHardware(), getModels(), getOcrTools()])
            .then(([hw, mdl, ocr]) => {
                setHardware(hw);
                setModels(mdl);
                setOcrTools(ocr);
                setLoadingData(false);
            })
            .catch(err => {
                console.error('Failed to load data:', err);
                setLoadingData(false);
            });
    }, []);

    // Set defaults once data is loaded (only for hardware mode, model mode copies on switch)
    useEffect(() => {
        if (loadingData || defaultsLoaded || hardware.length === 0 || models.length === 0) return;

        const setDefaultsAsync = async () => {
            // Find a good default hardware (MacBook Pro M4 Max 64GB or first Apple)
            const defaultHw = hardware.find(h => h.name.includes('MacBook Pro M4 Max 64GB'))
                || hardware.find(h => h.vendor === 'Apple')
                || hardware[0];

            if (defaultHw) {
                setSelectedHardware(String(defaultHw.id));
            }

            // Find Llama 3.1 8B or similar for hardware mode
            const llama = models.find(m => m.family === 'Llama 3.1' && m.size_b === 8)
                || models.find(m => m.family === 'Llama 3.1')
                || models.find(m => m.family.includes('Llama'))
                || models[0];

            if (llama) {
                setSelectedFamily(llama.family);
                setSelectedSize(String(llama.id));
                const detail = await getModelDetail(llama.id);
                setQuantizations(detail.quantizations);
                // Select Q4_K_M or first quantization
                const defaultQuant = detail.quantizations.find(q => q.quant_type === 'Q4_K_M')
                    || detail.quantizations[0];
                if (defaultQuant) {
                    setSelectedQuant(String(defaultQuant.id));
                }
            }

            setDefaultsLoaded(true);
        };

        setDefaultsAsync();
    }, [loadingData, defaultsLoaded, hardware, models]);

    const families = [...new Set(models.map(m => m.family))];

    const getSizesForFamily = (family) => {
        return models
            .filter(m => m.family === family)
            .map(m => ({ id: m.id, size: m.size_b, variant: m.variant }))
            .sort((a, b) => a.size - b.size);
    };

    // Get available memory for current hardware selection
    const getAvailableMemory = () => {
        if (isCustomHardware) {
            return parseFloat(customHardware.memory_gb) || 0;
        }
        if (selectedHardware) {
            const hw = hardware.find(h => h.id === parseInt(selectedHardware));
            return hw?.memory_gb || 0;
        }
        return 0;
    };

    // Get memory indicator for a quantization
    const getMemoryIndicator = (sizeGb) => {
        const available = getAvailableMemory();
        if (available === 0) return '';

        if (sizeGb <= available * 0.85) return ' ✓';
        if (sizeGb <= available) return ' ⚠️';
        return ' ✗';
    };

    const handleFamilyChange = async (family, isTarget = false) => {
        if (isTarget) {
            setTargetFamily(family);
            setTargetSize('');
            setTargetQuant('');
            setTargetQuantizations([]);
        } else {
            setSelectedFamily(family);
            setSelectedSize('');
            setSelectedQuant('');
            setQuantizations([]);
        }
    };

    const handleSizeChange = async (modelId, isTarget = false) => {
        if (isTarget) {
            setTargetSize(modelId);
            setTargetQuant('');
            if (modelId) {
                const detail = await getModelDetail(modelId);
                setTargetQuantizations(detail.quantizations);
            } else {
                setTargetQuantizations([]);
            }
        } else {
            setSelectedSize(modelId);
            setSelectedQuant('');
            if (modelId) {
                const detail = await getModelDetail(modelId);
                setQuantizations(detail.quantizations);
            } else {
                setQuantizations([]);
            }
        }
    };

    const handleSimulate = () => {
        if (!isCustomModel && !selectedQuant) return;
        if (!isCustomHardware && !selectedHardware) return;

        // Increment animation key to restart the animation
        setAnimationKey(k => k + 1);

        const payload = {
            prompt_tokens: promptTokens,
        };

        if (isCustomHardware) {
            payload.custom_hardware = {
                memory_gb: parseFloat(customHardware.memory_gb),
                memory_bandwidth_gbs: parseFloat(customHardware.memory_bandwidth_gbs),
                gpu_cores: customHardware.gpu_cores ? parseInt(customHardware.gpu_cores) : null,
                memory_type: customHardware.memory_type
            };
        } else {
            payload.hardware_id = parseInt(selectedHardware);
        }

        if (isCustomModel) {
            payload.custom_model = {
                size_gb: parseFloat(customModel.size_gb),
                bits_per_weight: 4.5,
                is_moe: customModel.is_moe
            };
        } else {
            payload.quantization_id = parseInt(selectedQuant);
        }

        runSimulation(payload);
    };

    const handleSearch = () => {
        if (!targetQuant) return;
        search(parseInt(targetQuant), targetTps);
    };

    const handleOcrSimulate = () => {
        if (!ocrHardware || !selectedOcrTool) return;
        runOcrSimulation({
            hardware_id: parseInt(ocrHardware),
            tool_id: parseInt(selectedOcrTool),
            page_count: pageCount,
            document_type: documentType
        });
    };

    const formatTime = (seconds) => {
        if (seconds < 60) return `${seconds.toFixed(1)}s`;
        const mins = Math.floor(seconds / 60);
        const secs = Math.round(seconds % 60);
        return `${mins}m ${secs}s`;
    };

    const groupByVendor = (items) => {
        return items.reduce((acc, item) => {
            const vendor = item.hardware?.vendor || item.vendor;
            if (!acc[vendor]) acc[vendor] = [];
            acc[vendor].push(item);
            return acc;
        }, {});
    };

    if (loadingData) {
        return <div className="app"><div className="container loading">Loading...</div></div>;
    }

    return (
        <div className="app">
            <div className="container">
                <header className="header">
                    <h1>Local LLM Performance Simulator</h1>
                    <p>Estimate inference speed on your hardware — Based on 207 real benchmarks</p>
                </header>

                <div className="mode-switch">
                    <button
                        className={mode === 'hardware' ? 'active' : ''}
                        onClick={() => {
                            if (mode === 'model') {
                                setSelectedFamily(targetFamily);
                                setSelectedSize(targetSize);
                                setSelectedQuant(targetQuant);
                                setQuantizations(targetQuantizations);
                            }
                            setMode('hardware');
                        }}
                    >
                        <div className="mode-title">I have hardware</div>
                        <div className="mode-subtitle">What can it run?</div>
                    </button>
                    <button
                        className={mode === 'model' ? 'active' : ''}
                        onClick={() => {
                            if (mode === 'hardware') {
                                setTargetFamily(selectedFamily);
                                setTargetSize(selectedSize);
                                setTargetQuant(selectedQuant);
                                setTargetQuantizations(quantizations);
                            }
                            setMode('model');
                        }}
                    >
                        <div className="mode-title">I want a model</div>
                        <div className="mode-subtitle">What do I need?</div>
                    </button>
                </div>

                {mode === 'hardware' && (
                    <>
                        <div className="card">
                            <div className="form-group">
                                <label>Your hardware</label>
                                <select
                                    value={selectedHardware}
                                    onChange={e => setSelectedHardware(e.target.value)}
                                    disabled={isCustomHardware}
                                >
                                    <option value="">Select hardware...</option>
                                    {Object.entries(groupByVendor(hardware)).map(([vendor, items]) => (
                                        <optgroup key={vendor} label={vendor}>
                                            {items.map(hw => (
                                                <option key={hw.id} value={hw.id}>
                                                    {hw.name}
                                                </option>
                                            ))}
                                        </optgroup>
                                    ))}
                                </select>
                                <label className="checkbox-label">
                                    <input
                                        type="checkbox"
                                        checked={isCustomHardware}
                                        onChange={e => {
                                            const checked = e.target.checked;
                                            if (checked && selectedHardware) {
                                                const hw = hardware.find(h => h.id === parseInt(selectedHardware));
                                                if (hw) {
                                                    setCustomHardware({
                                                        name: hw.name,
                                                        memory_gb: hw.memory_gb,
                                                        memory_bandwidth_gbs: hw.memory_bandwidth_gbs,
                                                        gpu_cores: hw.gpu_cores || '',
                                                        memory_type: hw.memory_type
                                                    });
                                                }
                                            }
                                            setIsCustomHardware(checked);
                                        }}
                                    />
                                    Custom specs
                                </label>
                            </div>

                            {isCustomHardware && (
                                <div className="custom-hardware-form">
                                    <div className="form-row">
                                        <div className="form-field">
                                            <label>Name (optional)</label>
                                            <input
                                                type="text"
                                                value={customHardware.name}
                                                onChange={e => setCustomHardware({...customHardware, name: e.target.value})}
                                                placeholder="M5 Ultra 512GB"
                                            />
                                        </div>
                                    </div>
                                    <div className="form-row">
                                        <div className="form-field">
                                            <label>RAM/VRAM</label>
                                            <div className="input-with-unit">
                                                <input
                                                    type="number"
                                                    value={customHardware.memory_gb}
                                                    onChange={e => setCustomHardware({...customHardware, memory_gb: e.target.value})}
                                                    min="1"
                                                    max="2048"
                                                />
                                                <span>GB</span>
                                            </div>
                                        </div>
                                        <div className="form-field">
                                            <label>Bandwidth</label>
                                            <div className="input-with-unit">
                                                <input
                                                    type="number"
                                                    value={customHardware.memory_bandwidth_gbs}
                                                    onChange={e => setCustomHardware({...customHardware, memory_bandwidth_gbs: e.target.value})}
                                                    min="1"
                                                    max="10000"
                                                />
                                                <span>GB/s</span>
                                            </div>
                                        </div>
                                        <div className="form-field">
                                            <label>GPU Cores</label>
                                            <input
                                                type="number"
                                                value={customHardware.gpu_cores}
                                                onChange={e => setCustomHardware({...customHardware, gpu_cores: e.target.value})}
                                                placeholder="optional"
                                                min="0"
                                            />
                                        </div>
                                    </div>
                                    <div className="form-row">
                                        <div className="form-field">
                                            <label>Memory type</label>
                                            <div className="radio-group">
                                                <label>
                                                    <input
                                                        type="radio"
                                                        name="memoryType"
                                                        checked={customHardware.memory_type === 'unified'}
                                                        onChange={() => setCustomHardware({...customHardware, memory_type: 'unified'})}
                                                    />
                                                    Unified (Apple Silicon)
                                                </label>
                                                <label>
                                                    <input
                                                        type="radio"
                                                        name="memoryType"
                                                        checked={customHardware.memory_type === 'vram'}
                                                        onChange={() => setCustomHardware({...customHardware, memory_type: 'vram'})}
                                                    />
                                                    Dedicated VRAM (discrete GPU)
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            <div className="form-group">
                                <label>Model to test</label>
                                <div className="select-row">
                                    <select
                                        value={selectedFamily}
                                        onChange={e => handleFamilyChange(e.target.value)}
                                        disabled={isCustomModel}
                                    >
                                        <option value="">Family</option>
                                        {families.map(f => (
                                            <option key={f} value={f}>{f}</option>
                                        ))}
                                    </select>
                                    <select
                                        value={selectedSize}
                                        onChange={e => handleSizeChange(e.target.value)}
                                        disabled={!selectedFamily || isCustomModel}
                                    >
                                        <option value="">Size</option>
                                        {getSizesForFamily(selectedFamily).map(s => (
                                            <option key={s.id} value={s.id}>
                                                {s.size}B{s.variant ? ` (${s.variant})` : ''}
                                            </option>
                                        ))}
                                    </select>
                                    <select
                                        value={selectedQuant}
                                        onChange={e => setSelectedQuant(e.target.value)}
                                        disabled={!selectedSize || isCustomModel}
                                    >
                                        <option value="">Quant</option>
                                        {quantizations.map(q => (
                                            <option key={q.id} value={q.id}>
                                                {q.quant_type} ({q.size_gb}GB){getMemoryIndicator(q.size_gb)}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                <label className="checkbox-label">
                                    <input
                                        type="checkbox"
                                        checked={isCustomModel}
                                        onChange={e => setIsCustomModel(e.target.checked)}
                                    />
                                    Custom model
                                </label>
                            </div>

                            {isCustomModel && (
                                <div className="custom-hardware-form">
                                    <div className="form-row">
                                        <div className="form-field">
                                            <label>Name (optional)</label>
                                            <input
                                                type="text"
                                                value={customModel.name}
                                                onChange={e => setCustomModel({...customModel, name: e.target.value})}
                                                placeholder="GPT-5 70B Q4"
                                            />
                                        </div>
                                    </div>
                                    <div className="form-row">
                                        <div className="form-field">
                                            <label>Model size on disk</label>
                                            <div className="input-with-unit">
                                                <input
                                                    type="number"
                                                    value={customModel.size_gb}
                                                    onChange={e => setCustomModel({...customModel, size_gb: e.target.value})}
                                                    min="0.1"
                                                    max="1000"
                                                    step="0.1"
                                                />
                                                <span>GB</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="form-row">
                                        <div className="form-field">
                                            <label>Architecture</label>
                                            <div className="radio-group">
                                                <label>
                                                    <input
                                                        type="radio"
                                                        name="modelArch"
                                                        checked={!customModel.is_moe}
                                                        onChange={() => setCustomModel({...customModel, is_moe: false})}
                                                    />
                                                    Dense
                                                </label>
                                                <label>
                                                    <input
                                                        type="radio"
                                                        name="modelArch"
                                                        checked={customModel.is_moe}
                                                        onChange={() => setCustomModel({...customModel, is_moe: true})}
                                                    />
                                                    MoE (Mixture of Experts)
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            <div className="form-group">
                                <label>Context length</label>
                                <div className="radio-group">
                                    {[
                                        { value: 2000, label: 'Short (2K)' },
                                        { value: 8000, label: 'Medium (8K)' },
                                        { value: 32000, label: 'Large (32K)' },
                                        { value: 1000000, label: 'Max (1M)' },
                                    ].map(opt => (
                                        <label key={opt.value}>
                                            <input
                                                type="radio"
                                                name="promptTokens"
                                                checked={promptTokens === opt.value}
                                                onChange={() => setPromptTokens(opt.value)}
                                            />
                                            {opt.label}
                                        </label>
                                    ))}
                                </div>
                                <p className="context-note">
                                    Context length affects memory usage (KV cache). TTFT is calculated for a typical 500-token prompt.
                                    Longer prompts increase TTFT linearly: <em>TTFT = 50ms + (tokens / prompt_speed)</em>
                                </p>
                            </div>

                            <button
                                className="btn"
                                onClick={handleSimulate}
                                disabled={simLoading || (isCustomModel ? false : !selectedQuant) || (isCustomHardware ? false : !selectedHardware)}
                            >
                                {simLoading ? 'Simulating...' : 'Simulate Experience'}
                            </button>
                        </div>

                        {simError && (
                            <div className="error-message">{simError}</div>
                        )}

                        {simResult && (
                            <div className="result-card" ref={resultCardRef}>
                                <div className={`result-header ${simResult.can_run ? 'success' : 'error'}`}>
                                    {simResult.can_run ? '✓' : '✗'}
                                    {simResult.can_run
                                        ? `Model and prompt fit in memory (${simResult.memory_required_gb}GB / ${simResult.memory_available_gb}GB)`
                                        : `Not enough memory (need ${simResult.memory_required_gb}GB, have ${simResult.memory_available_gb}GB)`
                                    }
                                </div>

                                {simResult.can_run && (
                                    <>
                                        <div className="result-stats">
                                            <div className="stat">
                                                <div className="stat-label">Time to first token</div>
                                                <div className="stat-value">
                                                    {(simResult.ttft_ms / 1000).toFixed(1)} <small>s</small>
                                                </div>
                                            </div>
                                            <div className="stat">
                                                <div className="stat-label">Generation speed</div>
                                                <div className="stat-value">
                                                    {simResult.generation_tps.toFixed(1)} <small>tok/s</small>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="chat-container">
                                            <div className="token-stream" ref={tokenStreamRef}>
                                                {phase === 'ttft' && (
                                                    <div className="ttft-waiting">
                                                        <div className="ttft-spinner" />
                                                        <span>
                                                            {showCountdown
                                                                ? `Waiting for first token... ${(ttftRemaining / 1000).toFixed(1)}s`
                                                                : 'Waiting for first token...'
                                                            }
                                                        </span>
                                                    </div>
                                                )}
                                                {displayed}
                                                {phase === 'streaming' && <span className="cursor" />}
                                            </div>
                                            <div className="chat-input-bar">
                                                <div className="chat-input-placeholder">
                                                    Explore the deep connections between Pirsig's concept of Quality, Faggin's theory of qualia and consciousness, quantum physics' observer effect, and indigenous wisdom like the Lakota's Mitakuye Oyasin...
                                                </div>
                                                {(phase === 'ttft' || phase === 'streaming') ? (
                                                    <button
                                                        className="stop-btn active"
                                                        onClick={stop}
                                                    >
                                                        <span className="stop-icon" />
                                                        Stop
                                                    </button>
                                                ) : (
                                                    <button
                                                        className="simulate-btn"
                                                        onClick={handleSimulate}
                                                        disabled={simLoading}
                                                    >
                                                        Simulate
                                                    </button>
                                                )}
                                            </div>
                                        </div>

                                        <label className="countdown-toggle">
                                            <input
                                                type="checkbox"
                                                checked={showCountdown}
                                                onChange={e => setShowCountdown(e.target.checked)}
                                            />
                                            Show countdown to the first token
                                        </label>

                                        <div className="result-info">
                                            {simResult.is_measured ? '📊 Based on real benchmark' : '📈 Estimated'}
                                            {' • '}
                                            <button
                                                className="confidence-link"
                                                onClick={() => setShowConfidenceExplainer(true)}
                                            >
                                                Confidence: {Math.round(simResult.confidence * 100)}%
                                            </button>
                                        </div>

                                        {simResult.warnings.length > 0 && (
                                            <div className="warnings">
                                                {simResult.warnings.join(' • ')}
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        )}
                    </>
                )}

                {mode === 'model' && (
                    <>
                        <div className="card">
                            <div className="form-group">
                                <label>Model you want to run</label>
                                <div className="select-row">
                                    <select
                                        value={targetFamily}
                                        onChange={e => handleFamilyChange(e.target.value, true)}
                                    >
                                        <option value="">Family</option>
                                        {families.map(f => (
                                            <option key={f} value={f}>{f}</option>
                                        ))}
                                    </select>
                                    <select
                                        value={targetSize}
                                        onChange={e => handleSizeChange(e.target.value, true)}
                                        disabled={!targetFamily}
                                    >
                                        <option value="">Size</option>
                                        {getSizesForFamily(targetFamily).map(s => (
                                            <option key={s.id} value={s.id}>
                                                {s.size}B{s.variant ? ` (${s.variant})` : ''}
                                            </option>
                                        ))}
                                    </select>
                                    <select
                                        value={targetQuant}
                                        onChange={e => setTargetQuant(e.target.value)}
                                        disabled={!targetSize}
                                    >
                                        <option value="">Quant</option>
                                        {targetQuantizations.map(q => (
                                            <option key={q.id} value={q.id}>
                                                {q.quant_type} ({q.size_gb}GB)
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            </div>

                            <div className="form-group">
                                <label>Minimum speed</label>
                                <div className="slider-value">
                                    {targetTps} <small>tok/s</small>
                                </div>
                                <div className="slider-container">
                                    <input
                                        type="range"
                                        min="10"
                                        max="150"
                                        value={targetTps}
                                        onChange={e => setTargetTps(parseInt(e.target.value))}
                                    />
                                    <div className="slider-labels">
                                        <span>10 (slow)</span>
                                        <span>30 (fluent)</span>
                                        <span>150 (instant)</span>
                                    </div>
                                </div>
                            </div>

                            <button
                                className="btn"
                                onClick={handleSearch}
                                disabled={searchLoading || !targetQuant}
                            >
                                {searchLoading ? 'Searching...' : 'Search'}
                            </button>
                        </div>

                        {searchError && (
                            <div className="error-message">{searchError}</div>
                        )}

                        {searchResult && (
                            <div className="result-card">
                                <div className="min-requirements">
                                    <h3>Minimum requirements for {targetTps} tok/s</h3>
                                    <ul>
                                        <li>• {searchResult.min_requirements.memory_gb} GB RAM/VRAM</li>
                                        <li>• ~{searchResult.min_requirements.bandwidth_gbs} GB/s memory bandwidth</li>
                                    </ul>
                                </div>

                                <h3 style={{ marginBottom: 16 }}>Hardware ranked by speed</h3>
                                <div className="hardware-list-scrollable" ref={hardwareListRef}>
                                    {(() => {
                                        // Combine compatible and incompatible, sort by TPS descending
                                        const allHardware = [
                                            ...searchResult.compatible.map(item => ({
                                                ...item,
                                                type: 'compatible'
                                            })),
                                            ...searchResult.incompatible
                                                .filter(item => item.reason === 'speed')
                                                .map(item => ({
                                                    hardware: item.hardware,
                                                    estimated_tps: 0, // Too slow to calculate
                                                    meets_target: false,
                                                    type: 'too_slow'
                                                })),
                                            ...searchResult.incompatible
                                                .filter(item => item.reason === 'memory')
                                                .map(item => ({
                                                    hardware: item.hardware,
                                                    estimated_tps: null,
                                                    meets_target: false,
                                                    type: 'no_memory',
                                                    shortfall_gb: item.shortfall_gb
                                                }))
                                        ].sort((a, b) => (b.estimated_tps || 0) - (a.estimated_tps || 0));

                                        // Find the best match (closest to target TPS from above)
                                        const bestMatchIndex = allHardware.findIndex(item => 
                                            item.type === 'compatible' && 
                                            item.estimated_tps >= targetTps &&
                                            item.estimated_tps <= targetTps * 1.5
                                        );
                                        const bestMatch = bestMatchIndex !== -1 ? bestMatchIndex : 
                                            allHardware.findIndex(item => item.type === 'compatible' && item.meets_target);

                                        return allHardware.map((item, idx) => {
                                            const isBestMatch = idx === bestMatch;
                                            const isAboveTarget = item.type === 'compatible' && item.estimated_tps > targetTps * 1.5;
                                            const isBelowTarget = item.type === 'compatible' && !item.meets_target;
                                            
                                            return (
                                                <div
                                                    key={idx}
                                                    ref={isBestMatch ? bestMatchRef : null}
                                                    className={`hardware-item ${
                                                        item.type === 'no_memory' ? 'no-memory' :
                                                        item.type === 'too_slow' ? 'insufficient' :
                                                        isBestMatch ? 'best-match' :
                                                        isAboveTarget ? 'above-target' :
                                                        isBelowTarget ? 'below-target' : ''
                                                    }`}
                                                >
                                                    <div className="hardware-item-left">
                                                        {isBestMatch && <span className="best-match-badge">Best Match</span>}
                                                        <span className="hardware-item-name">
                                                            {item.hardware.name}
                                                        </span>
                                                    </div>
                                                    <div className="hardware-item-tps">
                                                        {item.type === 'no_memory' ? (
                                                            <span style={{ color: 'var(--color-error)', fontSize: 13 }}>
                                                                Not enough memory (-{item.shortfall_gb}GB)
                                                            </span>
                                                        ) : item.type === 'too_slow' ? (
                                                            <span style={{ color: 'var(--color-error)', fontSize: 13 }}>
                                                                Too slow
                                                            </span>
                                                        ) : (
                                                            <>
                                                                <span>{item.estimated_tps} tok/s</span>
                                                                <div className="tps-bar">
                                                                    <div
                                                                        className={`tps-bar-fill ${isBestMatch ? 'best-match' : ''}`}
                                                                        style={{
                                                                            width: `${Math.min(100, (item.estimated_tps / 150) * 100)}%`
                                                                        }}
                                                                    />
                                                                    <div 
                                                                        className="tps-target-line"
                                                                        style={{
                                                                            left: `${Math.min(100, (targetTps / 150) * 100)}%`
                                                                        }}
                                                                    />
                                                                </div>
                                                            </>
                                                        )}
                                                    </div>
                                                </div>
                                            );
                                        });
                                    })()}
                                </div>
                            </div>
                        )}
                    </>
                )}

                <footer className="footer">
                    <p>
                        <a href="https://ht-x.com" target="_blank" rel="noopener">Human Technology eXcellence</a>
                        {' | HTX SRL | PIVA: IT01407090321 | '}
                        <button
                            className="footer-sources-link"
                            onClick={() => setShowSourcesModal(true)}
                        >
                            Data sources
                        </button>
                    </p>
                </footer>
            </div>

            {showConfidenceExplainer && simResult && (
                <ConfidenceExplainer
                    simResult={simResult}
                    onClose={() => setShowConfidenceExplainer(false)}
                />
            )}

            {showSourcesModal && (
                <SourcesModal onClose={() => setShowSourcesModal(false)} />
            )}
        </div>
    );
}

export default App;
