import { useMemo, useState, useEffect } from 'react';

const FEATURE_LABELS = {
    'memory_bandwidth_gbs': 'Memory Bandwidth',
    'model_size_gb': 'Model Size',
    'context_length': 'Context Length',
    'theoretical_tps': 'Theoretical Speed',
    'mem_utilization': 'Memory Usage %',
    'bw_per_gb': 'Bandwidth/Model Ratio',
    'gpu_cores': 'GPU Cores',
    'is_unified': 'Unified Memory',
    'is_moe': 'MoE Architecture',
    'bits_per_weight': 'Bits per Weight',
};

function ConfidenceExplainer({ simResult, onClose }) {
    const [modelInfo, setModelInfo] = useState(null);

    useEffect(() => {
        fetch('/api/v1/model-info')
            .then(r => r.json())
            .then(setModelInfo)
            .catch(() => setModelInfo(null));
    }, []);

    const chartData = useMemo(() => {
        if (!simResult || !simResult.nearest_benchmarks?.length) return null;

        const estimated = {
            tps: simResult.generation_tps,
            ttft: simResult.ttft_ms,
            isEstimated: true
        };

        const benchmarks = simResult.nearest_benchmarks.map(b => ({
            tps: b.generation_tps,
            ttft: b.ttft_ms,
            hardware: b.hardware_name,
            model: b.model_name,
            quant: b.quant_type,
            source: b.source
        }));

        const allPoints = [estimated, ...benchmarks];
        const tpsValues = allPoints.map(p => p.tps);
        const ttftValues = allPoints.map(p => p.ttft);

        const tpsMin = Math.min(...tpsValues) * 0.8;
        const tpsMax = Math.max(...tpsValues) * 1.2;
        const ttftMin = Math.min(...ttftValues) * 0.8;
        const ttftMax = Math.max(...ttftValues) * 1.2;

        return {
            estimated,
            benchmarks,
            tpsRange: [tpsMin, tpsMax],
            ttftRange: [ttftMin, ttftMax]
        };
    }, [simResult]);

    if (!chartData) return null;

    const { estimated, benchmarks, tpsRange, ttftRange } = chartData;

    const chartWidth = 400;
    const chartHeight = 300;
    const padding = { top: 30, right: 30, bottom: 50, left: 70 };
    const innerWidth = chartWidth - padding.left - padding.right;
    const innerHeight = chartHeight - padding.top - padding.bottom;

    const scaleX = (tps) => {
        return padding.left + ((tps - tpsRange[0]) / (tpsRange[1] - tpsRange[0])) * innerWidth;
    };

    const scaleY = (ttft) => {
        return padding.top + innerHeight - ((ttft - ttftRange[0]) / (ttftRange[1] - ttftRange[0])) * innerHeight;
    };

    const formatMs = (ms) => {
        if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
        return `${Math.round(ms)}ms`;
    };

    const confidencePercent = Math.round(simResult.confidence * 100);

    return (
        <div className="confidence-modal-overlay" onClick={onClose}>
            <div className="confidence-modal" onClick={e => e.stopPropagation()}>
                <button className="close-btn" onClick={onClose}>×</button>
                <h2>Confidence: {confidencePercent}%</h2>

                <div className="chart-container">
                    <svg width={chartWidth} height={chartHeight} viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
                        {/* Grid lines */}
                        {[0, 0.25, 0.5, 0.75, 1].map(t => {
                            const x = padding.left + t * innerWidth;
                            const y = padding.top + t * innerHeight;
                            return (
                                <g key={t}>
                                    <line
                                        x1={x} y1={padding.top}
                                        x2={x} y2={padding.top + innerHeight}
                                        stroke="#e5e7eb" strokeWidth="1"
                                    />
                                    <line
                                        x1={padding.left} y1={y}
                                        x2={padding.left + innerWidth} y2={y}
                                        stroke="#e5e7eb" strokeWidth="1"
                                    />
                                </g>
                            );
                        })}

                        {/* X axis label */}
                        <text
                            x={padding.left + innerWidth / 2}
                            y={chartHeight - 10}
                            textAnchor="middle"
                            fontSize="12"
                            fill="#6b7280"
                        >
                            Generation Speed (tok/s)
                        </text>

                        {/* Y axis label */}
                        <text
                            x={15}
                            y={padding.top + innerHeight / 2}
                            textAnchor="middle"
                            fontSize="12"
                            fill="#6b7280"
                            transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}
                        >
                            Time to First Token
                        </text>

                        {/* X axis ticks */}
                        {[0, 0.5, 1].map(t => {
                            const val = tpsRange[0] + t * (tpsRange[1] - tpsRange[0]);
                            return (
                                <text
                                    key={`x-${t}`}
                                    x={padding.left + t * innerWidth}
                                    y={padding.top + innerHeight + 20}
                                    textAnchor="middle"
                                    fontSize="10"
                                    fill="#9ca3af"
                                >
                                    {val.toFixed(0)}
                                </text>
                            );
                        })}

                        {/* Y axis ticks */}
                        {[0, 0.5, 1].map(t => {
                            const val = ttftRange[0] + (1 - t) * (ttftRange[1] - ttftRange[0]);
                            return (
                                <text
                                    key={`y-${t}`}
                                    x={padding.left - 10}
                                    y={padding.top + t * innerHeight + 4}
                                    textAnchor="end"
                                    fontSize="10"
                                    fill="#9ca3af"
                                >
                                    {formatMs(val)}
                                </text>
                            );
                        })}

                        {/* Lines from estimated to each benchmark */}
                        {benchmarks.map((b, i) => (
                            <line
                                key={`line-${i}`}
                                x1={scaleX(estimated.tps)}
                                y1={scaleY(estimated.ttft)}
                                x2={scaleX(b.tps)}
                                y2={scaleY(b.ttft)}
                                stroke="#e5e7eb"
                                strokeWidth="1"
                                strokeDasharray="4,4"
                            />
                        ))}

                        {/* Benchmark points (gray) */}
                        {benchmarks.map((b, i) => (
                            <g key={`bench-${i}`}>
                                <circle
                                    cx={scaleX(b.tps)}
                                    cy={scaleY(b.ttft)}
                                    r={7}
                                    fill="#a1a1aa"
                                    stroke="#52525b"
                                    strokeWidth="2"
                                />
                                <text
                                    x={scaleX(b.tps)}
                                    y={scaleY(b.ttft) - 12}
                                    textAnchor="middle"
                                    fontSize="9"
                                    fill="#71717a"
                                    fontFamily="JetBrains Mono, monospace"
                                >
                                    {i + 1}
                                </text>
                            </g>
                        ))}

                        {/* Estimated point (accent) */}
                        <circle
                            cx={scaleX(estimated.tps)}
                            cy={scaleY(estimated.ttft)}
                            r={9}
                            fill="#3a7a8c"
                            stroke="#1e3d4a"
                            strokeWidth="2"
                        />
                        <text
                            x={scaleX(estimated.tps)}
                            y={scaleY(estimated.ttft) - 14}
                            textAnchor="middle"
                            fontSize="10"
                            fill="#2d5a6b"
                            fontWeight="600"
                            fontFamily="DM Sans, sans-serif"
                        >
                            Estimate
                        </text>
                    </svg>
                </div>

                <div className="legend">
                    <div className="legend-item">
                        <span className="legend-dot estimated" />
                        <span>Estimated</span>
                    </div>
                    <div className="legend-item">
                        <span className="legend-dot benchmark" />
                        <span>Real benchmarks</span>
                    </div>
                </div>

                <div className="benchmark-list">
                    <h3>Nearest Real Benchmarks</h3>
                    {benchmarks.map((b, i) => (
                        <div key={i} className="benchmark-item">
                            <span className="benchmark-number">{i + 1}</span>
                            <div className="benchmark-info">
                                <div className="benchmark-hardware">{b.hardware}</div>
                                <div className="benchmark-model">{b.model} {b.quant}</div>
                                <div className="benchmark-stats">
                                    {b.tps.toFixed(1)} tok/s • {formatMs(b.ttft)}
                                </div>
                            </div>
                            {b.source && (
                                <a
                                    href={b.source}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="benchmark-source"
                                >
                                    Source
                                </a>
                            )}
                        </div>
                    ))}
                </div>

                <div className="confidence-explanation">
                    <h3>How is confidence calculated?</h3>
                    <p>
                        The confidence score indicates how close your configuration is to real benchmark data.
                        {simResult.is_measured ? (
                            <span> Since your exact configuration was found in our database, confidence is <strong>95%</strong>.</span>
                        ) : (
                            <span>
                                {' '}We use a machine learning model trained on {modelInfo?.training_samples || 237} real benchmarks to predict performance.
                                The confidence ({confidencePercent}%) is based on the distance from your configuration to the nearest
                                training data points in our feature space (memory bandwidth, model size, context length, etc.).
                            </span>
                        )}
                    </p>
                    <p className="confidence-note">
                        Closer benchmarks = higher confidence. The chart above shows the 4 nearest real measurements to your estimate.
                    </p>
                </div>

                {modelInfo && (
                    <div className="model-info-section">
                        <h3>ML Model Details</h3>
                        <div className="model-metrics">
                            <div className="metric-item">
                                <span className="metric-label">Algorithm</span>
                                <span className="metric-value">
                                    <a
                                        href="https://en.wikipedia.org/wiki/Random_forest#Extremely_randomized_trees"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="algorithm-link"
                                    >
                                        {modelInfo.algorithm}
                                    </a>
                                </span>
                            </div>
                            {modelInfo.r2_score && (
                                <div className="metric-item">
                                    <span className="metric-label">R² Score</span>
                                    <span className="metric-value">{(modelInfo.r2_score * 100).toFixed(1)}%</span>
                                </div>
                            )}
                            {modelInfo.mean_absolute_error && (
                                <div className="metric-item">
                                    <span className="metric-label">Mean Abs. Error</span>
                                    <span className="metric-value">{modelInfo.mean_absolute_error.toFixed(2)} tok/s</span>
                                </div>
                            )}
                            {modelInfo.mean_percentage_error && (
                                <div className="metric-item">
                                    <span className="metric-label">Mean % Error</span>
                                    <span className="metric-value">{modelInfo.mean_percentage_error.toFixed(1)}%</span>
                                </div>
                            )}
                            <div className="metric-item">
                                <span className="metric-label">Training Samples</span>
                                <span className="metric-value">{modelInfo.training_samples}</span>
                            </div>
                        </div>

                        {modelInfo.feature_importance?.length > 0 && (
                            <div className="feature-importance">
                                <h4>Feature Importance</h4>
                                <div className="feature-bars">
                                    {modelInfo.feature_importance.slice(0, 6).map((f, i) => (
                                        <div key={i} className="feature-item">
                                            <span className="feature-name">{FEATURE_LABELS[f.name] || f.name.replace(/_/g, ' ')}</span>
                                            <div className="feature-bar-container">
                                                <div
                                                    className="feature-bar"
                                                    style={{ width: `${Math.min(f.importance * 4, 100)}%` }}
                                                />
                                                <span className="feature-percent">{f.importance.toFixed(1)}%</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default ConfidenceExplainer;
