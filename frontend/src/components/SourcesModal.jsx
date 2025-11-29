import { useState, useEffect, useMemo } from 'react';
import { getBenchmarks } from '../api/client';

function SourcesModal({ onClose }) {
    const [benchmarks, setBenchmarks] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [sortColumn, setSortColumn] = useState('hardware_name');
    const [sortDirection, setSortDirection] = useState('asc');

    useEffect(() => {
        getBenchmarks()
            .then(data => {
                setBenchmarks(data);
                setLoading(false);
            })
            .catch(err => {
                console.error('Failed to load benchmarks:', err);
                setLoading(false);
            });
    }, []);

    const handleSort = (column) => {
        if (sortColumn === column) {
            setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
        } else {
            setSortColumn(column);
            setSortDirection('asc');
        }
    };

    const filteredAndSorted = useMemo(() => {
        let result = [...benchmarks];

        // Filter by search query
        if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase();
            result = result.filter(b =>
                b.hardware_name.toLowerCase().includes(query) ||
                b.model_family.toLowerCase().includes(query) ||
                b.quant_type.toLowerCase().includes(query)
            );
        }

        // Sort
        result.sort((a, b) => {
            let aVal = a[sortColumn];
            let bVal = b[sortColumn];

            // Handle null/undefined values
            if (aVal == null) aVal = '';
            if (bVal == null) bVal = '';

            // For strings, compare case-insensitively
            if (typeof aVal === 'string') {
                aVal = aVal.toLowerCase();
                bVal = bVal.toLowerCase();
            }

            if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
            if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
            return 0;
        });

        return result;
    }, [benchmarks, searchQuery, sortColumn, sortDirection]);

    const getSortIcon = (column) => {
        if (sortColumn !== column) return '';
        return sortDirection === 'asc' ? ' ▲' : ' ▼';
    };

    const formatCtx = (ctx) => {
        if (!ctx) return '-';
        if (ctx >= 1000) return `${(ctx / 1000).toFixed(0)}K`;
        return ctx.toString();
    };

    return (
        <div className="confidence-modal-overlay" onClick={onClose}>
            <div className="confidence-modal sources-modal" onClick={e => e.stopPropagation()}>
                <button className="close-btn" onClick={onClose}>×</button>
                <h2>Benchmark Sources ({benchmarks.length} data points)</h2>

                <div className="sources-search">
                    <input
                        type="text"
                        placeholder="Search hardware or model..."
                        value={searchQuery}
                        onChange={e => setSearchQuery(e.target.value)}
                        autoFocus
                    />
                </div>

                {loading ? (
                    <div className="sources-loading">Loading benchmarks...</div>
                ) : (
                    <div className="sources-table-container">
                        <table className="sources-table">
                            <thead>
                                <tr>
                                    <th
                                        className="sortable"
                                        onClick={() => handleSort('hardware_name')}
                                    >
                                        Hardware{getSortIcon('hardware_name')}
                                    </th>
                                    <th
                                        className="sortable"
                                        onClick={() => handleSort('model_family')}
                                    >
                                        Model{getSortIcon('model_family')}
                                    </th>
                                    <th
                                        className="sortable"
                                        onClick={() => handleSort('model_size_b')}
                                    >
                                        Size{getSortIcon('model_size_b')}
                                    </th>
                                    <th>Quant</th>
                                    <th
                                        className="sortable"
                                        onClick={() => handleSort('context_length')}
                                    >
                                        Ctx{getSortIcon('context_length')}
                                    </th>
                                    <th
                                        className="sortable"
                                        onClick={() => handleSort('generation_tps')}
                                    >
                                        TPS{getSortIcon('generation_tps')}
                                    </th>
                                    <th>TTFT</th>
                                    <th>Source</th>
                                </tr>
                            </thead>
                            <tbody>
                                {filteredAndSorted.map(b => (
                                    <tr key={b.id}>
                                        <td className="hardware-cell">{b.hardware_name}</td>
                                        <td>{b.model_family}</td>
                                        <td className="num-cell">{b.model_size_b}B</td>
                                        <td>{b.quant_type}</td>
                                        <td className="num-cell">{formatCtx(b.context_length)}</td>
                                        <td className="num-cell">{b.generation_tps.toFixed(1)}</td>
                                        <td className="num-cell">
                                            {b.ttft_ms ? `${Math.round(b.ttft_ms)}ms` : '-'}
                                        </td>
                                        <td className="source-cell">
                                            {b.source ? (
                                                <a
                                                    href={b.source}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    title={b.source}
                                                >
                                                    Link
                                                </a>
                                            ) : '-'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {filteredAndSorted.length === 0 && (
                            <div className="no-results">No benchmarks found matching "{searchQuery}"</div>
                        )}
                    </div>
                )}

                <div className="sources-footer">
                    <p>
                        Data collected from: llama.cpp discussions, LocalScore.ai, hardware-corner.net,
                        VALDI docs, RunPod blog, and more.
                    </p>
                </div>
            </div>
        </div>
    );
}

export default SourcesModal;
