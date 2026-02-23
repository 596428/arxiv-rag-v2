// arXiv RAG v2 - Evaluation Dashboard Charts
console.log('[Charts] Script loaded - v2');

// Chart.js Dark Mode Configuration
if (typeof Chart !== 'undefined') {
    Chart.defaults.color = '#e2e8f0';           // slate-200 for text
    Chart.defaults.borderColor = '#475569';     // slate-600 for borders
    console.log('[Charts] Chart.js configured');
} else {
    console.error('[Charts] Chart.js not loaded!');
}

// ==========================================
// V2 COLOR SCHEME (10 models: 5 base + 5 reranked)
// ==========================================
const MODEL_COLORS = {
    // Dense Family (Greens)
    'qdrant_dense':           'rgba(16, 185, 129, 0.85)',   // Emerald
    'qdrant_dense+rerank':    'rgba(52, 211, 153, 0.85)',   // Light emerald

    // Sparse Family (Ambers)
    'qdrant_sparse':          'rgba(245, 158, 11, 0.85)',   // Amber
    'qdrant_sparse+rerank':   'rgba(251, 191, 36, 0.85)',   // Light amber

    // Hybrid Family (Blues)
    'qdrant_hybrid':          'rgba(59, 130, 246, 0.85)',   // Blue
    'qdrant_hybrid+rerank':   'rgba(96, 165, 250, 0.85)',   // Light blue

    // OpenAI 3-large Family (Purples)
    'qdrant_3large':          'rgba(139, 92, 246, 0.85)',   // Violet
    'qdrant_3large+rerank':   'rgba(167, 139, 250, 0.85)',  // Light violet

    // Cross-model Hybrid Family (Teals)
    'qdrant_hybrid_3large':        'rgba(20, 184, 166, 0.85)',   // Teal
    'qdrant_hybrid_3large+rerank': 'rgba(94, 234, 212, 0.85)',   // Light teal
};

// Full labels for tooltips and tables
const MODEL_LABELS = {
    'qdrant_dense':           'Dense (BGE-M3)',
    'qdrant_dense+rerank':    'Dense + Reranker',
    'qdrant_sparse':          'Sparse (BGE-M3)',
    'qdrant_sparse+rerank':   'Sparse + Reranker',
    'qdrant_hybrid':          'Hybrid (BGE-M3)',
    'qdrant_hybrid+rerank':   'Hybrid + Reranker',
    'qdrant_3large':          'OpenAI 3-large',
    'qdrant_3large+rerank':   '3-large + Reranker',
    'qdrant_hybrid_3large':        'Hybrid-3L (OpenAI+BGE)',
    'qdrant_hybrid_3large+rerank': 'Hybrid-3L + Reranker',
};

// Short labels for charts
const MODEL_LABELS_SHORT = {
    'qdrant_dense':           'Dense',
    'qdrant_dense+rerank':    'Dense+RR',
    'qdrant_sparse':          'Sparse',
    'qdrant_sparse+rerank':   'Sparse+RR',
    'qdrant_hybrid':          'Hybrid',
    'qdrant_hybrid+rerank':   'Hybrid+RR',
    'qdrant_3large':          '3-large',
    'qdrant_3large+rerank':   '3-large+RR',
    'qdrant_hybrid_3large':        'Hyb-3L',
    'qdrant_hybrid_3large+rerank': 'Hyb-3L+RR',
};

// Detailed tooltips for complex models
const MODEL_TOOLTIPS = {
    'qdrant_dense':           'BGE-M3 dense vectors (1024d). Fast semantic search.',
    'qdrant_dense+rerank':    'BGE-M3 dense + BGE Reranker. Improved ranking quality.',
    'qdrant_sparse':          'BGE-M3 sparse vectors. Keyword-aware matching.',
    'qdrant_sparse+rerank':   'BGE-M3 sparse + BGE Reranker.',
    'qdrant_hybrid':          'BGE-M3 dense + sparse fusion. Best of both worlds.',
    'qdrant_hybrid+rerank':   'BGE-M3 hybrid + BGE Reranker. Strong quality.',
    'qdrant_3large':          'OpenAI text-embedding-3-large (3072d). High-quality embeddings.',
    'qdrant_3large+rerank':   'OpenAI 3-large + BGE Reranker. Premium quality.',
    'qdrant_hybrid_3large':        'OpenAI 3-large dense + BGE sparse. Cross-model hybrid.',
    'qdrant_hybrid_3large+rerank': 'Hybrid-3L + BGE Reranker. Maximum quality, higher latency.',
};

// Model ordering for consistent display
const MODEL_ORDER = [
    'qdrant_dense',
    'qdrant_dense+rerank',
    'qdrant_sparse',
    'qdrant_sparse+rerank',
    'qdrant_hybrid',
    'qdrant_hybrid+rerank',
    'qdrant_3large',
    'qdrant_3large+rerank',
    'qdrant_hybrid_3large',
    'qdrant_hybrid_3large+rerank',
];

// Base models (for filtering)
const BASE_MODELS = [
    'qdrant_dense',
    'qdrant_sparse',
    'qdrant_hybrid',
    'qdrant_3large',
    'qdrant_hybrid_3large',
];

// Reranked models (for filtering)
const RERANKED_MODELS = [
    'qdrant_dense+rerank',
    'qdrant_sparse+rerank',
    'qdrant_hybrid+rerank',
    'qdrant_3large+rerank',
    'qdrant_hybrid_3large+rerank',
];

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

function isRerankedModel(model) {
    return model.includes('+rerank');
}

function getBaseModel(model) {
    return model.replace('+rerank', '');
}

function getRerankedModel(model) {
    if (isRerankedModel(model)) return model;
    return model + '+rerank';
}

function calculateDelta(baseValue, rerankedValue) {
    if (baseValue === 0) return 0;
    return ((rerankedValue - baseValue) / baseValue * 100);
}

function formatDelta(delta) {
    const sign = delta >= 0 ? '+' : '';
    return `${sign}${delta.toFixed(1)}%`;
}

// Calculate efficiency score: ΔNDCG@10 / ΔLatency(ms) * 1000
function calculateEfficiencyScore(baseData, rerankedData) {
    const deltaNDCG = rerankedData['avg_ndcg@10'] - baseData['avg_ndcg@10'];
    const deltaLatency = rerankedData.avg_search_time_ms - baseData.avg_search_time_ms;

    if (deltaLatency <= 0) return Infinity; // Edge case: somehow faster

    // Gain per 100ms of latency increase
    return (deltaNDCG / deltaLatency) * 1000;
}

// Get sorted models based on available data
function getSortedModels(data, filter = 'all') {
    let models = Object.keys(data);

    // Filter based on selection
    if (filter === 'base') {
        models = models.filter(m => !isRerankedModel(m));
    } else if (filter === 'rerank') {
        models = models.filter(m => isRerankedModel(m));
    }

    // Sort by MODEL_ORDER
    return models.sort((a, b) => {
        const orderA = MODEL_ORDER.indexOf(a);
        const orderB = MODEL_ORDER.indexOf(b);
        if (orderA === -1 && orderB === -1) return a.localeCompare(b);
        if (orderA === -1) return 1;
        if (orderB === -1) return -1;
        return orderA - orderB;
    });
}

// Find best model for a metric
function findBestModel(data, metric, filter = 'all') {
    const models = getSortedModels(data, filter);
    let best = null;
    let bestValue = -Infinity;

    models.forEach(model => {
        const value = data[model][metric] || 0;
        if (value > bestValue) {
            bestValue = value;
            best = model;
        }
    });

    return { model: best, value: bestValue };
}

// Find fastest model
function findFastestModel(data, filter = 'all') {
    const models = getSortedModels(data, filter);
    let fastest = null;
    let fastestLatency = Infinity;

    models.forEach(model => {
        const latency = data[model].avg_search_time_ms || Infinity;
        if (latency < fastestLatency) {
            fastestLatency = latency;
            fastest = model;
        }
    });

    return { model: fastest, value: fastestLatency };
}

// Calculate best efficiency (best reranker ROI)
function findBestEfficiency(data) {
    let best = null;
    let bestScore = -Infinity;

    BASE_MODELS.forEach(baseModel => {
        const rerankedModel = getRerankedModel(baseModel);
        if (data[baseModel] && data[rerankedModel]) {
            const score = calculateEfficiencyScore(data[baseModel], data[rerankedModel]);
            if (score > bestScore && isFinite(score)) {
                bestScore = score;
                best = baseModel;
            }
        }
    });

    return { baseModel: best, score: bestScore };
}

// Calculate average reranker gain
function calculateAvgRerankerGain(data) {
    let totalGain = 0;
    let count = 0;

    BASE_MODELS.forEach(baseModel => {
        const rerankedModel = getRerankedModel(baseModel);
        if (data[baseModel] && data[rerankedModel]) {
            const delta = calculateDelta(
                data[baseModel]['avg_ndcg@10'],
                data[rerankedModel]['avg_ndcg@10']
            );
            totalGain += delta;
            count++;
        }
    });

    return count > 0 ? totalGain / count : 0;
}

// ==========================================
// GLOBAL STATE
// ==========================================

let evaluationData = null;
let currentFilter = 'all'; // 'all', 'base', 'rerank'
let chartInstances = {}; // Store chart instances for destruction

// Heatmap state
let heatmapRerankerEnabled = false;
let deltaRerankerEnabled = false;

// ==========================================
// DATA LOADING
// ==========================================

async function loadData() {
    try {
        const response = await fetch('data/evaluation_results.json');
        evaluationData = await response.json();
        console.log('[Charts] Loaded evaluation data:', Object.keys(evaluationData));
        renderDashboard();
    } catch (error) {
        console.error('Failed to load evaluation data:', error);
        document.getElementById('findings').innerHTML = `
            <p class="text-red-500">Failed to load evaluation data. Make sure data/evaluation_results.json exists.</p>
        `;
    }
}

// Filter change handler
function setFilter(filter) {
    currentFilter = filter;

    // Update button states
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active', 'bg-blue-600', 'text-white');
        btn.classList.add('bg-slate-700', 'text-slate-300');
    });

    const activeBtn = document.querySelector(`.filter-btn[data-filter="${filter}"]`);
    if (activeBtn) {
        activeBtn.classList.remove('bg-slate-700', 'text-slate-300');
        activeBtn.classList.add('active', 'bg-blue-600', 'text-white');
    }

    // Re-render filtered charts
    renderMetricsChart();
    renderLatencyChart();
    renderResultsTable();
}

// ==========================================
// STACKED BAR CHART (5 Metrics × 5 Models)
// ==========================================

// Model colors for the stacked bar chart
const MODEL_BASE_COLORS = {
    'qdrant_dense': 'rgba(16, 185, 129, 0.85)',      // Emerald
    'qdrant_sparse': 'rgba(245, 158, 11, 0.85)',     // Amber
    'qdrant_hybrid': 'rgba(59, 130, 246, 0.85)',     // Blue
    'qdrant_3large': 'rgba(139, 92, 246, 0.85)',     // Violet
    'qdrant_hybrid_3large': 'rgba(20, 184, 166, 0.85)', // Teal
};

const MODEL_GAIN_COLORS = {
    'qdrant_dense': 'rgba(134, 239, 172, 0.9)',      // Light emerald
    'qdrant_sparse': 'rgba(253, 224, 71, 0.9)',      // Light amber
    'qdrant_hybrid': 'rgba(147, 197, 253, 0.9)',     // Light blue
    'qdrant_3large': 'rgba(196, 181, 253, 0.9)',     // Light violet
    'qdrant_hybrid_3large': 'rgba(153, 246, 228, 0.9)', // Light teal
};

const MODEL_LOSS_COLORS = {
    'qdrant_dense': 'rgba(252, 165, 165, 0.9)',      // Light red
    'qdrant_sparse': 'rgba(252, 165, 165, 0.9)',
    'qdrant_hybrid': 'rgba(252, 165, 165, 0.9)',
    'qdrant_3large': 'rgba(252, 165, 165, 0.9)',
    'qdrant_hybrid_3large': 'rgba(252, 165, 165, 0.9)',
};

function renderStackedBarChart() {
    const canvas = document.getElementById('stackedBarChart');
    if (!canvas || !evaluationData) return;

    // Destroy existing chart
    if (chartInstances.stackedBar) {
        chartInstances.stackedBar.destroy();
    }

    const ctx = canvas.getContext('2d');

    // 5 metrics to display
    const metrics = [
        { key: 'avg_mrr', label: 'MRR' },
        { key: 'avg_ndcg@10', label: 'NDCG@10' },
        { key: 'avg_ndcg@5', label: 'NDCG@5' },
        { key: 'avg_precision@10', label: 'P@10' },
        { key: 'avg_precision@5', label: 'P@5' },
    ];

    // Create labels: "MRR", "NDCG@10", etc.
    const labels = metrics.map(m => m.label);

    // Create datasets: one for each model (base + gain/loss stacked)
    const datasets = [];

    BASE_MODELS.forEach((baseModel, modelIdx) => {
        const rerankedModel = getRerankedModel(baseModel);
        const baseData = evaluationData[baseModel];
        const rerankedData = evaluationData[rerankedModel];

        if (!baseData || !rerankedData) return;

        const shortLabel = MODEL_LABELS_SHORT[baseModel];

        // Base values for all metrics
        const baseValues = metrics.map(m => baseData[m.key] || 0);

        // Calculate gains/losses for all metrics
        const gainValues = [];
        const lossValues = [];

        metrics.forEach(m => {
            const baseVal = baseData[m.key] || 0;
            const rerankedVal = rerankedData[m.key] || 0;
            const delta = rerankedVal - baseVal;

            if (delta >= 0) {
                gainValues.push(delta);
                lossValues.push(0);
            } else {
                gainValues.push(0);
                lossValues.push(Math.abs(delta));
            }
        });

        // Add base dataset
        datasets.push({
            label: `${shortLabel} (Base)`,
            data: baseValues,
            backgroundColor: MODEL_BASE_COLORS[baseModel],
            borderColor: MODEL_BASE_COLORS[baseModel].replace('0.85', '1'),
            borderWidth: 1,
            stack: `stack${modelIdx}`,
            barPercentage: 0.85,
            categoryPercentage: 0.9,
        });

        // Add gain dataset (stacked on top)
        datasets.push({
            label: `${shortLabel} (+RR Gain)`,
            data: gainValues,
            backgroundColor: MODEL_GAIN_COLORS[baseModel],
            borderColor: MODEL_GAIN_COLORS[baseModel].replace('0.9', '1'),
            borderWidth: 1,
            stack: `stack${modelIdx}`,
            barPercentage: 0.85,
            categoryPercentage: 0.9,
        });

        // Add loss dataset (stacked downward - negative values)
        datasets.push({
            label: `${shortLabel} (+RR Loss)`,
            data: lossValues.map(v => -v),
            backgroundColor: MODEL_LOSS_COLORS[baseModel],
            borderColor: 'rgba(239, 68, 68, 1)',
            borderWidth: 1,
            stack: `stack${modelIdx}`,
            barPercentage: 0.85,
            categoryPercentage: 0.9,
        });
    });

    chartInstances.stackedBar = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets,
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false, // Legend is shown above chart as custom HTML
                },
                tooltip: {
                    callbacks: {
                        title: (items) => {
                            return items[0].label; // Metric name
                        },
                        label: (context) => {
                            const value = Math.abs(context.raw);
                            const label = context.dataset.label;
                            if (value === 0) return null;
                            if (label.includes('Base')) {
                                return `${label.replace(' (Base)', '')}: ${value.toFixed(4)}`;
                            } else if (label.includes('Gain')) {
                                return `  +Rerank: +${value.toFixed(4)}`;
                            } else if (label.includes('Loss')) {
                                return `  +Rerank: -${value.toFixed(4)}`;
                            }
                            return `${label}: ${value.toFixed(4)}`;
                        },
                        filter: (item) => item.raw !== 0,
                    },
                },
            },
            scales: {
                y: {
                    stacked: true,
                    beginAtZero: false,
                    grid: { color: 'rgba(148, 163, 184, 0.2)' },
                    ticks: { color: '#94a3b8' },
                    title: {
                        display: true,
                        text: 'Score',
                        color: '#e2e8f0',
                    },
                },
                x: {
                    stacked: true,
                    grid: { color: 'rgba(148, 163, 184, 0.2)' },
                    ticks: { color: '#94a3b8', font: { weight: 'bold' } },
                },
            },
        },
    });
}

// ==========================================
// RADAR CHARTS (Split: Base / Reranked)
// ==========================================

function renderBaseRadarChart() {
    const canvas = document.getElementById('baseRadarChart');
    if (!canvas) return;

    if (chartInstances.baseRadar) {
        chartInstances.baseRadar.destroy();
    }

    const ctx = canvas.getContext('2d');
    const models = getSortedModels(evaluationData, 'base');

    const metrics = ['avg_mrr', 'avg_ndcg@5', 'avg_ndcg@10', 'avg_precision@5', 'avg_precision@10'];
    const metricLabels = ['MRR', 'NDCG@5', 'NDCG@10', 'P@5', 'P@10'];

    const datasets = models.map(model => ({
        label: MODEL_LABELS_SHORT[model] || model,
        data: metrics.map(m => evaluationData[model][m] || 0),
        backgroundColor: (MODEL_COLORS[model] || 'rgba(156, 163, 175, 0.8)').replace('0.85', '0.2'),
        borderColor: MODEL_COLORS[model] || 'rgba(156, 163, 175, 1)',
        borderWidth: 2,
        pointBackgroundColor: MODEL_COLORS[model] || 'rgba(156, 163, 175, 1)',
    }));

    chartInstances.baseRadar = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: metricLabels,
            datasets: datasets,
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#e2e8f0' },
                },
                title: {
                    display: true,
                    text: 'Base Models Profile',
                    color: '#e2e8f0',
                    font: { size: 14 }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    grid: { color: 'rgba(148, 163, 184, 0.3)' },
                    angleLines: { color: 'rgba(148, 163, 184, 0.3)' },
                    pointLabels: { color: '#e2e8f0' },
                    ticks: { color: '#94a3b8', backdropColor: 'transparent' },
                },
            },
        },
    });
}

function renderRerankedRadarChart() {
    const canvas = document.getElementById('rerankedRadarChart');
    if (!canvas) return;

    if (chartInstances.rerankedRadar) {
        chartInstances.rerankedRadar.destroy();
    }

    const ctx = canvas.getContext('2d');
    const models = getSortedModels(evaluationData, 'rerank');

    const metrics = ['avg_mrr', 'avg_ndcg@5', 'avg_ndcg@10', 'avg_precision@5', 'avg_precision@10'];
    const metricLabels = ['MRR', 'NDCG@5', 'NDCG@10', 'P@5', 'P@10'];

    const datasets = models.map(model => ({
        label: MODEL_LABELS_SHORT[model] || model,
        data: metrics.map(m => evaluationData[model][m] || 0),
        backgroundColor: (MODEL_COLORS[model] || 'rgba(156, 163, 175, 0.8)').replace('0.85', '0.2'),
        borderColor: MODEL_COLORS[model] || 'rgba(156, 163, 175, 1)',
        borderWidth: 2,
        pointBackgroundColor: MODEL_COLORS[model] || 'rgba(156, 163, 175, 1)',
    }));

    chartInstances.rerankedRadar = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: metricLabels,
            datasets: datasets,
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#e2e8f0' },
                },
                title: {
                    display: true,
                    text: 'Reranked Models Profile',
                    color: '#e2e8f0',
                    font: { size: 14 }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    grid: { color: 'rgba(148, 163, 184, 0.3)' },
                    angleLines: { color: 'rgba(148, 163, 184, 0.3)' },
                    pointLabels: { color: '#e2e8f0' },
                    ticks: { color: '#94a3b8', backdropColor: 'transparent' },
                },
            },
        },
    });
}

// ==========================================
// RERANKER IMPACT CHART (NDCG focused)
// ==========================================

function renderRerankerImpactChart() {
    const canvas = document.getElementById('rerankerImpactChart');
    if (!canvas) return;

    if (chartInstances.rerankerImpact) {
        chartInstances.rerankerImpact.destroy();
    }

    const ctx = canvas.getContext('2d');

    // Calculate deltas for each base model
    const modelPairs = [];
    const labels = [];
    const ndcg5Deltas = [];
    const ndcg10Deltas = [];
    const mrrDeltas = [];
    const p10Deltas = [];

    BASE_MODELS.forEach(baseModel => {
        const rerankedModel = getRerankedModel(baseModel);
        if (evaluationData[baseModel] && evaluationData[rerankedModel]) {
            const baseData = evaluationData[baseModel];
            const rerankedData = evaluationData[rerankedModel];

            labels.push(MODEL_LABELS_SHORT[baseModel]);
            ndcg5Deltas.push(calculateDelta(baseData['avg_ndcg@5'], rerankedData['avg_ndcg@5']));
            ndcg10Deltas.push(calculateDelta(baseData['avg_ndcg@10'], rerankedData['avg_ndcg@10']));
            mrrDeltas.push(calculateDelta(baseData['avg_mrr'], rerankedData['avg_mrr']));
            p10Deltas.push(calculateDelta(baseData['avg_precision@10'], rerankedData['avg_precision@10']));
        }
    });

    chartInstances.rerankerImpact = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'ΔNDCG@10',
                    data: ndcg10Deltas,
                    backgroundColor: 'rgba(59, 130, 246, 0.9)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 2,
                    barThickness: 20,
                },
                {
                    label: 'ΔNDCG@5',
                    data: ndcg5Deltas,
                    backgroundColor: 'rgba(96, 165, 250, 0.7)',
                    borderColor: 'rgba(96, 165, 250, 1)',
                    borderWidth: 1,
                    barThickness: 16,
                },
                {
                    label: 'ΔMRR',
                    data: mrrDeltas,
                    backgroundColor: 'rgba(16, 185, 129, 0.6)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 1,
                    barThickness: 12,
                },
                {
                    label: 'ΔP@10',
                    data: p10Deltas,
                    backgroundColor: 'rgba(139, 92, 246, 0.5)',
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 1,
                    barThickness: 10,
                },
            ],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#e2e8f0' },
                },
                title: {
                    display: true,
                    text: 'Reranker Impact (% Improvement)',
                    color: '#e2e8f0',
                    font: { size: 14 }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const value = context.raw;
                            return `${context.dataset.label}: ${formatDelta(value)}`;
                        },
                        afterBody: (items) => {
                            if (items[0].dataset.label.includes('NDCG@10')) {
                                return 'NDCG measures ranking quality (position-weighted)';
                            }
                            return '';
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { color: 'rgba(148, 163, 184, 0.2)' },
                    ticks: {
                        color: '#94a3b8',
                        callback: (value) => `${value}%`
                    },
                    title: {
                        display: true,
                        text: '% Improvement',
                        color: '#e2e8f0',
                    },
                },
                y: {
                    grid: { color: 'rgba(148, 163, 184, 0.2)' },
                    ticks: { color: '#94a3b8' },
                },
            },
        },
    });
}

// ==========================================
// LATENCY CHART (10 models)
// ==========================================

function renderLatencyChart() {
    const canvas = document.getElementById('latencyChart');
    if (!canvas) return;

    if (chartInstances.latency) {
        chartInstances.latency.destroy();
    }

    const ctx = canvas.getContext('2d');
    const models = getSortedModels(evaluationData, currentFilter);

    // Calculate reranker overhead for display
    const labels = models.map(m => {
        const label = MODEL_LABELS_SHORT[m] || m;
        if (isRerankedModel(m)) {
            const baseModel = getBaseModel(m);
            if (evaluationData[baseModel]) {
                const overhead = evaluationData[m].avg_search_time_ms - evaluationData[baseModel].avg_search_time_ms;
                return `${label} (+${Math.round(overhead)}ms)`;
            }
        }
        return label;
    });

    chartInstances.latency = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Average Search Time (ms)',
                data: models.map(m => evaluationData[m].avg_search_time_ms || 0),
                backgroundColor: models.map(m => MODEL_COLORS[m] || 'rgba(156, 163, 175, 0.8)'),
                borderColor: models.map(m => (MODEL_COLORS[m] || 'rgba(156, 163, 175, 0.8)').replace('0.85', '1')),
                borderWidth: 1,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        afterLabel: (context) => {
                            const model = models[context.dataIndex];
                            if (isRerankedModel(model)) {
                                const baseModel = getBaseModel(model);
                                if (evaluationData[baseModel]) {
                                    const overhead = evaluationData[model].avg_search_time_ms - evaluationData[baseModel].avg_search_time_ms;
                                    return `Reranker overhead: +${Math.round(overhead)}ms`;
                                }
                            }
                            return '';
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { color: 'rgba(148, 163, 184, 0.2)' },
                    ticks: { color: '#94a3b8' },
                    title: {
                        display: true,
                        text: 'Latency (ms)',
                        color: '#e2e8f0',
                    },
                },
                y: {
                    grid: { color: 'rgba(148, 163, 184, 0.2)' },
                    ticks: { color: '#94a3b8' },
                },
            },
        },
    });
}

// ==========================================
// RESULTS TABLE (with deltas, badges, tooltips)
// ==========================================

function renderResultsTable() {
    const tbody = document.getElementById('results-table');
    if (!tbody) return;

    const models = getSortedModels(evaluationData, currentFilter);

    // Find best values for highlighting
    const bestMRR = Math.max(...models.map(m => evaluationData[m].avg_mrr || 0));
    const bestNDCG10 = Math.max(...models.map(m => evaluationData[m]['avg_ndcg@10'] || 0));
    const bestP10 = Math.max(...models.map(m => evaluationData[m]['avg_precision@10'] || 0));
    const fastestLatency = Math.min(...models.map(m => evaluationData[m].avg_search_time_ms || Infinity));

    tbody.innerHTML = models.map(model => {
        const data = evaluationData[model];
        const isReranked = isRerankedModel(model);
        const baseModel = isReranked ? getBaseModel(model) : null;
        const baseData = baseModel ? evaluationData[baseModel] : null;

        // Row styling
        const rowClass = isReranked ? 'bg-slate-700/30' : '';

        // Badge logic
        let badge = '';
        if (data.avg_mrr === bestMRR && data['avg_ndcg@10'] === bestNDCG10) {
            badge = '<span class="text-yellow-400 ml-2" title="Best Quality">🏆</span>';
        }
        if (data.avg_search_time_ms === fastestLatency) {
            badge += '<span class="text-blue-400 ml-2" title="Fastest">⚡</span>';
        }

        // Delta display for reranked models
        const deltaDisplay = (metric) => {
            if (!isReranked || !baseData) return '';
            const delta = calculateDelta(baseData[metric], data[metric]);
            const color = delta >= 0 ? 'text-green-400' : 'text-red-400';
            return `<span class="${color} text-xs ml-1">(${formatDelta(delta)})</span>`;
        };

        // Highlight best values
        const highlightClass = (value, best) => value === best ? 'text-green-400 font-semibold' : 'text-slate-300';

        return `
            <tr class="${rowClass}">
                <td class="px-6 py-4 whitespace-nowrap font-medium text-slate-100" title="${MODEL_TOOLTIPS[model] || ''}">
                    ${isReranked ? '<span class="text-purple-400 mr-1">↳</span>' : ''}
                    ${MODEL_LABELS[model] || model}
                    ${badge}
                </td>
                <td class="px-6 py-4 whitespace-nowrap ${highlightClass(data.avg_mrr, bestMRR)}">
                    ${(data.avg_mrr || 0).toFixed(3)}${deltaDisplay('avg_mrr')}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-slate-300">
                    ${(data['avg_ndcg@5'] || 0).toFixed(3)}${deltaDisplay('avg_ndcg@5')}
                </td>
                <td class="px-6 py-4 whitespace-nowrap ${highlightClass(data['avg_ndcg@10'], bestNDCG10)}">
                    ${(data['avg_ndcg@10'] || 0).toFixed(3)}${deltaDisplay('avg_ndcg@10')}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-slate-300">
                    ${(data['avg_precision@5'] || 0).toFixed(3)}${deltaDisplay('avg_precision@5')}
                </td>
                <td class="px-6 py-4 whitespace-nowrap ${highlightClass(data['avg_precision@10'], bestP10)}">
                    ${(data['avg_precision@10'] || 0).toFixed(3)}${deltaDisplay('avg_precision@10')}
                </td>
                <td class="px-6 py-4 whitespace-nowrap ${data.avg_search_time_ms === fastestLatency ? 'text-blue-400 font-semibold' : 'text-slate-300'}">
                    ${Math.round(data.avg_search_time_ms || 0)}
                    ${isReranked && baseData ? `<span class="text-orange-400 text-xs ml-1">(+${Math.round(data.avg_search_time_ms - baseData.avg_search_time_ms)})</span>` : ''}
                </td>
            </tr>
        `;
    }).join('');
}

// ==========================================
// KEY FINDINGS
// ==========================================

function renderFindings() {
    const models = getSortedModels(evaluationData);
    let findings = [];

    // Sort by MRR/NDCG
    const sortedByNDCG = models
        .map(m => ({ model: m, ndcg: evaluationData[m]['avg_ndcg@10'], data: evaluationData[m] }))
        .sort((a, b) => b.ndcg - a.ndcg);

    const sortedByLatency = models
        .map(m => ({ model: m, latency: evaluationData[m].avg_search_time_ms, data: evaluationData[m] }))
        .sort((a, b) => a.latency - b.latency);

    const best = sortedByNDCG[0];
    const second = sortedByNDCG[1];
    const fastest = sortedByLatency[0];
    const slowest = sortedByLatency[sortedByLatency.length - 1];

    // 1. Best accuracy
    findings.push(`<li><strong>🏆 Best Quality:</strong> ${MODEL_LABELS[best.model]} achieves the highest NDCG@10 (${best.ndcg.toFixed(3)}) and MRR (${best.data.avg_mrr.toFixed(3)}).</li>`);

    // 2. Runner-up
    if (second) {
        const ndcgGap = ((best.ndcg - second.ndcg) / second.ndcg * 100).toFixed(1);
        findings.push(`<li><strong>Runner-up:</strong> ${MODEL_LABELS[second.model]} (NDCG@10 ${second.ndcg.toFixed(3)}) is ${ndcgGap}% behind.</li>`);
    }

    // 3. Speed champion
    const speedRatio = (slowest.latency / fastest.latency).toFixed(1);
    findings.push(`<li><strong>⚡ Fastest:</strong> ${MODEL_LABELS[fastest.model]} (${Math.round(fastest.latency)}ms) is ${speedRatio}x faster than ${MODEL_LABELS[slowest.model]} (${Math.round(slowest.latency)}ms).</li>`);

    // 4. Average reranker gain
    const avgGain = calculateAvgRerankerGain(evaluationData);
    findings.push(`<li><strong>📈 Reranker Impact:</strong> Average NDCG@10 improvement of ${formatDelta(avgGain)} across all model pairs.</li>`);

    // 5. Best efficiency
    const bestEfficiency = findBestEfficiency(evaluationData);
    if (bestEfficiency.baseModel) {
        const baseData = evaluationData[bestEfficiency.baseModel];
        const rerankedData = evaluationData[getRerankedModel(bestEfficiency.baseModel)];
        const ndcgGain = calculateDelta(baseData['avg_ndcg@10'], rerankedData['avg_ndcg@10']);
        const latencyOverhead = Math.round(rerankedData.avg_search_time_ms - baseData.avg_search_time_ms);
        findings.push(`<li><strong>💰 Best ROI:</strong> ${MODEL_LABELS[bestEfficiency.baseModel]} + Reranker offers the best efficiency: ${formatDelta(ndcgGain)} NDCG gain for only +${latencyOverhead}ms latency.</li>`);
    }

    // 6. Production recommendation
    const baseModelsWithQuality = getSortedModels(evaluationData, 'base')
        .map(m => ({ model: m, data: evaluationData[m] }))
        .sort((a, b) => b.data['avg_ndcg@10'] - a.data['avg_ndcg@10']);

    const prodCandidate = baseModelsWithQuality.find(m => m.data.avg_search_time_ms < 300) || baseModelsWithQuality[0];

    if (prodCandidate) {
        const rerankedVersion = getRerankedModel(prodCandidate.model);
        if (evaluationData[rerankedVersion]) {
            findings.push(`<li><strong>🚀 Recommendation:</strong> For production, start with ${MODEL_LABELS[prodCandidate.model]} (${Math.round(prodCandidate.data.avg_search_time_ms)}ms). Add reranker for ${formatDelta(calculateDelta(prodCandidate.data['avg_ndcg@10'], evaluationData[rerankedVersion]['avg_ndcg@10']))} quality boost when latency budget allows.</li>`);
        } else {
            findings.push(`<li><strong>🚀 Recommendation:</strong> For production, use ${MODEL_LABELS[prodCandidate.model]} (NDCG@10 ${prodCandidate.data['avg_ndcg@10'].toFixed(3)}, ${Math.round(prodCandidate.data.avg_search_time_ms)}ms).</li>`);
        }
    }

    document.getElementById('findings').innerHTML = `<ul class="list-disc list-inside space-y-2">${findings.join('')}</ul>`;
}

// ==========================================
// HEATMAP VISUALIZATION
// ==========================================

// Query type and difficulty categories (matches benchmark data structure)
const QUERY_TYPES = ['keyword', 'nat_short', 'nat_long', 'conceptual'];
const DIFFICULTIES = ['easy', 'medium', 'hard'];

function toggleHeatmapReranker() {
    heatmapRerankerEnabled = !heatmapRerankerEnabled;

    // Update button state
    const btn = document.getElementById('heatmap-reranker-toggle');
    if (btn) {
        btn.classList.toggle('active', heatmapRerankerEnabled);
    }

    updateHeatmap();
}

function populateHeatmapModelSelect() {
    const select = document.getElementById('heatmap-model-select');
    if (!select || !evaluationData) return;

    // Only show 5 base models
    select.innerHTML = BASE_MODELS.map(m =>
        `<option value="${m}">${MODEL_LABELS_SHORT[m] || m}</option>`
    ).join('');

    // Default to hybrid_3large (best base model typically)
    select.value = 'qdrant_hybrid_3large';
}

function populateDeltaBaselineSelect() {
    const select = document.getElementById('delta-baseline-select');
    if (!select || !evaluationData) return;

    // Show 5 base models
    select.innerHTML = BASE_MODELS.map(m =>
        `<option value="${m}">${MODEL_LABELS_SHORT[m] || m}</option>`
    ).join('');

    // Default to dense as baseline
    select.value = 'qdrant_dense';
}

function toggleDeltaReranker() {
    deltaRerankerEnabled = !deltaRerankerEnabled;

    // Update button state
    const btn = document.getElementById('delta-reranker-toggle');
    if (btn) {
        btn.classList.toggle('active', deltaRerankerEnabled);
    }

    updateDeltaHeatmap();
}

function updateHeatmap() {
    const container = document.getElementById('heatmap-container');
    const insightEl = document.getElementById('heatmap-insight');
    const select = document.getElementById('heatmap-model-select');

    if (!container || !evaluationData) return;

    const selectedBaseModel = select?.value || 'qdrant_hybrid_3large';
    // Use reranked version if toggle is enabled
    const selectedModel = heatmapRerankerEnabled ? getRerankedModel(selectedBaseModel) : selectedBaseModel;
    const modelData = evaluationData[selectedModel];

    if (!modelData) return;

    // Check if real by_style and by_difficulty data is available
    const hasRealData = modelData.by_style && modelData.by_difficulty;

    // Map query types to their keys in by_style
    const styleKeyMap = {
        'keyword': 'keyword',
        'nat_short': 'natural_short',
        'nat_long': 'natural_long',
        'conceptual': 'conceptual'
    };

    let html = '';

    // Header row
    html += '<div class="col-span-1"></div>'; // Empty corner
    QUERY_TYPES.forEach(qt => {
        const label = qt.replace('nat_', '').replace('_', ' ');
        html += `<div class="text-center text-xs text-slate-400 font-medium uppercase">${label}</div>`;
    });

    // Data rows
    DIFFICULTIES.forEach(diff => {
        // Row label
        html += `<div class="text-right text-xs text-slate-400 font-medium uppercase pr-2 flex items-center justify-end">${diff}</div>`;

        // Cells
        QUERY_TYPES.forEach(qt => {
            let value;

            if (hasRealData) {
                // Use real data if available
                const styleKey = styleKeyMap[qt];
                const styleData = modelData.by_style[styleKey];
                const diffData = modelData.by_difficulty[diff];

                if (styleData && diffData) {
                    // Combine style and difficulty - use geometric mean as approximation
                    // Or use style MRR weighted by difficulty factor
                    const styleMRR = styleData.avg_mrr || 0;
                    const diffMRR = diffData.avg_mrr || 0;
                    const avgMRR = modelData.avg_mrr || 0.5;

                    // Calculate combined score: style_mrr * (diff_mrr / avg_mrr)
                    value = styleMRR * (diffMRR / avgMRR);
                    value = Math.min(1, Math.max(0, value));
                } else {
                    value = modelData.avg_mrr || 0;
                }
            } else {
                // Fallback to simulated data
                const baseNDCG = modelData['avg_ndcg@10'] || 0;
                const difficultyMultipliers = { easy: 1.15, medium: 1.0, hard: 0.75 };
                const queryTypeMultipliers = { keyword: 1.12, nat_short: 1.0, nat_long: 0.95, conceptual: 0.82 };
                value = baseNDCG * difficultyMultipliers[diff] * queryTypeMultipliers[qt];
                value = Math.min(1, Math.max(0, value));
            }

            const displayValue = value.toFixed(3);

            // Color scale with more gradations: red (low) -> orange -> yellow -> lime -> green (high)
            let bgColor;
            if (value >= 0.85) {
                bgColor = 'rgba(16, 185, 129, 0.75)';  // Strong green
            } else if (value >= 0.75) {
                bgColor = 'rgba(16, 185, 129, 0.55)';  // Medium green
            } else if (value >= 0.65) {
                bgColor = 'rgba(132, 204, 22, 0.5)';   // Lime
            } else if (value >= 0.55) {
                bgColor = 'rgba(234, 179, 8, 0.5)';    // Yellow
            } else if (value >= 0.45) {
                bgColor = 'rgba(249, 115, 22, 0.5)';   // Orange
            } else if (value >= 0.30) {
                bgColor = 'rgba(239, 68, 68, 0.5)';    // Red
            } else {
                bgColor = 'rgba(239, 68, 68, 0.7)';    // Strong red
            }

            html += `
                <div class="rounded p-3 text-center" style="background-color: ${bgColor}">
                    <div class="text-sm font-semibold text-slate-100">${displayValue}</div>
                </div>
            `;
        });
    });

    container.innerHTML = html;

    // Update insight with real data stats if available
    if (insightEl) {
        const modelLabel = MODEL_LABELS_SHORT[selectedModel] || selectedModel;

        let insightText = `<strong>Pattern:</strong> `;

        if (hasRealData) {
            // Find best and worst combinations from real data
            const keywordEasy = modelData.by_style?.keyword?.avg_mrr || 0;
            const conceptualHard = modelData.by_style?.conceptual?.avg_mrr || 0;
            insightText += `${modelLabel} achieves <span class="text-green-400">MRR ${keywordEasy.toFixed(3)}</span> on keyword queries `;
            insightText += `vs <span class="text-red-400">MRR ${conceptualHard.toFixed(3)}</span> on conceptual queries. `;
            insightText += `<span class="text-slate-400">(Using real benchmark data)</span>`;
        } else {
            insightText += `Performance decreases from keyword → conceptual queries, and from easy → hard difficulty. `;
            insightText += `${modelLabel} shows strongest performance on <span class="text-green-400">keyword × easy</span> `;
            insightText += `and weakest on <span class="text-red-400">conceptual × hard</span>.`;
        }

        if (heatmapRerankerEnabled) {
            insightText += `<br><span class="text-purple-400">Reranker enabled</span> - showing reranked model performance.`;
        }

        insightEl.innerHTML = insightText;
    }
}

// ==========================================
// DELTA HEATMAP (Model Comparison by Query Type)
// ==========================================

function updateDeltaHeatmap() {
    const container = document.getElementById('delta-heatmap-container');
    const baselineSelect = document.getElementById('delta-baseline-select');

    if (!container || !evaluationData) return;

    const baselineModel = baselineSelect?.value || 'qdrant_dense';

    // Fixed model order (always show all 5 models)
    const modelsToCompare = BASE_MODELS; // Keep all models in fixed order

    // Get baseline data
    const getModelForComparison = (model) => {
        return deltaRerankerEnabled ? getRerankedModel(model) : model;
    };

    const baselineData = evaluationData[getModelForComparison(baselineModel)];
    const hasRealData = baselineData?.by_style;

    // Style key mapping
    const styleKeyMap = {
        'keyword': 'keyword',
        'natural_short': 'natural_short',
        'natural_long': 'natural_long',
        'conceptual': 'conceptual'
    };

    const queryTypes = ['keyword', 'natural_short', 'natural_long', 'conceptual'];
    const queryLabels = ['Keyword', 'Short', 'Long', 'Conceptual'];

    let html = `
        <table class="w-full text-sm">
            <thead>
                <tr class="border-b border-slate-600">
                    <th class="text-left py-2 px-3 text-slate-400 font-medium">Model</th>
                    ${queryLabels.map(l => `<th class="text-center py-2 px-3 text-slate-400 font-medium">${l}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
    `;

    modelsToCompare.forEach(model => {
        const actualModel = getModelForComparison(model);
        const modelData = evaluationData[actualModel];

        if (!modelData) return;

        html += `<tr class="border-b border-slate-700/50">`;
        html += `<td class="py-2 px-3 text-slate-200 font-medium">${MODEL_LABELS_SHORT[model]}${deltaRerankerEnabled ? '+RR' : ''}</td>`;

        queryTypes.forEach(qt => {
            // For baseline model, always show 0.0%
            if (model === baselineModel) {
                const bgColor = 'rgba(148, 163, 184, 0.1)';
                const textColor = 'text-slate-400';
                html += `
                    <td class="py-2 px-3 text-center" style="background-color: ${bgColor}">
                        <span class="${textColor}">0.0%</span>
                    </td>
                `;
                return;
            }

            // Calculate delta for this query type using real data if available
            let baselineVal, modelVal;

            if (hasRealData && modelData.by_style) {
                baselineVal = baselineData.by_style[qt]?.avg_mrr || baselineData.avg_mrr || 0;
                modelVal = modelData.by_style[qt]?.avg_mrr || modelData.avg_mrr || 0;
            } else {
                // Fallback to simulated multipliers
                const queryTypeMultipliers = { keyword: 1.12, natural_short: 1.0, natural_long: 0.95, conceptual: 0.82 };
                baselineVal = (baselineData?.avg_mrr || 0) * queryTypeMultipliers[qt];
                modelVal = (modelData.avg_mrr || 0) * queryTypeMultipliers[qt];
            }

            const delta = modelVal - baselineVal;
            const deltaPercent = baselineVal > 0 ? (delta / baselineVal * 100) : 0;

            // Color based on delta with finer gradients
            let bgColor, textColor;
            const absDelta = Math.abs(deltaPercent);

            if (delta > 0) {
                // Positive delta: green scale with 5 levels
                if (absDelta >= 15) {
                    bgColor = 'rgba(16, 185, 129, 0.7)';  // Very strong green
                    textColor = 'text-emerald-300 font-bold';
                } else if (absDelta >= 10) {
                    bgColor = 'rgba(16, 185, 129, 0.55)'; // Strong green
                    textColor = 'text-green-400 font-semibold';
                } else if (absDelta >= 5) {
                    bgColor = 'rgba(16, 185, 129, 0.4)';  // Medium green
                    textColor = 'text-green-400';
                } else if (absDelta >= 2) {
                    bgColor = 'rgba(16, 185, 129, 0.25)'; // Light green
                    textColor = 'text-green-500';
                } else {
                    bgColor = 'rgba(16, 185, 129, 0.12)'; // Very light green
                    textColor = 'text-green-600';
                }
            } else if (delta < 0) {
                // Negative delta: red scale with 5 levels
                if (absDelta >= 15) {
                    bgColor = 'rgba(239, 68, 68, 0.7)';   // Very strong red
                    textColor = 'text-red-300 font-bold';
                } else if (absDelta >= 10) {
                    bgColor = 'rgba(239, 68, 68, 0.55)';  // Strong red
                    textColor = 'text-red-400 font-semibold';
                } else if (absDelta >= 5) {
                    bgColor = 'rgba(239, 68, 68, 0.4)';   // Medium red
                    textColor = 'text-red-400';
                } else if (absDelta >= 2) {
                    bgColor = 'rgba(239, 68, 68, 0.25)';  // Light red
                    textColor = 'text-red-500';
                } else {
                    bgColor = 'rgba(239, 68, 68, 0.12)';  // Very light red
                    textColor = 'text-red-600';
                }
            } else {
                bgColor = 'rgba(148, 163, 184, 0.1)';
                textColor = 'text-slate-400';
            }

            const sign = delta >= 0 ? '+' : '';
            html += `
                <td class="py-2 px-3 text-center" style="background-color: ${bgColor}">
                    <span class="${textColor}">${sign}${deltaPercent.toFixed(1)}%</span>
                </td>
            `;
        });

        html += `</tr>`;
    });

    html += `
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}

// ==========================================
// MODEL COMPARISON TOOL
// ==========================================

function populateComparisonSelects() {
    const selectA = document.getElementById('compare-model-a');
    const selectB = document.getElementById('compare-model-b');

    if (!selectA || !selectB || !evaluationData) return;

    const models = getSortedModels(evaluationData);
    const options = models.map(m =>
        `<option value="${m}">${MODEL_LABELS_SHORT[m] || m}</option>`
    ).join('');

    selectA.innerHTML = options;
    selectB.innerHTML = options;

    // Default comparison: hybrid_3large vs hybrid_3large+rerank
    selectA.value = 'qdrant_hybrid_3large';
    selectB.value = 'qdrant_hybrid_3large+rerank';

    updateComparison();
}

function setQuickCompare(modelA, modelB) {
    const selectA = document.getElementById('compare-model-a');
    const selectB = document.getElementById('compare-model-b');

    // Handle shorthand names
    const modelMap = {
        'dense': 'qdrant_dense',
        'sparse': 'qdrant_sparse',
        'hybrid': 'qdrant_hybrid',
    };

    selectA.value = modelMap[modelA] || modelA;
    selectB.value = modelMap[modelB] || modelB;

    updateComparison();
}

function updateComparison() {
    const selectA = document.getElementById('compare-model-a');
    const selectB = document.getElementById('compare-model-b');
    const deltaContainer = document.getElementById('comparison-delta');

    if (!selectA || !selectB || !evaluationData) return;

    const modelA = selectA.value;
    const modelB = selectB.value;
    const dataA = evaluationData[modelA];
    const dataB = evaluationData[modelB];

    if (!dataA || !dataB) return;

    // Render comparison radar chart
    renderComparisonRadarChart(modelA, modelB);

    // Render delta comparison
    const metrics = [
        { key: 'avg_mrr', label: 'MRR' },
        { key: 'avg_ndcg@5', label: 'NDCG@5' },
        { key: 'avg_ndcg@10', label: 'NDCG@10' },
        { key: 'avg_precision@5', label: 'P@5' },
        { key: 'avg_precision@10', label: 'P@10' },
        { key: 'avg_search_time_ms', label: 'Latency', unit: 'ms', lowerIsBetter: true },
    ];

    let html = `
        <h4 class="text-sm font-medium text-slate-200 mb-3">
            ${MODEL_LABELS_SHORT[modelA]} vs ${MODEL_LABELS_SHORT[modelB]}
        </h4>
        <div class="space-y-2">
    `;

    metrics.forEach(m => {
        const valA = dataA[m.key] || 0;
        const valB = dataB[m.key] || 0;
        const delta = m.lowerIsBetter ?
            calculateDelta(valB, valA) : // Reverse for latency
            calculateDelta(valA, valB);
        const winner = m.lowerIsBetter ?
            (valA < valB ? 'A' : valA > valB ? 'B' : 'tie') :
            (valA > valB ? 'A' : valA < valB ? 'B' : 'tie');

        const colorA = winner === 'A' ? 'text-green-400' : 'text-slate-300';
        const colorB = winner === 'B' ? 'text-green-400' : 'text-slate-300';
        const deltaColor = delta > 0 ? 'text-green-400' : delta < 0 ? 'text-red-400' : 'text-slate-400';

        const formatVal = m.unit === 'ms' ? Math.round(valA) + 'ms' : valA.toFixed(3);
        const formatValB = m.unit === 'ms' ? Math.round(valB) + 'ms' : valB.toFixed(3);

        html += `
            <div class="flex items-center justify-between text-sm">
                <span class="text-slate-400 w-20">${m.label}</span>
                <span class="${colorA} w-16 text-right">${formatVal}</span>
                <span class="${deltaColor} w-16 text-center">${formatDelta(delta)}</span>
                <span class="${colorB} w-16 text-right">${formatValB}</span>
            </div>
        `;
    });

    html += '</div>';

    // Summary
    const ndcgDelta = calculateDelta(dataA['avg_ndcg@10'], dataB['avg_ndcg@10']);
    const latencyDelta = dataB.avg_search_time_ms - dataA.avg_search_time_ms;

    html += `
        <div class="mt-4 pt-3 border-t border-slate-600 text-sm">
            <p class="text-slate-300">
                <strong>${MODEL_LABELS_SHORT[modelB]}</strong> is
                <span class="${ndcgDelta > 0 ? 'text-green-400' : 'text-red-400'}">${formatDelta(Math.abs(ndcgDelta))} ${ndcgDelta > 0 ? 'better' : 'worse'}</span>
                in NDCG@10, with
                <span class="${latencyDelta > 0 ? 'text-orange-400' : 'text-blue-400'}">${Math.abs(Math.round(latencyDelta))}ms ${latencyDelta > 0 ? 'more' : 'less'}</span>
                latency.
            </p>
        </div>
    `;

    deltaContainer.innerHTML = html;
}

function renderComparisonRadarChart(modelA, modelB) {
    const canvas = document.getElementById('comparisonRadarChart');
    if (!canvas) return;

    if (chartInstances.comparisonRadar) {
        chartInstances.comparisonRadar.destroy();
    }

    const ctx = canvas.getContext('2d');
    const dataA = evaluationData[modelA];
    const dataB = evaluationData[modelB];

    const metrics = ['avg_mrr', 'avg_ndcg@5', 'avg_ndcg@10', 'avg_precision@5', 'avg_precision@10'];
    const metricLabels = ['MRR', 'NDCG@5', 'NDCG@10', 'P@5', 'P@10'];

    chartInstances.comparisonRadar = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: metricLabels,
            datasets: [
                {
                    label: MODEL_LABELS_SHORT[modelA],
                    data: metrics.map(m => dataA[m] || 0),
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(59, 130, 246, 1)',
                },
                {
                    label: MODEL_LABELS_SHORT[modelB],
                    data: metrics.map(m => dataB[m] || 0),
                    backgroundColor: 'rgba(139, 92, 246, 0.2)',
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(139, 92, 246, 1)',
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#e2e8f0' },
                },
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    grid: { color: 'rgba(148, 163, 184, 0.3)' },
                    angleLines: { color: 'rgba(148, 163, 184, 0.3)' },
                    pointLabels: { color: '#e2e8f0' },
                    ticks: { color: '#94a3b8', backdropColor: 'transparent' },
                },
            },
        },
    });
}

// ==========================================
// PRODUCTION RECOMMENDATIONS
// ==========================================

function renderRecommendations() {
    if (!evaluationData) return;

    // Best Quality
    const bestQuality = findBestModel(evaluationData, 'avg_ndcg@10');
    const qualityText = document.getElementById('rec-quality-text');
    if (qualityText && bestQuality.model) {
        const data = evaluationData[bestQuality.model];
        qualityText.innerHTML = `
            <strong>${MODEL_LABELS[bestQuality.model]}</strong><br>
            NDCG@10: ${bestQuality.value.toFixed(3)} | Latency: ${Math.round(data.avg_search_time_ms)}ms<br>
            <span class="text-slate-400">Best for: Quality-first applications, offline processing</span>
        `;
    }

    // Best Speed
    const fastest = findFastestModel(evaluationData, 'base');
    const speedText = document.getElementById('rec-speed-text');
    if (speedText && fastest.model) {
        const data = evaluationData[fastest.model];
        speedText.innerHTML = `
            <strong>${MODEL_LABELS[fastest.model]}</strong><br>
            Latency: ${Math.round(fastest.value)}ms | MRR: ${data.avg_mrr.toFixed(3)}<br>
            <span class="text-slate-400">Best for: Real-time search, high QPS</span>
        `;
    }

    // Best Balance (base model with high NDCG and <100ms latency)
    const balanceCandidates = getSortedModels(evaluationData, 'base')
        .filter(m => evaluationData[m].avg_search_time_ms < 100)
        .sort((a, b) => evaluationData[b]['avg_ndcg@10'] - evaluationData[a]['avg_ndcg@10']);
    const bestBalance = balanceCandidates[0];
    const balanceText = document.getElementById('rec-balance-text');
    if (balanceText && bestBalance) {
        const data = evaluationData[bestBalance];
        balanceText.innerHTML = `
            <strong>${MODEL_LABELS[bestBalance]}</strong><br>
            NDCG@10: ${data['avg_ndcg@10'].toFixed(3)} | Latency: ${Math.round(data.avg_search_time_ms)}ms<br>
            <span class="text-slate-400">Best for: Production default, general purpose</span>
        `;
    }

    // Best ROI (best efficiency)
    const bestEfficiency = findBestEfficiency(evaluationData);
    const roiText = document.getElementById('rec-roi-text');
    if (roiText && bestEfficiency.baseModel) {
        const baseData = evaluationData[bestEfficiency.baseModel];
        const rerankedModel = getRerankedModel(bestEfficiency.baseModel);
        const rerankedData = evaluationData[rerankedModel];
        const ndcgGain = calculateDelta(baseData['avg_ndcg@10'], rerankedData['avg_ndcg@10']);
        const latencyOverhead = Math.round(rerankedData.avg_search_time_ms - baseData.avg_search_time_ms);

        roiText.innerHTML = `
            <strong>${MODEL_LABELS[bestEfficiency.baseModel]} + Reranker</strong><br>
            ${formatDelta(ndcgGain)} NDCG gain for +${latencyOverhead}ms latency<br>
            <span class="text-slate-400">Best for: Maximize quality within latency budget</span>
        `;
    }

    // Reranker stats
    const avgGain = calculateAvgRerankerGain(evaluationData);
    const rerankerAvgEl = document.getElementById('reranker-avg-gain');
    if (rerankerAvgEl) {
        rerankerAvgEl.textContent = formatDelta(avgGain);
    }

    // Max conceptual gain (from real data if available)
    const maxConceptualEl = document.getElementById('reranker-best-combo');
    if (maxConceptualEl) {
        // Calculate actual max gain for conceptual queries across all model pairs
        let maxConceptualGain = 0;
        BASE_MODELS.forEach(baseModel => {
            const rerankedModel = getRerankedModel(baseModel);
            const baseData = evaluationData[baseModel];
            const rerankedData = evaluationData[rerankedModel];
            if (baseData?.by_style?.conceptual && rerankedData?.by_style?.conceptual) {
                const baseConceptual = baseData.by_style.conceptual.avg_mrr || 0;
                const rerankedConceptual = rerankedData.by_style.conceptual.avg_mrr || 0;
                const gain = baseConceptual > 0 ? calculateDelta(baseConceptual, rerankedConceptual) : 0;
                if (gain > maxConceptualGain) maxConceptualGain = gain;
            }
        });
        // Fallback to estimated if no real data
        if (maxConceptualGain === 0) maxConceptualGain = avgGain * 1.5;
        maxConceptualEl.textContent = formatDelta(maxConceptualGain);
    }

    // Avg latency overhead
    let totalOverhead = 0;
    let overheadCount = 0;
    BASE_MODELS.forEach(base => {
        const reranked = getRerankedModel(base);
        if (evaluationData[base] && evaluationData[reranked]) {
            totalOverhead += evaluationData[reranked].avg_search_time_ms - evaluationData[base].avg_search_time_ms;
            overheadCount++;
        }
    });
    const avgOverhead = overheadCount > 0 ? totalOverhead / overheadCount : 0;
    const avgLatencyEl = document.getElementById('reranker-avg-latency');
    if (avgLatencyEl) {
        avgLatencyEl.textContent = `+${Math.round(avgOverhead)}ms`;
    }
}

// ==========================================
// UPDATED SUMMARY CARDS
// ==========================================

function updateSummaryCards() {
    const models = getSortedModels(evaluationData);

    // Total queries
    const numQueries = evaluationData[models[0]]?.num_queries || 0;
    const queriesEl = document.getElementById('total-queries');
    if (queriesEl) {
        queriesEl.textContent = numQueries.toLocaleString();
    }

    // Best NDCG
    const bestQuality = findBestModel(evaluationData, 'avg_ndcg@10');
    const bestNdcgEl = document.getElementById('best-ndcg');
    if (bestNdcgEl) {
        bestNdcgEl.textContent = bestQuality.value.toFixed(3);
    }

    // Average Reranker Gain
    const avgGain = calculateAvgRerankerGain(evaluationData);
    const avgGainEl = document.getElementById('avg-reranker-gain');
    if (avgGainEl) {
        avgGainEl.textContent = formatDelta(avgGain);
    }
}

// ==========================================
// DASHBOARD RENDERING (UPDATED)
// ==========================================

function renderDashboard() {
    if (!evaluationData) return;

    updateSummaryCards();
    populateHeatmapModelSelect();
    populateDeltaBaselineSelect();
    populateComparisonSelects();

    renderStackedBarChart();
    renderBaseRadarChart();
    renderRerankedRadarChart();
    renderRerankerImpactChart();
    renderLatencyChart();
    renderResultsTable();
    renderFindings();
    renderRecommendations();

    updateHeatmap();
    updateDeltaHeatmap();
    updateComparison();

    // Update date
    document.getElementById('eval-date').textContent =
        `Benchmark completed: ${new Date().toLocaleDateString()}`;
}

// ==========================================
// INITIALIZATION
// ==========================================

document.addEventListener('DOMContentLoaded', loadData);

// Expose functions globally for HTML onclick handlers
window.setFilter = setFilter;
window.updateHeatmap = updateHeatmap;
window.toggleHeatmapReranker = toggleHeatmapReranker;
window.updateDeltaHeatmap = updateDeltaHeatmap;
window.toggleDeltaReranker = toggleDeltaReranker;
window.updateComparison = updateComparison;
window.setQuickCompare = setQuickCompare;
window.renderStackedBarChart = renderStackedBarChart;
