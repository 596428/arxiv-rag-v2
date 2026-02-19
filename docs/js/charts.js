// arXiv RAG v1 - Evaluation Dashboard Charts
console.log('[Charts] Script loaded');

// Chart.js Dark Mode Configuration
if (typeof Chart !== 'undefined') {
    Chart.defaults.color = '#e2e8f0';           // slate-200 for text
    Chart.defaults.borderColor = '#475569';     // slate-600 for borders
    console.log('[Charts] Chart.js configured');
} else {
    console.error('[Charts] Chart.js not loaded!');
}

const MODEL_COLORS = {
    'hybrid': 'rgba(59, 130, 246, 0.8)',   // blue
    'dense': 'rgba(16, 185, 129, 0.8)',    // green
    'sparse': 'rgba(245, 158, 11, 0.8)',   // amber
    'openai': 'rgba(139, 92, 246, 0.8)',   // purple
};

const MODEL_LABELS = {
    'hybrid': 'Hybrid (BGE-M3)',
    'dense': 'Dense (BGE-M3)',
    'sparse': 'Sparse (BGE-M3)',
    'openai': 'OpenAI text-embedding-3-large',
};

let evaluationData = null;

// Load evaluation data
async function loadData() {
    try {
        const response = await fetch('data/evaluation_results.json');
        evaluationData = await response.json();
        renderDashboard();
    } catch (error) {
        console.error('Failed to load evaluation data:', error);
        document.getElementById('findings').innerHTML = `
            <p class="text-red-500">Failed to load evaluation data. Make sure data/evaluation_results.json exists.</p>
        `;
    }
}

// Render all dashboard components
function renderDashboard() {
    if (!evaluationData) return;

    updateSummaryCards();
    renderMetricsChart();
    renderRadarChart();
    renderLatencyChart();
    renderResultsTable();
    renderFindings();

    // Update date
    document.getElementById('eval-date').textContent =
        `Last updated: ${new Date().toLocaleDateString()}`;
}

// Update summary cards
function updateSummaryCards() {
    const models = Object.keys(evaluationData);
    let bestModel = '';
    let bestMRR = 0;
    let bestNDCG = 0;
    let numQueries = 0;

    models.forEach(model => {
        const data = evaluationData[model];
        if (data.avg_mrr > bestMRR) {
            bestMRR = data.avg_mrr;
            bestModel = model;
        }
        if (data['avg_ndcg@10'] > bestNDCG) {
            bestNDCG = data['avg_ndcg@10'];
        }
        numQueries = data.num_queries || numQueries;
    });

    document.getElementById('total-queries').textContent = numQueries;
    document.getElementById('best-model').textContent = MODEL_LABELS[bestModel] || bestModel;
    document.getElementById('best-mrr').textContent = bestMRR.toFixed(3);
    document.getElementById('best-ndcg').textContent = bestNDCG.toFixed(3);
}

// Metrics comparison bar chart
function renderMetricsChart() {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    const models = Object.keys(evaluationData);

    const metrics = ['avg_mrr', 'avg_ndcg@10', 'avg_precision@10'];
    const metricLabels = ['MRR', 'NDCG@10', 'P@10'];

    const datasets = models.map(model => ({
        label: MODEL_LABELS[model] || model,
        data: metrics.map(m => evaluationData[model][m] || 0),
        backgroundColor: MODEL_COLORS[model] || 'rgba(156, 163, 175, 0.8)',
        borderColor: MODEL_COLORS[model]?.replace('0.8', '1') || 'rgba(156, 163, 175, 1)',
        borderWidth: 1,
    }));

    new Chart(ctx, {
        type: 'bar',
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
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    grid: { color: 'rgba(148, 163, 184, 0.2)' },
                    ticks: { color: '#94a3b8' },
                    title: {
                        display: true,
                        text: 'Score',
                        color: '#e2e8f0',
                    },
                },
                x: {
                    grid: { color: 'rgba(148, 163, 184, 0.2)' },
                    ticks: { color: '#94a3b8' },
                },
            },
        },
    });
}

// Radar chart for model profiles
function renderRadarChart() {
    const ctx = document.getElementById('radarChart').getContext('2d');
    const models = Object.keys(evaluationData);

    const metrics = ['avg_mrr', 'avg_ndcg@5', 'avg_ndcg@10', 'avg_precision@5', 'avg_precision@10'];
    const metricLabels = ['MRR', 'NDCG@5', 'NDCG@10', 'P@5', 'P@10'];

    const datasets = models.map(model => ({
        label: MODEL_LABELS[model] || model,
        data: metrics.map(m => evaluationData[model][m] || 0),
        backgroundColor: MODEL_COLORS[model]?.replace('0.8', '0.2') || 'rgba(156, 163, 175, 0.2)',
        borderColor: MODEL_COLORS[model] || 'rgba(156, 163, 175, 1)',
        borderWidth: 2,
        pointBackgroundColor: MODEL_COLORS[model] || 'rgba(156, 163, 175, 1)',
    }));

    new Chart(ctx, {
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
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    grid: {
                        color: 'rgba(148, 163, 184, 0.3)',
                    },
                    angleLines: {
                        color: 'rgba(148, 163, 184, 0.3)',
                    },
                    pointLabels: {
                        color: '#e2e8f0',
                    },
                    ticks: {
                        color: '#94a3b8',
                        backdropColor: 'transparent',
                    },
                },
            },
        },
    });
}

// Latency comparison chart
function renderLatencyChart() {
    const ctx = document.getElementById('latencyChart').getContext('2d');
    const models = Object.keys(evaluationData);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models.map(m => MODEL_LABELS[m] || m),
            datasets: [{
                label: 'Average Search Time (ms)',
                data: models.map(m => evaluationData[m].avg_search_time_ms || 0),
                backgroundColor: models.map(m => MODEL_COLORS[m] || 'rgba(156, 163, 175, 0.8)'),
                borderColor: models.map(m => MODEL_COLORS[m]?.replace('0.8', '1') || 'rgba(156, 163, 175, 1)'),
                borderWidth: 1,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: {
                    display: false,
                },
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

// Results table
function renderResultsTable() {
    const tbody = document.getElementById('results-table');
    const models = Object.keys(evaluationData);

    tbody.innerHTML = models.map(model => {
        const data = evaluationData[model];
        const isOpenAI = model === 'openai';
        const rowClass = isOpenAI ? 'bg-purple-900/30' : '';

        return `
            <tr class="${rowClass}">
                <td class="px-6 py-4 whitespace-nowrap font-medium ${isOpenAI ? 'text-purple-300' : 'text-slate-100'}">
                    ${MODEL_LABELS[model] || model}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-slate-300">${(data.avg_mrr || 0).toFixed(3)}</td>
                <td class="px-6 py-4 whitespace-nowrap text-slate-300">${(data['avg_ndcg@5'] || 0).toFixed(3)}</td>
                <td class="px-6 py-4 whitespace-nowrap text-slate-300">${(data['avg_ndcg@10'] || 0).toFixed(3)}</td>
                <td class="px-6 py-4 whitespace-nowrap text-slate-300">${(data['avg_precision@5'] || 0).toFixed(3)}</td>
                <td class="px-6 py-4 whitespace-nowrap text-slate-300">${(data['avg_precision@10'] || 0).toFixed(3)}</td>
                <td class="px-6 py-4 whitespace-nowrap text-slate-300">${Math.round(data.avg_search_time_ms || 0)}</td>
            </tr>
        `;
    }).join('');
}

// Key findings analysis
function renderFindings() {
    const models = Object.keys(evaluationData);
    const openai = evaluationData.openai;
    const dense = evaluationData.dense;
    const hybrid = evaluationData.hybrid;

    let findings = [];

    // Compare OpenAI vs BGE
    if (openai && dense) {
        const mrrImprovement = ((openai.avg_mrr - dense.avg_mrr) / dense.avg_mrr * 100).toFixed(1);
        const ndcgImprovement = ((openai['avg_ndcg@10'] - dense['avg_ndcg@10']) / dense['avg_ndcg@10'] * 100).toFixed(1);

        findings.push(`<li><strong>OpenAI vs BGE-M3 Dense:</strong> OpenAI shows ${mrrImprovement}% improvement in MRR and ${ndcgImprovement}% improvement in NDCG@10.</li>`);
    }

    // Latency comparison
    if (openai && dense) {
        const latencyRatio = (dense.avg_search_time_ms / openai.avg_search_time_ms).toFixed(1);
        if (openai.avg_search_time_ms < dense.avg_search_time_ms) {
            findings.push(`<li><strong>Latency:</strong> OpenAI embedding search is ${latencyRatio}x faster than BGE-M3 dense search.</li>`);
        }
    }

    // Hybrid vs Dense
    if (hybrid && dense) {
        if (hybrid.avg_mrr > dense.avg_mrr) {
            findings.push(`<li><strong>Hybrid Search:</strong> Combining dense and sparse retrieval (RRF) improves results over dense-only search.</li>`);
        } else {
            findings.push(`<li><strong>Hybrid Search:</strong> Hybrid search shows similar performance to dense-only search on this benchmark.</li>`);
        }
    }

    // Best model recommendation
    const bestModel = models.reduce((best, model) =>
        (evaluationData[model].avg_mrr > evaluationData[best].avg_mrr) ? model : best
    , models[0]);

    findings.push(`<li><strong>Recommendation:</strong> ${MODEL_LABELS[bestModel]} provides the best retrieval quality on this benchmark.</li>`);

    document.getElementById('findings').innerHTML = `<ul class="list-disc list-inside space-y-2">${findings.join('')}</ul>`;
}

// Initialize
document.addEventListener('DOMContentLoaded', loadData);
