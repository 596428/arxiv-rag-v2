// arXiv RAG Papers List with Pagination
console.log('[Papers] Script loaded');

const SUPABASE_URL = 'https://wfkectgpoifwbgyjslcl.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indma2VjdGdwb2lmd2JneWpzbGNsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA3Nzk4MjIsImV4cCI6MjA4NjM1NTgyMn0.xHR7kXUleU9ZPRmg4PoXfebr3iJUYXTwHnQwzZ5qEZs';

const PAGE_SIZE = 50;

// Track pagination state
window.papersLoaded = false;
window.papersData = [];
window.totalPapers = 0;
window.currentOffset = 0;
window.isLoadingMore = false;

// Get total paper count
async function getTotalCount() {
    const response = await fetch(
        `${SUPABASE_URL}/rest/v1/papers?select=count`,
        {
            headers: {
                'apikey': SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
                'Prefer': 'count=exact'
            }
        }
    );

    if (response.ok) {
        const data = await response.json();
        return data[0]?.count || 0;
    }
    return 0;
}

// Load papers from Supabase with pagination
async function loadPapers(append = false) {
    const tableBody = document.getElementById('papers-table');
    const countSpan = document.getElementById('papers-count');
    const loadMoreBtn = document.getElementById('load-more-btn');

    if (!append) {
        // Reset state for fresh load
        window.papersData = [];
        window.currentOffset = 0;

        // Show loading state
        tableBody.innerHTML = `
            <tr>
                <td colspan="4" class="px-4 py-8 text-center text-slate-400">
                    <div class="flex flex-col items-center">
                        <svg class="animate-spin h-8 w-8 text-blue-500 mb-2" fill="none" viewBox="0 0 24 24">
                            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Loading papers...
                    </div>
                </td>
            </tr>
        `;

        // Get total count
        window.totalPapers = await getTotalCount();
    }

    window.isLoadingMore = true;
    if (loadMoreBtn) {
        loadMoreBtn.disabled = true;
        loadMoreBtn.textContent = 'Loading...';
    }

    try {
        // Fetch papers with offset
        const response = await fetch(
            `${SUPABASE_URL}/rest/v1/papers?select=arxiv_id,title,abstract,published_date,citation_count&order=citation_count.desc.nullslast,published_date.desc&limit=${PAGE_SIZE}&offset=${window.currentOffset}`,
            {
                headers: {
                    'apikey': SUPABASE_ANON_KEY,
                    'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const papers = await response.json();
        window.papersData = [...window.papersData, ...papers];
        window.currentOffset += papers.length;
        window.papersLoaded = true;

        // Update count display
        countSpan.textContent = `${window.papersData.length} / ${window.totalPapers.toLocaleString()} papers`;

        if (window.papersData.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="4" class="px-4 py-8 text-center text-slate-400">
                        No papers found in database.
                    </td>
                </tr>
            `;
            if (loadMoreBtn) loadMoreBtn.style.display = 'none';
            return;
        }

        // Render papers table
        tableBody.innerHTML = window.papersData.map((paper, index) => {
            const date = paper.published_date
                ? new Date(paper.published_date).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric'
                  })
                : 'N/A';

            const safeTitle = escapeHtml(paper.title || 'Untitled');
            const truncatedTitle = safeTitle.length > 100
                ? safeTitle.substring(0, 100) + '...'
                : safeTitle;

            return `
                <tr class="hover:bg-slate-700 cursor-pointer transition-colors" onclick="showAbstract('${escapeHtml(paper.title?.replace(/'/g, "\\'") || '')}', '${date}', '${paper.arxiv_id}', \`${escapeHtml(paper.abstract?.replace(/`/g, "\\`") || '')}\`)">
                    <td class="px-4 py-3 text-sm text-slate-400">${index + 1}</td>
                    <td class="px-4 py-3">
                        <div class="text-sm font-medium text-slate-200">${truncatedTitle}</div>
                    </td>
                    <td class="px-4 py-3 text-sm text-slate-400 whitespace-nowrap">${date}</td>
                    <td class="px-4 py-3">
                        <a href="https://arxiv.org/abs/${paper.arxiv_id}"
                           target="_blank"
                           onclick="event.stopPropagation()"
                           class="text-blue-400 hover:text-blue-300 hover:underline text-sm">
                            ${paper.arxiv_id}
                        </a>
                    </td>
                </tr>
            `;
        }).join('');

        // Show/hide Load More button
        if (loadMoreBtn) {
            const remaining = window.totalPapers - window.papersData.length;
            if (remaining > 0) {
                loadMoreBtn.style.display = 'block';
                loadMoreBtn.textContent = `Load More (${Math.min(remaining, PAGE_SIZE)} more)`;
                loadMoreBtn.disabled = false;
            } else {
                loadMoreBtn.style.display = 'none';
            }
        }

    } catch (error) {
        console.error('Error loading papers:', error);
        if (!append) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="4" class="px-4 py-8 text-center text-red-400">
                        <div class="flex flex-col items-center">
                            <svg class="w-8 h-8 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                            Failed to load papers: ${escapeHtml(error.message)}
                            <button onclick="loadPapers()" class="mt-2 px-3 py-1 bg-red-900/50 text-red-300 rounded hover:bg-red-900 text-sm border border-red-700">
                                Retry
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        }
        countSpan.textContent = 'Error';
        if (loadMoreBtn) {
            loadMoreBtn.disabled = false;
            loadMoreBtn.textContent = 'Retry';
        }
    } finally {
        window.isLoadingMore = false;
    }
}

// Load more papers (called by button)
function loadMorePapers() {
    if (!window.isLoadingMore) {
        loadPapers(true);
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-load papers if on papers tab at page load
document.addEventListener('DOMContentLoaded', () => {
    if (window.location.hash === '#papers') {
        document.querySelector('[onclick="showTab(\'papers\')"]')?.click();
    }
});
