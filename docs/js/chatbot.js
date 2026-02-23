// arXiv RAG Inline Chatbot Client (Dark Mode)
console.log('[Chatbot] Script loaded');

// API URL configuration - use FastAPI backend via Cloudflare Tunnel
const CHAT_API_URL = 'https://api.acacia.chat/api/v1/chat';

// Fallback for local development
const CHAT_API_URL_LOCAL = 'http://localhost:8000/api/v1/chat';

// Determine which URL to use based on environment
function getChatApiUrl() {
    // Use local URL if running on localhost
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return CHAT_API_URL_LOCAL;
    }
    return CHAT_API_URL;
}

class InlineChatbot {
  constructor() {
    this.history = [];
    this.isLoading = false;
    this.init();
  }

  init() {
    // Cache DOM elements
    this.messages = document.getElementById('inline-chat-messages');
    this.input = document.getElementById('inline-chat-input');
    this.sendBtn = document.getElementById('inline-chat-send');
    this.status = document.getElementById('inline-chat-status');
    this.metrics = document.getElementById('inline-chat-metrics');

    console.log('[Chatbot] Elements found:', {
      messages: !!this.messages,
      input: !!this.input,
      sendBtn: !!this.sendBtn
    });

    if (!this.messages || !this.input || !this.sendBtn) {
      console.warn('Inline chat elements not found');
      return;
    }

    this.bindEvents();
    console.log('[Chatbot] Events bound successfully');
  }

  bindEvents() {
    this.sendBtn.addEventListener('click', () => this.sendMessage());
    this.input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });
  }

  async sendMessage() {
    console.log('[Chatbot] sendMessage called');
    const query = this.input.value.trim();
    console.log('[Chatbot] Query:', query, 'isLoading:', this.isLoading);
    if (!query || this.isLoading) return;

    // Add user message
    this.addMessage(query, 'user');
    this.input.value = '';
    this.isLoading = true;
    this.sendBtn.disabled = true;
    this.status.textContent = 'Searching papers...';

    try {
      const response = await fetch(getChatApiUrl(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          search_mode: 'hybrid',
          embedding_model: 'openai',
          history: this.history.slice(-6).map(msg => ({
            role: msg.role,
            content: msg.content
          })),
          top_k: 5,
          stream: false,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Request failed');
      }

      const data = await response.json();

      // Add assistant message with sources
      this.addMessage(data.answer, 'assistant', data.sources);

      // Update history
      this.history.push(
        { role: 'user', content: query },
        { role: 'assistant', content: data.answer }
      );

      // Show metrics
      this.metrics.textContent = `${data.metrics.total_time_ms}ms | ${data.metrics.chunks_found} sources`;

    } catch (error) {
      console.error('Chat error:', error);
      this.addMessage(`Error: ${error.message}`, 'error');
    } finally {
      this.isLoading = false;
      this.sendBtn.disabled = false;
      this.status.textContent = '';
    }
  }

  addMessage(content, type, sources = []) {
    const messageDiv = document.createElement('div');

    if (type === 'user') {
      messageDiv.className = 'flex justify-end';
      messageDiv.innerHTML = `
        <div class="bg-blue-600 text-white rounded-lg px-4 py-3 max-w-[80%]">
          ${this.escapeHtml(content)}
        </div>
      `;
    } else if (type === 'assistant') {
      messageDiv.className = 'flex flex-col space-y-2';

      const formattedContent = this.formatAnswer(content);

      let sourcesHTML = '';
      if (sources.length > 0) {
        sourcesHTML = `
          <div class="mt-3 pt-3 border-t border-slate-600">
            <p class="text-xs text-slate-400 font-medium mb-2">Sources:</p>
            <div class="space-y-1">
              ${sources.map((s, i) => `
                <a href="https://arxiv.org/abs/${s.paper_id}" target="_blank"
                   class="block text-xs text-blue-400 hover:text-blue-300 hover:underline truncate">
                  [${i + 1}] ${this.escapeHtml(s.title)} (${(s.similarity * 100).toFixed(0)}% match)
                </a>
              `).join('')}
            </div>
          </div>
        `;
      }

      messageDiv.innerHTML = `
        <div class="bg-slate-700 rounded-lg px-4 py-3 max-w-[90%]">
          <div class="prose prose-sm prose-invert max-w-none text-slate-200">${formattedContent}</div>
          ${sourcesHTML}
        </div>
      `;
    } else if (type === 'error') {
      messageDiv.className = 'flex justify-center';
      messageDiv.innerHTML = `
        <div class="bg-red-900/50 text-red-300 border border-red-700 rounded-lg px-4 py-2 text-sm">
          ${this.escapeHtml(content)}
        </div>
      `;
    }

    this.messages.appendChild(messageDiv);
    this.messages.scrollTop = this.messages.scrollHeight;
  }

  formatAnswer(text) {
    // Convert [1], [2] references to superscript
    let formatted = text.replace(/\[(\d+)\]/g, '<sup class="text-blue-400">[$1]</sup>');

    // Convert **bold** to <strong>
    formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong class="text-slate-100">$1</strong>');

    // Convert *italic* to <em>
    formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Convert bullet points
    formatted = formatted.replace(/^[•\-]\s/gm, '• ');

    // Convert newlines to <br>
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
  }

  escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize inline chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  console.log('[Chatbot] Initializing...');
  window.inlineChatbot = new InlineChatbot();
  console.log('[Chatbot] Initialized:', window.inlineChatbot);
});
