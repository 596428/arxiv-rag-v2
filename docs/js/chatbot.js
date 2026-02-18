// arXiv RAG Chatbot Client
const CHAT_API_URL = 'https://wfkectgpoifwbgyjslcl.supabase.co/functions/v1/chat';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indma2VjdGdwb2lmd2JneWpzbGNsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA3Nzk4MjIsImV4cCI6MjA4NjM1NTgyMn0.xHR7kXUleU9ZPRmg4PoXfebr3iJUYXTwHnQwzZ5qEZs';

class ArxivChatbot {
  constructor() {
    this.history = [];
    this.isOpen = false;
    this.isLoading = false;
    this.init();
  }

  init() {
    // Create chatbot UI
    this.createUI();
    this.bindEvents();
  }

  createUI() {
    const chatbotHTML = `
      <div id="chatbot-container" class="fixed bottom-4 right-4 z-50">
        <!-- Toggle Button -->
        <button id="chatbot-toggle" class="w-14 h-14 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full shadow-lg flex items-center justify-center hover:scale-110 transition-transform">
          <svg id="chat-icon" class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path>
          </svg>
          <svg id="close-icon" class="w-6 h-6 text-white hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        </button>

        <!-- Chat Panel -->
        <div id="chat-panel" class="hidden absolute bottom-16 right-0 w-96 h-[500px] bg-white rounded-lg shadow-2xl flex flex-col overflow-hidden">
          <!-- Header -->
          <div class="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4">
            <h3 class="font-semibold">arXiv RAG Assistant</h3>
            <p class="text-xs text-blue-100">Ask questions about AI/ML research papers</p>
          </div>

          <!-- Messages -->
          <div id="chat-messages" class="flex-1 overflow-y-auto p-4 space-y-4">
            <div class="bg-gray-100 rounded-lg p-3 text-sm">
              <p>Hello! I can help you explore AI and machine learning research papers. Ask me anything about:</p>
              <ul class="list-disc list-inside mt-2 text-gray-600 text-xs">
                <li>Reinforcement learning from human feedback (RLHF)</li>
                <li>Large language models and transformers</li>
                <li>Prompt engineering techniques</li>
                <li>AI safety and alignment</li>
              </ul>
            </div>
          </div>

          <!-- Input Area -->
          <div class="border-t p-4">
            <div class="flex space-x-2">
              <input
                type="text"
                id="chat-input"
                placeholder="Ask about AI research..."
                class="flex-1 border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
              <button
                id="chat-send"
                class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                </svg>
              </button>
            </div>
            <div class="flex items-center justify-between mt-2 text-xs text-gray-500">
              <span id="chat-status"></span>
              <span id="chat-metrics"></span>
            </div>
          </div>
        </div>
      </div>
    `;

    document.body.insertAdjacentHTML('beforeend', chatbotHTML);

    // Cache DOM elements
    this.container = document.getElementById('chatbot-container');
    this.toggle = document.getElementById('chatbot-toggle');
    this.panel = document.getElementById('chat-panel');
    this.messages = document.getElementById('chat-messages');
    this.input = document.getElementById('chat-input');
    this.sendBtn = document.getElementById('chat-send');
    this.status = document.getElementById('chat-status');
    this.metrics = document.getElementById('chat-metrics');
    this.chatIcon = document.getElementById('chat-icon');
    this.closeIcon = document.getElementById('close-icon');
  }

  bindEvents() {
    this.toggle.addEventListener('click', () => this.toggleChat());
    this.sendBtn.addEventListener('click', () => this.sendMessage());
    this.input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });
  }

  toggleChat() {
    this.isOpen = !this.isOpen;
    this.panel.classList.toggle('hidden', !this.isOpen);
    this.chatIcon.classList.toggle('hidden', this.isOpen);
    this.closeIcon.classList.toggle('hidden', !this.isOpen);

    if (this.isOpen) {
      this.input.focus();
    }
  }

  async sendMessage() {
    const query = this.input.value.trim();
    if (!query || this.isLoading) return;

    // Add user message
    this.addMessage(query, 'user');
    this.input.value = '';
    this.isLoading = true;
    this.sendBtn.disabled = true;
    this.status.textContent = 'Searching papers...';

    try {
      const response = await fetch(CHAT_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
        },
        body: JSON.stringify({
          query,
          embedding_model: 'openai',
          history: this.history.slice(-6), // Last 3 exchanges
          top_k: 5,
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
        <div class="bg-blue-600 text-white rounded-lg px-4 py-2 max-w-[80%] text-sm">
          ${this.escapeHtml(content)}
        </div>
      `;
    } else if (type === 'assistant') {
      messageDiv.className = 'flex flex-col space-y-2';

      // Format answer with markdown-like styling
      const formattedContent = this.formatAnswer(content);

      let sourcesHTML = '';
      if (sources.length > 0) {
        sourcesHTML = `
          <div class="mt-2 pt-2 border-t border-gray-200">
            <p class="text-xs text-gray-500 font-medium mb-1">Sources:</p>
            <div class="space-y-1">
              ${sources.map((s, i) => `
                <a href="https://arxiv.org/abs/${s.paper_id}" target="_blank"
                   class="block text-xs text-blue-600 hover:underline truncate">
                  [${i + 1}] ${s.title} (${(s.similarity * 100).toFixed(0)}% match)
                </a>
              `).join('')}
            </div>
          </div>
        `;
      }

      messageDiv.innerHTML = `
        <div class="bg-gray-100 rounded-lg px-4 py-3 max-w-[90%] text-sm">
          <div class="prose prose-sm max-w-none">${formattedContent}</div>
          ${sourcesHTML}
        </div>
      `;
    } else if (type === 'error') {
      messageDiv.className = 'flex justify-center';
      messageDiv.innerHTML = `
        <div class="bg-red-100 text-red-700 rounded-lg px-4 py-2 text-sm">
          ${this.escapeHtml(content)}
        </div>
      `;
    }

    this.messages.appendChild(messageDiv);
    this.messages.scrollTop = this.messages.scrollHeight;
  }

  formatAnswer(text) {
    // Convert [1], [2] references to superscript
    let formatted = text.replace(/\[(\d+)\]/g, '<sup class="text-blue-600">[$1]</sup>');

    // Convert **bold** to <strong>
    formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Convert bullet points
    formatted = formatted.replace(/^[•\-]\s/gm, '• ');

    // Convert newlines to <br>
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.arxivChatbot = new ArxivChatbot();
});
