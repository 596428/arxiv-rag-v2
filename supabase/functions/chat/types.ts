// Types for arXiv RAG Chat Edge Function

export interface ChatRequest {
  query: string;
  embedding_model?: 'openai' | 'bge';
  history?: ChatMessage[];
  top_k?: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  metrics: Metrics;
}

export interface Source {
  paper_id: string;
  title: string;
  section: string;
  similarity: number;
  chunk_text: string;
}

export interface Metrics {
  embed_time_ms: number;
  search_time_ms: number;
  generate_time_ms: number;
  total_time_ms: number;
  chunks_found: number;
  embedding_model: string;
}

export interface ChunkMatch {
  id: string;
  paper_id: string;
  paper_title: string;
  section_title: string;
  chunk_text: string;
  similarity: number;
}

export interface EmbeddingResult {
  embedding: number[];
  time_ms: number;
}
