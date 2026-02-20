"""
arXiv RAG v1 - Embedding Module

Chunking and embedding pipeline for RAG system.
"""

from .models import (
    Chunk,
    ChunkType,
    ChunkingConfig,
    ChunkingStats,
    ColBERTVector,
    EmbeddedChunk,
    EmbeddingConfig,
    EmbeddingStats,
    SparseVector,
)

from .chunker import (
    HybridChunker,
    chunk_papers,
)

from .bge_embedder import (
    BGEEmbedder,
    embed_chunks_bge,
)

from .openai_embedder import (
    OpenAIEmbedder,
    embed_chunks_openai,
    embed_chunks_openai_async,
)


__all__ = [
    # Models
    "Chunk",
    "ChunkType",
    "ChunkingConfig",
    "ChunkingStats",
    "ColBERTVector",
    "EmbeddedChunk",
    "EmbeddingConfig",
    "EmbeddingStats",
    "SparseVector",
    # Chunker
    "HybridChunker",
    "chunk_papers",
    # BGE Embedder
    "BGEEmbedder",
    "embed_chunks_bge",
    # OpenAI Embedder
    "OpenAIEmbedder",
    "embed_chunks_openai",
    "embed_chunks_openai_async",
]
