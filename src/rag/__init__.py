"""
arXiv RAG v1 - RAG Module

Hybrid retrieval and search functionality.
"""

from .retriever import (
    SearchResult,
    SearchResponse,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    hybrid_search,
    dense_search,
    sparse_search,
)

from .reranker import (
    BGEReranker,
    LightweightReranker,
    rerank_results,
)

from .api import app as api_app

__all__ = [
    # Retriever
    "SearchResult",
    "SearchResponse",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "hybrid_search",
    "dense_search",
    "sparse_search",
    # Reranker
    "BGEReranker",
    "LightweightReranker",
    "rerank_results",
    # API
    "api_app",
]
