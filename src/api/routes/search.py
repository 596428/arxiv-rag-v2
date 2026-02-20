"""
arXiv RAG v1 - Search API Routes

Hybrid vector search endpoints using Qdrant.
"""

import time
from enum import Enum
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...embedding.bge_embedder import BGEEmbedder
from ...embedding.openai_embedder import OpenAIEmbedder
from ...embedding.models import EmbeddingConfig
from ...storage.qdrant_client import get_qdrant_client
from ...rag.reranker import BGEReranker
from ...utils.logging import get_logger

logger = get_logger("api.search")

router = APIRouter()


class SearchMode(str, Enum):
    """Search mode options."""
    dense = "dense"
    sparse = "sparse"
    hybrid = "hybrid"
    colbert = "colbert"
    openai = "openai"


class SearchRequest(BaseModel):
    """Search request payload."""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    mode: SearchMode = Field(default=SearchMode.hybrid, description="Search mode")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    rerank: bool = Field(default=False, description="Apply reranking")
    dense_weight: float = Field(default=0.5, ge=0, le=1, description="Dense weight for hybrid")
    sparse_weight: float = Field(default=0.5, ge=0, le=1, description="Sparse weight for hybrid")


class SearchResult(BaseModel):
    """Single search result."""
    chunk_id: str
    paper_id: str
    content: str
    section_title: Optional[str] = None
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    colbert_score: Optional[float] = None


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    mode: str
    results: list[SearchResult]
    total: int
    search_time_ms: float
    reranked: bool = False


# Lazy-loaded embedders
_bge_embedder: Optional[BGEEmbedder] = None
_openai_embedder: Optional[OpenAIEmbedder] = None
_reranker: Optional[BGEReranker] = None


def get_bge_embedder() -> BGEEmbedder:
    """Get or create BGE embedder."""
    global _bge_embedder
    if _bge_embedder is None:
        _bge_embedder = BGEEmbedder(EmbeddingConfig(use_openai=False))
    return _bge_embedder


def get_openai_embedder() -> OpenAIEmbedder:
    """Get or create OpenAI embedder."""
    global _openai_embedder
    if _openai_embedder is None:
        _openai_embedder = OpenAIEmbedder(EmbeddingConfig(use_openai=True))
    return _openai_embedder


def get_reranker() -> BGEReranker:
    """Get or create reranker."""
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Hybrid vector search.

    Supports multiple search modes:
    - dense: BGE-M3 dense embeddings
    - sparse: BGE-M3 sparse/lexical matching
    - hybrid: RRF fusion of dense + sparse
    - colbert: ColBERT late interaction (coming soon)
    - openai: OpenAI text-embedding-3-large

    Args:
        request: Search request with query and options

    Returns:
        Search results with scores
    """
    start_time = time.time()

    try:
        qdrant = get_qdrant_client()

        # Check Qdrant health
        if not qdrant.health_check():
            raise HTTPException(status_code=503, detail="Vector database unavailable")

        results = []

        if request.mode == SearchMode.dense:
            # Dense search with BGE-M3
            embedder = get_bge_embedder()
            dense_vec, _, _ = embedder.embed_single(request.query)

            raw_results = qdrant.search_dense(
                query_vector=dense_vec,
                vector_name="dense_bge",
                top_k=request.top_k * 2 if request.rerank else request.top_k,
            )

            results = [
                SearchResult(
                    chunk_id=r["chunk_id"],
                    paper_id=r["paper_id"],
                    content=r["content"],
                    section_title=r.get("section_title"),
                    score=r["similarity"],
                    dense_score=r["similarity"],
                )
                for r in raw_results
            ]

        elif request.mode == SearchMode.sparse:
            # Sparse search
            embedder = get_bge_embedder()
            _, sparse_vec, _ = embedder.embed_single(request.query)

            if sparse_vec:
                raw_results = qdrant.search_sparse(
                    query_indices=sparse_vec.indices,
                    query_values=sparse_vec.values,
                    top_k=request.top_k * 2 if request.rerank else request.top_k,
                )

                results = [
                    SearchResult(
                        chunk_id=r["chunk_id"],
                        paper_id=r["paper_id"],
                        content=r["content"],
                        section_title=r.get("section_title"),
                        score=r["score"],
                        sparse_score=r["score"],
                    )
                    for r in raw_results
                ]

        elif request.mode == SearchMode.hybrid:
            # Hybrid search with RRF fusion
            embedder = get_bge_embedder()
            dense_vec, sparse_vec, _ = embedder.embed_single(request.query)

            if sparse_vec:
                raw_results = qdrant.search_hybrid(
                    dense_vector=dense_vec,
                    sparse_indices=sparse_vec.indices,
                    sparse_values=sparse_vec.values,
                    top_k=request.top_k * 2 if request.rerank else request.top_k,
                    dense_weight=request.dense_weight,
                    sparse_weight=request.sparse_weight,
                )

                results = [
                    SearchResult(
                        chunk_id=r["chunk_id"],
                        paper_id=r["paper_id"],
                        content=r["content"],
                        section_title=r.get("section_title"),
                        score=r["score"],
                        dense_score=r.get("dense_score"),
                        sparse_score=r.get("sparse_score"),
                    )
                    for r in raw_results
                ]
            else:
                # Fallback to dense only
                raw_results = qdrant.search_dense(
                    query_vector=dense_vec,
                    vector_name="dense_bge",
                    top_k=request.top_k,
                )
                results = [
                    SearchResult(
                        chunk_id=r["chunk_id"],
                        paper_id=r["paper_id"],
                        content=r["content"],
                        section_title=r.get("section_title"),
                        score=r["similarity"],
                        dense_score=r["similarity"],
                    )
                    for r in raw_results
                ]

        elif request.mode == SearchMode.openai:
            # OpenAI embedding search
            embedder = get_openai_embedder()
            dense_vec = embedder.embed_single(request.query)

            raw_results = qdrant.search_dense(
                query_vector=dense_vec,
                vector_name="dense_openai",
                top_k=request.top_k * 2 if request.rerank else request.top_k,
            )

            results = [
                SearchResult(
                    chunk_id=r["chunk_id"],
                    paper_id=r["paper_id"],
                    content=r["content"],
                    section_title=r.get("section_title"),
                    score=r["similarity"],
                    dense_score=r["similarity"],
                )
                for r in raw_results
            ]

        elif request.mode == SearchMode.colbert:
            # ColBERT search (placeholder - requires special handling)
            raise HTTPException(
                status_code=501,
                detail="ColBERT search not yet implemented in Qdrant API"
            )

        # Apply reranking if requested
        reranked = False
        if request.rerank and results:
            try:
                reranker = get_reranker()
                # Convert to format expected by reranker
                chunks_to_rerank = [
                    {"content": r.content, "chunk_id": r.chunk_id}
                    for r in results
                ]
                reranked_results = reranker.rerank(
                    request.query,
                    chunks_to_rerank,
                    top_k=request.top_k,
                )

                # Rebuild results with rerank scores
                reranked_map = {r["chunk_id"]: r["rerank_score"] for r in reranked_results}
                results = sorted(
                    [r for r in results if r.chunk_id in reranked_map],
                    key=lambda x: reranked_map[x.chunk_id],
                    reverse=True,
                )[:request.top_k]

                # Update scores
                for r in results:
                    r.score = reranked_map[r.chunk_id]

                reranked = True
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        # Trim to top_k if not reranked
        if not reranked:
            results = results[:request.top_k]

        elapsed_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            query=request.query,
            mode=request.mode.value,
            results=results,
            total=len(results),
            search_time_ms=round(elapsed_ms, 1),
            reranked=reranked,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/stats")
async def search_stats():
    """Get search statistics and collection info."""
    try:
        qdrant = get_qdrant_client()
        info = qdrant.get_collection_info()

        return {
            "collection": info,
            "available_modes": [m.value for m in SearchMode],
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
