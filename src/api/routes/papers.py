"""
arXiv RAG v1 - Papers API Routes

Paper metadata endpoints using Supabase.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...storage.supabase_client import get_supabase_client
from ...utils.logging import get_logger

logger = get_logger("api.papers")

router = APIRouter()


class PaperSummary(BaseModel):
    """Paper summary for list view."""
    arxiv_id: str
    title: str
    authors: list[str]
    published_date: Optional[str] = None
    categories: list[str] = []
    citation_count: Optional[int] = None
    parse_status: Optional[str] = None


class PaperDetail(BaseModel):
    """Full paper details."""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: Optional[str] = None
    updated_date: Optional[str] = None
    categories: list[str] = []
    citation_count: Optional[int] = None
    pdf_url: Optional[str] = None
    parse_status: Optional[str] = None
    parse_method: Optional[str] = None
    chunk_count: Optional[int] = None


class PapersListResponse(BaseModel):
    """Papers list response."""
    papers: list[PaperSummary]
    total: int
    page: int
    page_size: int


@router.get("/papers", response_model=PapersListResponse)
async def list_papers(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    status: Optional[str] = Query(default=None, description="Filter by parse status"),
    sort_by: str = Query(default="citation_count", description="Sort field"),
    order: str = Query(default="desc", description="Sort order (asc/desc)"),
):
    """
    List papers with pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of results per page
        status: Filter by parse status (pending, parsed, embedded, failed)
        sort_by: Sort field (citation_count, published_date, title)
        order: Sort order

    Returns:
        Paginated list of papers
    """
    try:
        supabase = get_supabase_client()
        offset = (page - 1) * page_size

        # Build query
        query = supabase.client.table("papers").select(
            "arxiv_id, title, authors, published_date, categories, citation_count, parse_status",
            count="exact"
        )

        # Apply status filter
        if status:
            query = query.eq("parse_status", status)

        # Apply sorting
        desc = order.lower() == "desc"
        query = query.order(sort_by, desc=desc)

        # Apply pagination
        query = query.range(offset, offset + page_size - 1)

        result = query.execute()

        papers = [
            PaperSummary(
                arxiv_id=p["arxiv_id"],
                title=p["title"],
                authors=p.get("authors", []),
                published_date=p.get("published_date"),
                categories=p.get("categories", []),
                citation_count=p.get("citation_count"),
                parse_status=p.get("parse_status"),
            )
            for p in result.data
        ]

        return PapersListResponse(
            papers=papers,
            total=result.count or len(papers),
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Failed to list papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{arxiv_id}", response_model=PaperDetail)
async def get_paper(arxiv_id: str):
    """
    Get paper details by arXiv ID.

    Args:
        arxiv_id: arXiv paper identifier (e.g., "2401.12345")

    Returns:
        Full paper details including abstract
    """
    try:
        supabase = get_supabase_client()

        # Get paper
        paper = supabase.get_paper(arxiv_id)

        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {arxiv_id}")

        # Get chunk count
        chunks = supabase.get_chunks_by_paper(arxiv_id)
        chunk_count = len(chunks)

        return PaperDetail(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            authors=paper.get("authors", []),
            abstract=paper.get("abstract", ""),
            published_date=paper.get("published_date"),
            updated_date=paper.get("updated_date"),
            categories=paper.get("categories", []),
            citation_count=paper.get("citation_count"),
            pdf_url=paper.get("pdf_url"),
            parse_status=paper.get("parse_status"),
            parse_method=paper.get("parse_method"),
            chunk_count=chunk_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get paper {arxiv_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{arxiv_id}/chunks")
async def get_paper_chunks(
    arxiv_id: str,
    include_embeddings: bool = Query(default=False, description="Include embedding vectors"),
):
    """
    Get all chunks for a paper.

    Args:
        arxiv_id: arXiv paper identifier
        include_embeddings: Whether to include embedding vectors

    Returns:
        List of chunks with content and metadata
    """
    try:
        supabase = get_supabase_client()

        # Verify paper exists
        paper = supabase.get_paper(arxiv_id)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {arxiv_id}")

        # Get chunks
        chunks = supabase.get_chunks_by_paper(arxiv_id)

        # Format response
        response_chunks = []
        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "section_title": chunk.get("section_title"),
                "metadata": chunk.get("metadata", {}),
            }

            if include_embeddings:
                chunk_data["has_dense_embedding"] = chunk.get("embedding_dense") is not None
                chunk_data["has_sparse_embedding"] = chunk.get("embedding_sparse") is not None
                chunk_data["has_colbert_embedding"] = chunk.get("embedding_colbert") is not None

            response_chunks.append(chunk_data)

        return {
            "arxiv_id": arxiv_id,
            "title": paper["title"],
            "chunks": response_chunks,
            "total": len(response_chunks),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunks for {arxiv_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get collection statistics."""
    try:
        supabase = get_supabase_client()
        stats = supabase.get_collection_stats()

        return {
            "papers": stats,
            "chunks": {
                "total": supabase.get_chunk_count(),
            }
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
