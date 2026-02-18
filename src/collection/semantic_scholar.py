"""
arXiv RAG v1 - Semantic Scholar API Client

Citation count and paper metrics lookup from Semantic Scholar.
Uses batch API for efficient bulk lookups.
"""

import asyncio
import time
from typing import Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..utils.config import settings
from ..utils.logging import get_logger
from .models import Paper

logger = get_logger("semantic_scholar")

# Semantic Scholar API endpoint
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

# Rate limits with API key: 1 request per second
# Batch API can process up to 500 papers per request
BATCH_SIZE = 500  # Max papers per batch request
REQUEST_INTERVAL = 1.5  # Conservative interval between requests (seconds)


class SemanticScholarError(Exception):
    """Semantic Scholar API error."""
    pass


class RateLimitError(SemanticScholarError):
    """Rate limit exceeded."""
    pass


class SemanticScholarClient:
    """
    Semantic Scholar API client for citation counts.

    Features:
    - Batch API for efficient bulk lookups (up to 500 papers per request)
    - arXiv ID to Semantic Scholar paper mapping
    - Citation count retrieval
    - Exponential backoff on rate limit errors
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        batch_size: int = BATCH_SIZE,
    ):
        self.api_key = api_key or settings.semantic_scholar_api_key
        self.batch_size = min(batch_size, BATCH_SIZE)  # Max 500
        self.request_interval = REQUEST_INTERVAL

        self._last_request_time = 0.0
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(
            f"SemanticScholarClient initialized: "
            f"authenticated={bool(self.api_key)}, "
            f"batch_size={self.batch_size}, "
            f"interval={self.request_interval:.2f}s"
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["x-api-key"] = self.api_key
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=60.0,  # Longer timeout for batch requests
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _rate_limit(self) -> None:
        """Enforce rate limiting with guaranteed minimum interval."""
        now = time.time()
        elapsed = now - self._last_request_time

        wait_time = max(0, self.request_interval - elapsed)
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        self._last_request_time = time.time()

    def _strip_version(self, arxiv_id: str) -> str:
        """Remove version suffix from arXiv ID (e.g., 2312.02120v1 -> 2312.02120)."""
        import re
        return re.sub(r'v\d+$', '', arxiv_id)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True,
    )
    async def _batch_request(
        self,
        arxiv_ids: list[str],
        fields: str = "paperId,citationCount",
    ) -> list[Optional[dict]]:
        """
        Make a batch request to Semantic Scholar API.

        Args:
            arxiv_ids: List of arXiv IDs (with or without version suffix)
            fields: Comma-separated list of fields to retrieve

        Returns:
            List of paper data dicts (None for papers not found)

        Raises:
            RateLimitError: Rate limit exceeded (will retry)
            SemanticScholarError: Other API errors
        """
        await self._rate_limit()

        client = await self._get_client()
        url = f"{S2_API_BASE}/paper/batch"

        # Convert arXiv IDs to Semantic Scholar format (strip version suffix)
        paper_ids = [f"ARXIV:{self._strip_version(arxiv_id)}" for arxiv_id in arxiv_ids]

        try:
            response = await client.post(
                url,
                params={"fields": fields},
                json={"ids": paper_ids},
            )

            if response.status_code == 429:
                logger.warning("Rate limit hit, will retry with backoff")
                raise RateLimitError("Rate limit exceeded")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            raise SemanticScholarError(f"HTTP error: {e}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise SemanticScholarError(f"Request error: {e}")

    async def batch_get_citations(
        self,
        arxiv_ids: list[str],
    ) -> dict[str, int]:
        """
        Get citation counts for multiple papers using batch API.

        Args:
            arxiv_ids: List of arXiv IDs

        Returns:
            Dict mapping arxiv_id to citation count
        """
        results = {}
        total = len(arxiv_ids)

        # Process in batches
        for i in range(0, total, self.batch_size):
            batch = arxiv_ids[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size

            logger.info(f"Batch {batch_num}/{total_batches}: fetching {len(batch)} papers")

            try:
                papers = await self._batch_request(batch)

                # Match results to arxiv_ids
                for arxiv_id, paper in zip(batch, papers):
                    if paper is not None:
                        results[arxiv_id] = paper.get("citationCount", 0) or 0
                    else:
                        results[arxiv_id] = 0

            except SemanticScholarError as e:
                logger.warning(f"Batch {batch_num} failed: {e}, setting citations to 0")
                for arxiv_id in batch:
                    results[arxiv_id] = 0

        return results

    async def enrich_papers_with_citations(
        self,
        papers: list[Paper],
    ) -> list[Paper]:
        """
        Enrich a list of papers with citation counts using batch API.

        Args:
            papers: List of Paper objects

        Returns:
            Papers with updated citation_count
        """
        if not papers:
            return papers

        logger.info(f"Enriching {len(papers)} papers with citations (batch API)")

        # Get all arxiv IDs
        arxiv_ids = [p.arxiv_id for p in papers]

        # Batch fetch citations
        citations = await self.batch_get_citations(arxiv_ids)

        # Update papers
        for paper in papers:
            paper.citation_count = citations.get(paper.arxiv_id, 0)

        # Log statistics
        total_citations = sum(p.citation_count for p in papers)
        papers_with_citations = sum(1 for p in papers if p.citation_count > 0)

        logger.info(
            f"Citation enrichment complete: "
            f"{papers_with_citations}/{len(papers)} papers have citations, "
            f"total={total_citations}"
        )

        return papers

    # Legacy single-paper methods (kept for compatibility)
    async def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[dict]:
        """Get paper details from Semantic Scholar by arXiv ID."""
        results = await self._batch_request([arxiv_id])
        return results[0] if results else None

    async def get_citation_count(self, arxiv_id: str) -> int:
        """Get citation count for an arXiv paper."""
        try:
            paper = await self.get_paper_by_arxiv_id(arxiv_id)
            if paper:
                return paper.get("citationCount", 0) or 0
            return 0
        except SemanticScholarError as e:
            logger.warning(f"Failed to get citations for {arxiv_id}: {e}")
            return 0


# Singleton client
_client: Optional[SemanticScholarClient] = None


def get_semantic_scholar_client() -> SemanticScholarClient:
    """Get or create the Semantic Scholar client singleton."""
    global _client
    if _client is None:
        _client = SemanticScholarClient()
    return _client


async def close_client() -> None:
    """Close the singleton client."""
    global _client
    if _client:
        await _client.close()
        _client = None
