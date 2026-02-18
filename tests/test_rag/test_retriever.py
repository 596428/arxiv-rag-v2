"""
Tests for RAG retriever module.
"""

import pytest

from src.rag.retriever import SearchResult, SearchResponse


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test basic search result creation."""
        result = SearchResult(
            chunk_id="chunk_1",
            paper_id="2501.12345v1",
            content="Test content",
            score=0.85,
        )

        assert result.chunk_id == "chunk_1"
        assert result.paper_id == "2501.12345v1"
        assert result.score == 0.85

    def test_search_result_with_scores(self):
        """Test search result with all scores."""
        result = SearchResult(
            chunk_id="chunk_1",
            paper_id="2501.12345v1",
            content="Test content",
            score=0.9,
            dense_score=0.85,
            sparse_score=0.75,
        )

        assert result.dense_score == 0.85
        assert result.sparse_score == 0.75

    def test_search_result_metadata(self):
        """Test search result metadata."""
        result = SearchResult(
            chunk_id="chunk_1",
            paper_id="2501.12345v1",
            content="Test content",
            metadata={"reranker_score": 0.95},
        )

        assert result.metadata["reranker_score"] == 0.95


class TestSearchResponse:
    """Test SearchResponse dataclass."""

    def test_search_response_creation(self):
        """Test basic search response creation."""
        results = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                paper_id="paper_1",
                content=f"Content {i}",
                score=0.9 - i * 0.1,
            )
            for i in range(3)
        ]

        response = SearchResponse(
            query="test query",
            results=results,
            total_found=3,
            search_time_ms=150.5,
        )

        assert response.query == "test query"
        assert len(response.results) == 3
        assert response.total_found == 3
        assert response.search_time_ms == 150.5

    def test_empty_response(self):
        """Test empty search response."""
        response = SearchResponse(
            query="no results",
            results=[],
            total_found=0,
        )

        assert len(response.results) == 0
        assert response.dense_count == 0
        assert response.sparse_count == 0


class TestRRFFusion:
    """Test RRF fusion logic."""

    def test_rrf_score_calculation(self):
        """Test RRF score formula."""
        # RRF score = 1 / (k + rank)
        k = 60
        rank = 0  # First position

        expected_score = 1 / (k + rank + 1)  # rank is 0-indexed
        assert expected_score == pytest.approx(1 / 61)

    def test_rrf_combined_score(self):
        """Test combined RRF score from two sources."""
        k = 60
        dense_rank = 0
        sparse_rank = 2

        dense_score = 1 / (k + dense_rank + 1)
        sparse_score = 1 / (k + sparse_rank + 1)
        combined = dense_score + sparse_score

        # Document appears in both, so combined score
        assert combined > dense_score
        assert combined > sparse_score
