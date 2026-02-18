"""
Tests for embedding module models.
"""

import pytest

from src.embedding.models import (
    Chunk,
    ChunkType,
    SparseVector,
    EmbeddedChunk,
    ChunkingConfig,
    EmbeddingConfig,
)


class TestChunk:
    """Test Chunk model."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            chunk_id="paper1_chunk_0",
            paper_id="2501.12345v1",
            content="This is the chunk content.",
            chunk_index=0,
        )

        assert chunk.chunk_id == "paper1_chunk_0"
        assert chunk.paper_id == "2501.12345v1"
        assert chunk.chunk_type == ChunkType.TEXT

    def test_chunk_word_count(self):
        """Test chunk word count property."""
        chunk = Chunk(
            chunk_id="test_chunk",
            paper_id="test_paper",
            content="One two three four five",
            chunk_index=0,
        )

        assert chunk.word_count == 5

    def test_chunk_to_db_dict(self):
        """Test chunk serialization."""
        chunk = Chunk(
            chunk_id="test_chunk",
            paper_id="test_paper",
            content="Test content",
            section_title="Introduction",
            chunk_index=0,
            token_count=10,
        )

        db_dict = chunk.to_db_dict()

        assert db_dict["chunk_id"] == "test_chunk"
        assert db_dict["section_title"] == "Introduction"
        assert db_dict["metadata"]["token_count"] == 10


class TestSparseVector:
    """Test SparseVector model."""

    def test_sparse_vector_from_dict(self):
        """Test creating sparse vector from dict."""
        token_weights = {1: 0.5, 2: 0.8, 3: 0.3, 4: 0.9}
        sv = SparseVector.from_dict(token_weights, top_k=3)

        assert sv.nnz == 3
        assert 4 in sv.indices  # Highest weight should be included
        assert 2 in sv.indices

    def test_sparse_vector_to_dict(self):
        """Test converting sparse vector to dict."""
        sv = SparseVector(indices=[1, 2, 3], values=[0.5, 0.8, 0.3])
        d = sv.to_dict()

        assert d[1] == 0.5
        assert d[2] == 0.8
        assert d[3] == 0.3

    def test_sparse_vector_to_jsonb(self):
        """Test JSONB format conversion."""
        sv = SparseVector(indices=[1, 2], values=[0.5, 0.8])
        jsonb = sv.to_jsonb()

        assert jsonb["1"] == 0.5
        assert jsonb["2"] == 0.8
        assert isinstance(list(jsonb.keys())[0], str)

    def test_sparse_vector_from_jsonb(self):
        """Test loading from JSONB format."""
        jsonb = {"1": 0.5, "2": 0.8}
        sv = SparseVector.from_jsonb(jsonb)

        assert 1 in sv.indices
        assert 2 in sv.indices

    def test_sparse_vector_empty(self):
        """Test empty sparse vector."""
        sv = SparseVector.from_dict({}, top_k=10)
        assert sv.nnz == 0


class TestChunkingConfig:
    """Test ChunkingConfig model."""

    def test_default_config(self):
        """Test default chunking configuration."""
        config = ChunkingConfig()

        assert config.max_tokens == 512
        assert config.overlap_tokens == 50
        assert config.section_based is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkingConfig(
            max_tokens=1024,
            overlap_tokens=100,
            include_equations=True,
        )

        assert config.max_tokens == 1024
        assert config.include_equations is True


class TestEmbeddingConfig:
    """Test EmbeddingConfig model."""

    def test_default_config(self):
        """Test default embedding configuration."""
        config = EmbeddingConfig()

        assert config.use_bge is True
        assert config.bge_model == "BAAI/bge-m3"
        assert config.sparse_top_k == 128
        assert config.use_openai is False

    def test_openai_config(self):
        """Test OpenAI embedding configuration."""
        config = EmbeddingConfig(
            use_openai=True,
            openai_model="text-embedding-3-large",
        )

        assert config.use_openai is True
        assert config.openai_dimensions == 3072
