"""
arXiv RAG v1 - BGE-M3 Embedder

Dense + Sparse vector generation using BGE-M3 model.
"""

import time
from typing import Optional

from ..utils.logging import get_logger
from .models import (
    Chunk,
    EmbeddedChunk,
    EmbeddingConfig,
    EmbeddingStats,
    SparseVector,
)

logger = get_logger("bge_embedder")


class BGEEmbedder:
    """
    BGE-M3 embedder for dense + sparse hybrid retrieval.

    Features:
    - Dense vectors (1024 dims) for semantic similarity
    - Sparse vectors (top-128 tokens) for lexical matching
    - GPU acceleration with FP16
    - Batch processing
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self.stats = EmbeddingStats()
        self._device = self._get_device()

    def _get_device(self) -> str:
        """Determine device to use."""
        try:
            import torch
            if self.config.device == "cuda" and torch.cuda.is_available():
                return "cuda"
            elif self.config.device == "mps" and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def model(self):
        """Lazy load BGE-M3 model."""
        if self._model is None:
            logger.info(f"Loading BGE-M3 model on {self._device}...")
            start = time.time()

            try:
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(
                    self.config.bge_model,
                    use_fp16=self.config.bge_use_fp16 and self._device == "cuda",
                    device=self._device,
                )
                logger.info(f"BGE-M3 loaded in {time.time() - start:.1f}s")
            except ImportError:
                raise ImportError(
                    "FlagEmbedding not installed. Run: pip install FlagEmbedding"
                )

        return self._model

    def embed_texts(
        self,
        texts: list[str],
        return_sparse: bool = True,
    ) -> tuple[list[list[float]], Optional[list[SparseVector]]]:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed
            return_sparse: Whether to return sparse vectors

        Returns:
            Tuple of (dense_vectors, sparse_vectors or None)
        """
        if not texts:
            return [], [] if return_sparse else None

        start = time.time()

        # Encode with BGE-M3
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,  # Not using ColBERT for V1
        )

        # Extract dense vectors
        dense_vectors = output["dense_vecs"].tolist()

        # Extract and filter sparse vectors
        sparse_vectors = None
        if return_sparse and "lexical_weights" in output:
            sparse_vectors = []
            for weights in output["lexical_weights"]:
                # weights is dict of {token_id: weight}
                sv = SparseVector.from_dict(weights, top_k=self.config.sparse_top_k)
                sparse_vectors.append(sv)

        elapsed = time.time() - start
        self.stats.total_bge_time += elapsed
        self.stats.bge_embedded += len(texts)

        logger.debug(f"Embedded {len(texts)} texts in {elapsed:.2f}s")
        return dense_vectors, sparse_vectors

    def embed_chunks(
        self,
        chunks: list[Chunk],
        return_sparse: bool = True,
    ) -> list[EmbeddedChunk]:
        """
        Embed chunks in batches.

        Args:
            chunks: List of chunks to embed
            return_sparse: Whether to generate sparse vectors

        Returns:
            List of embedded chunks
        """
        if not chunks:
            return []

        embedded_chunks = []
        batch_size = self.config.bge_batch_size

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]

            try:
                dense_vectors, sparse_vectors = self.embed_texts(texts, return_sparse)

                for j, chunk in enumerate(batch):
                    ec = EmbeddedChunk(
                        chunk=chunk,
                        embedding_dense=dense_vectors[j],
                        embedding_sparse=sparse_vectors[j] if sparse_vectors else None,
                        model_bge=self.config.bge_model,
                    )
                    embedded_chunks.append(ec)

            except Exception as e:
                logger.error(f"Failed to embed batch {i // batch_size}: {e}")
                self.stats.bge_failed += len(batch)

                # Create chunks without embeddings
                for chunk in batch:
                    ec = EmbeddedChunk(
                        chunk=chunk,
                        embedding_dense=None,
                        embedding_sparse=None,
                        model_bge=self.config.bge_model,
                    )
                    embedded_chunks.append(ec)

            # Progress logging
            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(
                    f"BGE embedding progress: {min(i + batch_size, len(chunks))}/{len(chunks)}"
                )

        self.stats.total_chunks = len(chunks)
        return embedded_chunks

    def embed_single(self, text: str) -> tuple[list[float], Optional[SparseVector]]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Tuple of (dense_vector, sparse_vector)
        """
        dense, sparse = self.embed_texts([text], return_sparse=True)
        return dense[0], sparse[0] if sparse else None

    def get_embedding_dim(self) -> int:
        """Get dense embedding dimension (1024 for BGE-M3)."""
        return 1024

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self._device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared")
            except ImportError:
                pass

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self.clear_cache()
            logger.info("BGE-M3 model unloaded")


# Convenience function
def embed_chunks_bge(
    chunks: list[Chunk],
    config: EmbeddingConfig = None,
) -> tuple[list[EmbeddedChunk], EmbeddingStats]:
    """
    Embed chunks using BGE-M3.

    Args:
        chunks: Chunks to embed
        config: Embedding configuration

    Returns:
        Tuple of (embedded chunks, statistics)
    """
    embedder = BGEEmbedder(config)
    embedded = embedder.embed_chunks(chunks)
    stats = embedder.stats
    embedder.unload()
    return embedded, stats
