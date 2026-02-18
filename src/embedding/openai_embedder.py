"""
arXiv RAG v1 - OpenAI Embedder

OpenAI text-embedding-3-large for comparison baseline.
"""

import asyncio
import time
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.config import settings
from ..utils.logging import get_logger
from .models import Chunk, EmbeddedChunk, EmbeddingConfig, EmbeddingStats

logger = get_logger("openai_embedder")


class OpenAIEmbedder:
    """
    OpenAI embedder for comparison with BGE-M3.

    Uses text-embedding-3-large (3072 dims) or configurable model.
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self._client = None
        self._async_client = None
        self.stats = EmbeddingStats()

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                api_key = settings.openai_api_key
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not configured")

                self._client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                raise ImportError("openai not installed. Run: pip install openai")

        return self._client

    @property
    def async_client(self):
        """Lazy load async OpenAI client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI

                api_key = settings.openai_api_key
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not configured")

                self._async_client = AsyncOpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai not installed. Run: pip install openai")

        return self._async_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts using OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        start = time.time()

        # OpenAI API call
        response = self.client.embeddings.create(
            model=self.config.openai_model,
            input=texts,
            dimensions=self.config.openai_dimensions,
        )

        # Extract embeddings in order
        embeddings = [None] * len(texts)
        for item in response.data:
            embeddings[item.index] = item.embedding

        elapsed = time.time() - start
        self.stats.total_openai_time += elapsed
        self.stats.openai_embedded += len(texts)

        logger.debug(f"OpenAI embedded {len(texts)} texts in {elapsed:.2f}s")
        return embeddings

    def embed_chunks(
        self,
        chunks: list[Chunk],
        update_existing: bool = True,
    ) -> list[EmbeddedChunk]:
        """
        Embed chunks using OpenAI in batches.

        Args:
            chunks: List of chunks (or EmbeddedChunks) to embed
            update_existing: If True, add OpenAI embeddings to existing EmbeddedChunks

        Returns:
            List of embedded chunks
        """
        if not chunks:
            return []

        embedded_chunks = []
        batch_size = self.config.openai_batch_size

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Extract texts
            if isinstance(batch[0], EmbeddedChunk):
                texts = [ec.chunk.content for ec in batch]
            else:
                texts = [c.content for c in batch]

            try:
                embeddings = self.embed_texts(texts)

                for j, item in enumerate(batch):
                    if isinstance(item, EmbeddedChunk):
                        # Update existing
                        item.embedding_openai = embeddings[j]
                        item.model_openai = self.config.openai_model
                        embedded_chunks.append(item)
                    else:
                        # Create new
                        ec = EmbeddedChunk(
                            chunk=item,
                            embedding_openai=embeddings[j],
                            model_openai=self.config.openai_model,
                        )
                        embedded_chunks.append(ec)

            except Exception as e:
                logger.error(f"Failed to embed batch {i // batch_size}: {e}")
                self.stats.openai_failed += len(batch)

                # Pass through without OpenAI embedding
                for item in batch:
                    if isinstance(item, EmbeddedChunk):
                        embedded_chunks.append(item)
                    else:
                        ec = EmbeddedChunk(chunk=item)
                        embedded_chunks.append(ec)

            # Rate limiting pause between batches
            if i + batch_size < len(chunks):
                time.sleep(0.5)

            # Progress logging
            if (i + batch_size) % (batch_size * 5) == 0:
                logger.info(
                    f"OpenAI embedding progress: {min(i + batch_size, len(chunks))}/{len(chunks)}"
                )

        self.stats.total_chunks = len(chunks)
        return embedded_chunks

    async def embed_texts_async(self, texts: list[str]) -> list[list[float]]:
        """
        Async version of embed_texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        response = await self.async_client.embeddings.create(
            model=self.config.openai_model,
            input=texts,
            dimensions=self.config.openai_dimensions,
        )

        embeddings = [None] * len(texts)
        for item in response.data:
            embeddings[item.index] = item.embedding

        self.stats.openai_embedded += len(texts)
        return embeddings

    async def embed_chunks_async(
        self,
        chunks: list[Chunk],
        update_existing: bool = True,
    ) -> list[EmbeddedChunk]:
        """
        Async batch embedding with concurrency control.

        Args:
            chunks: Chunks to embed
            update_existing: Whether to update existing EmbeddedChunks

        Returns:
            List of embedded chunks
        """
        if not chunks:
            return []

        start = time.time()
        semaphore = asyncio.Semaphore(self.config.max_concurrent_openai)
        batch_size = self.config.openai_batch_size

        async def process_batch(batch_chunks: list) -> list[EmbeddedChunk]:
            async with semaphore:
                if isinstance(batch_chunks[0], EmbeddedChunk):
                    texts = [ec.chunk.content for ec in batch_chunks]
                else:
                    texts = [c.content for c in batch_chunks]

                try:
                    embeddings = await self.embed_texts_async(texts)
                    results = []

                    for j, item in enumerate(batch_chunks):
                        if isinstance(item, EmbeddedChunk):
                            item.embedding_openai = embeddings[j]
                            item.model_openai = self.config.openai_model
                            results.append(item)
                        else:
                            ec = EmbeddedChunk(
                                chunk=item,
                                embedding_openai=embeddings[j],
                                model_openai=self.config.openai_model,
                            )
                            results.append(ec)

                    return results

                except Exception as e:
                    logger.error(f"Async batch failed: {e}")
                    self.stats.openai_failed += len(batch_chunks)

                    # Return without embeddings
                    results = []
                    for item in batch_chunks:
                        if isinstance(item, EmbeddedChunk):
                            results.append(item)
                        else:
                            results.append(EmbeddedChunk(chunk=item))
                    return results

        # Create batches
        batches = [
            chunks[i : i + batch_size]
            for i in range(0, len(chunks), batch_size)
        ]

        # Process concurrently
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        embedded_chunks = []
        for batch_result in batch_results:
            embedded_chunks.extend(batch_result)

        elapsed = time.time() - start
        self.stats.total_openai_time = elapsed
        self.stats.total_chunks = len(chunks)

        logger.info(f"Async OpenAI embedding completed in {elapsed:.1f}s")
        return embedded_chunks

    def embed_single(self, text: str) -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.embed_texts([text])[0]

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.config.openai_dimensions


# Convenience functions
def embed_chunks_openai(
    chunks: list[Chunk],
    config: EmbeddingConfig = None,
) -> tuple[list[EmbeddedChunk], EmbeddingStats]:
    """
    Embed chunks using OpenAI.

    Args:
        chunks: Chunks to embed
        config: Embedding configuration

    Returns:
        Tuple of (embedded chunks, statistics)
    """
    embedder = OpenAIEmbedder(config)
    embedded = embedder.embed_chunks(chunks)
    return embedded, embedder.stats


async def embed_chunks_openai_async(
    chunks: list[Chunk],
    config: EmbeddingConfig = None,
) -> tuple[list[EmbeddedChunk], EmbeddingStats]:
    """
    Async embed chunks using OpenAI.

    Args:
        chunks: Chunks to embed
        config: Embedding configuration

    Returns:
        Tuple of (embedded chunks, statistics)
    """
    embedder = OpenAIEmbedder(config)
    embedded = await embedder.embed_chunks_async(chunks)
    return embedded, embedder.stats
