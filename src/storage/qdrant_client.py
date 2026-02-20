"""
arXiv RAG v1 - Qdrant Vector Database Client

Qdrant client for hybrid vector search with dense, sparse, and ColBERT vectors.
Replaces Supabase pgvector for improved performance on sparse and multi-vector search.
"""

import os
from dataclasses import dataclass
from typing import Optional

from qdrant_client import QdrantClient as QdrantClientLib
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    SparseVector,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    QuantizationSearchParams,
)

from ..utils.logging import get_logger
from ..utils.config import settings

logger = get_logger("qdrant")


# Collection configuration
COLLECTION_NAME = "arxiv_chunks"
DENSE_VECTOR_SIZE = 1024  # BGE-M3 and OpenAI text-embedding-3-large (with MRL)


@dataclass
class QdrantConfig:
    """Qdrant connection configuration."""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    api_key: Optional[str] = None
    prefer_grpc: bool = True
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true",
            timeout=float(os.getenv("QDRANT_TIMEOUT", "60")),
        )


class QdrantVectorClient:
    """
    Qdrant vector database client for arXiv RAG.

    Collection schema:
    - vectors:
        - dense_bge: 1024-dim BGE-M3 dense embeddings
        - dense_openai: 1024-dim OpenAI text-embedding-3-large
        - colbert: Multi-vector ColBERT token embeddings
    - sparse_vectors:
        - sparse_bge: BGE-M3 sparse/lexical weights (inverted index)
    - payload:
        - chunk_id: str
        - paper_id: str
        - content: str
        - section_title: str
        - metadata: dict
    """

    def __init__(self, config: QdrantConfig = None):
        self.config = config or QdrantConfig.from_env()
        self._client: Optional[QdrantClientLib] = None
        logger.info(f"QdrantClient initialized: {self.config.host}:{self.config.port}")

    @property
    def client(self) -> QdrantClientLib:
        """Lazy initialization of Qdrant client."""
        if self._client is None:
            self._client = QdrantClientLib(
                host=self.config.host,
                port=self.config.port,
                grpc_port=self.config.grpc_port,
                api_key=self.config.api_key,
                prefer_grpc=self.config.prefer_grpc,
                timeout=self.config.timeout,
            )
        return self._client

    def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            info = self.client.get_collections()
            logger.debug(f"Qdrant healthy: {len(info.collections)} collections")
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create the arxiv_chunks collection with hybrid vector schema.

        Args:
            recreate: If True, delete and recreate the collection

        Returns:
            True if collection was created/exists
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == COLLECTION_NAME for c in collections)

            if exists and not recreate:
                logger.info(f"Collection '{COLLECTION_NAME}' already exists")
                return True

            if exists and recreate:
                logger.warning(f"Recreating collection '{COLLECTION_NAME}'")
                self.client.delete_collection(COLLECTION_NAME)

            # Create collection with multi-vector support
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense_bge": VectorParams(
                        size=DENSE_VECTOR_SIZE,
                        distance=Distance.COSINE,
                        on_disk=True,
                    ),
                    "dense_openai": VectorParams(
                        size=DENSE_VECTOR_SIZE,
                        distance=Distance.COSINE,
                        on_disk=True,
                    ),
                },
                sparse_vectors_config={
                    "sparse_bge": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,  # Keep in memory for fast lookup
                        ),
                    ),
                },
                # Enable payload indexing for filtering
                on_disk_payload=True,
            )

            # Create payload indexes for fast filtering
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="paper_id",
                field_schema="keyword",
            )
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="chunk_id",
                field_schema="keyword",
            )

            logger.info(f"Created collection '{COLLECTION_NAME}' with hybrid vectors")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def upsert_chunk(
        self,
        chunk_id: str,
        paper_id: str,
        content: str,
        section_title: Optional[str] = None,
        dense_bge: Optional[list[float]] = None,
        dense_openai: Optional[list[float]] = None,
        sparse_indices: Optional[list[int]] = None,
        sparse_values: Optional[list[float]] = None,
        colbert_tokens: Optional[list[list[float]]] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Upsert a single chunk with its vectors.

        Args:
            chunk_id: Unique chunk identifier
            paper_id: arXiv paper ID
            content: Chunk text content
            section_title: Section title
            dense_bge: BGE-M3 dense vector (1024 dims)
            dense_openai: OpenAI dense vector (1024 dims)
            sparse_indices: Sparse vector token indices
            sparse_values: Sparse vector token weights
            colbert_tokens: ColBERT token embeddings (variable length)
            metadata: Additional metadata

        Returns:
            True if successful
        """
        try:
            # Build vectors dict
            vectors = {}
            if dense_bge:
                vectors["dense_bge"] = dense_bge
            if dense_openai:
                vectors["dense_openai"] = dense_openai

            # Build sparse vectors dict
            sparse_vectors = {}
            if sparse_indices and sparse_values:
                sparse_vectors["sparse_bge"] = SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                )

            # Build payload
            payload = {
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "content": content,
                "section_title": section_title,
                "metadata": metadata or {},
            }

            # Store ColBERT tokens in payload (for late interaction scoring)
            if colbert_tokens:
                payload["colbert_tokens"] = colbert_tokens

            # Generate point ID from chunk_id (hash to int64)
            point_id = abs(hash(chunk_id)) % (2**63)

            point = PointStruct(
                id=point_id,
                vector=vectors if vectors else None,
                payload=payload,
            )

            # Add sparse vectors if present
            if sparse_vectors:
                point.vector = point.vector or {}
                point.vector.update(sparse_vectors)

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[point],
                wait=True,
            )

            logger.debug(f"Upserted chunk: {chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert chunk {chunk_id}: {e}")
            return False

    def batch_upsert_chunks(
        self,
        chunks: list[dict],
        batch_size: int = 100,
    ) -> int:
        """
        Batch upsert multiple chunks.

        Args:
            chunks: List of chunk dicts with keys:
                - chunk_id, paper_id, content, section_title
                - dense_bge, dense_openai (optional)
                - sparse_indices, sparse_values (optional)
                - colbert_tokens (optional)
                - metadata (optional)
            batch_size: Number of points per batch

        Returns:
            Number of chunks successfully upserted
        """
        if not chunks:
            return 0

        success_count = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            points = []

            for chunk in batch:
                try:
                    # Build vectors
                    vectors = {}
                    if chunk.get("dense_bge"):
                        vectors["dense_bge"] = chunk["dense_bge"]
                    if chunk.get("dense_openai"):
                        vectors["dense_openai"] = chunk["dense_openai"]

                    # Build payload
                    payload = {
                        "chunk_id": chunk["chunk_id"],
                        "paper_id": chunk["paper_id"],
                        "content": chunk["content"],
                        "section_title": chunk.get("section_title"),
                        "metadata": chunk.get("metadata", {}),
                    }

                    if chunk.get("colbert_tokens"):
                        payload["colbert_tokens"] = chunk["colbert_tokens"]

                    # Generate point ID
                    point_id = abs(hash(chunk["chunk_id"])) % (2**63)

                    point_data = {
                        "id": point_id,
                        "payload": payload,
                    }

                    # Add vectors
                    if vectors:
                        point_data["vector"] = vectors

                    # Add sparse vectors
                    if chunk.get("sparse_indices") and chunk.get("sparse_values"):
                        if "vector" not in point_data:
                            point_data["vector"] = {}
                        point_data["vector"]["sparse_bge"] = SparseVector(
                            indices=chunk["sparse_indices"],
                            values=chunk["sparse_values"],
                        )

                    points.append(PointStruct(**point_data))

                except Exception as e:
                    logger.warning(f"Failed to prepare chunk {chunk.get('chunk_id')}: {e}")
                    continue

            if points:
                try:
                    self.client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points,
                        wait=True,
                    )
                    success_count += len(points)
                except Exception as e:
                    logger.error(f"Batch upsert failed: {e}")

            # Progress logging
            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Upsert progress: {min(i + batch_size, len(chunks))}/{len(chunks)}")

        logger.info(f"Batch upsert complete: {success_count}/{len(chunks)} chunks")
        return success_count

    def search_dense(
        self,
        query_vector: list[float],
        vector_name: str = "dense_bge",
        top_k: int = 20,
        paper_id_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Search using dense vectors.

        Args:
            query_vector: Query embedding (1024 dims)
            vector_name: Which dense vector to search (dense_bge or dense_openai)
            top_k: Number of results
            paper_id_filter: Optional filter by paper ID

        Returns:
            List of results with chunk_id, paper_id, content, score
        """
        try:
            # Build filter if needed
            query_filter = None
            if paper_id_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="paper_id",
                            match=MatchValue(value=paper_id_filter),
                        )
                    ]
                )

            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=(vector_name, query_vector),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

            return [
                {
                    "chunk_id": r.payload.get("chunk_id"),
                    "paper_id": r.payload.get("paper_id"),
                    "content": r.payload.get("content"),
                    "section_title": r.payload.get("section_title"),
                    "similarity": r.score,
                    "metadata": r.payload.get("metadata", {}),
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    def search_sparse(
        self,
        query_indices: list[int],
        query_values: list[float],
        top_k: int = 20,
    ) -> list[dict]:
        """
        Search using sparse vectors (BM25-style lexical matching).

        Args:
            query_indices: Token indices
            query_values: Token weights
            top_k: Number of results

        Returns:
            List of results with chunk_id, paper_id, content, score
        """
        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=models.NamedSparseVector(
                    name="sparse_bge",
                    vector=SparseVector(
                        indices=query_indices,
                        values=query_values,
                    ),
                ),
                limit=top_k,
                with_payload=True,
            )

            return [
                {
                    "chunk_id": r.payload.get("chunk_id"),
                    "paper_id": r.payload.get("paper_id"),
                    "content": r.payload.get("content"),
                    "section_title": r.payload.get("section_title"),
                    "score": r.score,
                    "metadata": r.payload.get("metadata", {}),
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []

    def search_hybrid(
        self,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        vector_name: str = "dense_bge",
        top_k: int = 10,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> list[dict]:
        """
        Hybrid search combining dense and sparse with RRF fusion.

        Args:
            dense_vector: Query dense embedding
            sparse_indices: Query sparse indices
            sparse_values: Query sparse values
            vector_name: Dense vector name
            top_k: Number of final results
            dense_weight: Weight for dense results in RRF
            sparse_weight: Weight for sparse results in RRF

        Returns:
            Fused results
        """
        # Fetch more candidates for fusion
        candidate_k = top_k * 3

        # Run both searches
        dense_results = self.search_dense(dense_vector, vector_name, candidate_k)
        sparse_results = self.search_sparse(sparse_indices, sparse_values, candidate_k)

        # RRF fusion
        rrf_k = 60  # Standard RRF constant
        scores = {}  # chunk_id -> (rrf_score, result)

        # Process dense results
        for rank, result in enumerate(dense_results):
            chunk_id = result["chunk_id"]
            rrf_score = dense_weight / (rrf_k + rank + 1)
            result["dense_score"] = result.pop("similarity", 0)
            scores[chunk_id] = (rrf_score, result)

        # Process sparse results
        for rank, result in enumerate(sparse_results):
            chunk_id = result["chunk_id"]
            rrf_score = sparse_weight / (rrf_k + rank + 1)

            if chunk_id in scores:
                existing_score, existing_result = scores[chunk_id]
                existing_result["sparse_score"] = result.get("score", 0)
                scores[chunk_id] = (existing_score + rrf_score, existing_result)
            else:
                result["sparse_score"] = result.pop("score", 0)
                scores[chunk_id] = (rrf_score, result)

        # Sort by combined RRF score
        sorted_results = sorted(scores.values(), key=lambda x: x[0], reverse=True)

        # Return top-k with final scores
        final_results = []
        for score, result in sorted_results[:top_k]:
            result["score"] = score
            final_results.append(result)

        return final_results

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            return {
                "name": COLLECTION_NAME,
                "points_count": info.points_count,
                "vectors_count": getattr(info, "vectors_count", info.points_count),
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
                "status": info.status.name if hasattr(info.status, "name") else str(info.status),
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def delete_by_paper_id(self, paper_id: str) -> int:
        """Delete all chunks for a paper."""
        try:
            result = self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="paper_id",
                            match=MatchValue(value=paper_id),
                        )
                    ]
                ),
                wait=True,
            )
            logger.info(f"Deleted chunks for paper: {paper_id}")
            return 1  # Qdrant doesn't return count
        except Exception as e:
            logger.error(f"Failed to delete chunks for {paper_id}: {e}")
            return 0

    def ensure_collection(self, recreate: bool = False) -> bool:
        """
        Ensure the collection exists (v2 compatibility alias).

        Args:
            recreate: If True, delete and recreate the collection

        Returns:
            True if collection is ready
        """
        return self.create_collection(recreate=recreate)

    def upsert_chunks(self, chunks: list[dict], batch_size: int = 100) -> int:
        """
        Upsert chunks (v2 compatibility alias for batch_upsert_chunks).

        Args:
            chunks: List of chunk dicts
            batch_size: Number of points per batch

        Returns:
            Number of chunks successfully upserted
        """
        return self.batch_upsert_chunks(chunks, batch_size)

    def close(self):
        """Close the client connection."""
        if self._client:
            self._client.close()
            self._client = None


# Singleton client
_client: Optional[QdrantVectorClient] = None


def get_qdrant_client() -> QdrantVectorClient:
    """Get or create the Qdrant client singleton."""
    global _client
    if _client is None:
        _client = QdrantVectorClient()
    return _client
