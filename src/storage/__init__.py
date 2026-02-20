"""
arXiv RAG v1 - Storage Module

Database and file storage operations.
- SupabaseClient: Raw data, metadata (papers, equations, figures)
- QdrantVectorClient: Vector embeddings (dense, sparse, ColBERT)
"""

from .supabase_client import (
    SupabaseClient,
    SupabaseError,
    get_supabase_client,
)
from .qdrant_client import (
    QdrantVectorClient,
    QdrantConfig,
    get_qdrant_client,
    COLLECTION_NAME,
)

__all__ = [
    # Supabase (raw data)
    "SupabaseClient",
    "SupabaseError",
    "get_supabase_client",
    # Qdrant (vectors)
    "QdrantVectorClient",
    "QdrantConfig",
    "get_qdrant_client",
    "COLLECTION_NAME",
]
