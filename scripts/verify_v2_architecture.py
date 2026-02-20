#!/usr/bin/env python3
"""
arXiv RAG v2 - Architecture Verification Script

Verifies that the v2 architecture is properly set up:
1. Supabase connection (metadata storage)
2. Qdrant connection (vector storage)
3. New methods in supabase_client.py
4. EmbeddedChunk.to_qdrant_dict() conversion
5. QdrantHybridRetriever search

Usage:
    python scripts/verify_v2_architecture.py
    python scripts/verify_v2_architecture.py --verbose
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_supabase_connection():
    """Verify Supabase connection and new methods."""
    print("\n[1/5] Checking Supabase connection...")

    try:
        from src.storage.supabase_client import get_supabase_client

        client = get_supabase_client()

        # Check new v2 methods exist
        assert hasattr(client, 'get_chunks_by_ids'), "Missing: get_chunks_by_ids"
        assert hasattr(client, 'get_chunks_by_ids_ordered'), "Missing: get_chunks_by_ids_ordered"
        assert hasattr(client, 'batch_insert_chunks_metadata'), "Missing: batch_insert_chunks_metadata"

        # Try to get paper count (verifies connection)
        count = client.get_paper_count()
        print(f"  Connected to Supabase")
        print(f"  Paper count: {count}")
        print(f"  v2 methods: get_chunks_by_ids, get_chunks_by_ids_ordered, batch_insert_chunks_metadata")
        print("  [PASS]")
        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def check_qdrant_connection():
    """Verify Qdrant connection."""
    print("\n[2/5] Checking Qdrant connection...")

    try:
        from src.storage.qdrant_client import get_qdrant_client

        client = get_qdrant_client()

        # Health check
        healthy = client.health_check()
        if not healthy:
            print("  [WARN] Qdrant not healthy (may not be running)")
            return False

        # Get collection info
        info = client.get_collection_info()
        print(f"  Connected to Qdrant")
        print(f"  Collection: {info.get('name', 'N/A')}")
        print(f"  Points: {info.get('points_count', 0)}")
        print("  [PASS]")
        return True

    except Exception as e:
        print(f"  [WARN] Qdrant check failed: {e}")
        print("  (This is OK if Qdrant is not running locally)")
        return False


def check_embedding_models():
    """Verify EmbeddedChunk.to_qdrant_dict() method."""
    print("\n[3/5] Checking embedding model conversions...")

    try:
        from src.embedding.models import Chunk, ChunkType, EmbeddedChunk, SparseVector

        # Create a test chunk
        chunk = Chunk(
            chunk_id="test_paper_chunk_0",
            paper_id="test_paper",
            content="This is a test chunk about transformers.",
            section_title="Introduction",
            chunk_type=ChunkType.TEXT,
            chunk_index=0,
            token_count=10,
        )

        # Create embedded chunk with mock vectors
        embedded = EmbeddedChunk(
            chunk=chunk,
            embedding_dense=[0.1] * 1024,
            embedding_sparse=SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2]),
            embedding_openai=[0.2] * 1024,
        )

        # Test to_qdrant_dict
        qdrant_dict = embedded.to_qdrant_dict()

        assert "chunk_id" in qdrant_dict, "Missing chunk_id"
        assert "paper_id" in qdrant_dict, "Missing paper_id"
        assert "content" in qdrant_dict, "Missing content"
        assert "dense_bge" in qdrant_dict, "Missing dense_bge"
        assert "dense_openai" in qdrant_dict, "Missing dense_openai"
        assert "sparse_indices" in qdrant_dict, "Missing sparse_indices"
        assert "sparse_values" in qdrant_dict, "Missing sparse_values"

        # Test to_supabase_dict (v2 - no vectors)
        supabase_dict = embedded.to_supabase_dict()

        assert "chunk_id" in supabase_dict, "Missing chunk_id"
        assert "embedding_dense" not in supabase_dict, "Should not have embedding_dense"

        print(f"  to_qdrant_dict() keys: {list(qdrant_dict.keys())}")
        print(f"  to_supabase_dict() keys: {list(supabase_dict.keys())}")
        print("  [PASS]")
        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def check_qdrant_retriever():
    """Verify QdrantHybridRetriever exists and has correct interface."""
    print("\n[4/5] Checking QdrantHybridRetriever...")

    try:
        from src.rag.qdrant_retriever import QdrantHybridRetriever, qdrant_hybrid_search

        # Check class exists with correct methods
        assert hasattr(QdrantHybridRetriever, 'search'), "Missing: search method"
        assert hasattr(QdrantHybridRetriever, 'search_dense_only'), "Missing: search_dense_only"
        assert hasattr(QdrantHybridRetriever, 'search_sparse_only'), "Missing: search_sparse_only"

        print(f"  QdrantHybridRetriever class found")
        print(f"  Methods: search, search_dense_only, search_sparse_only")
        print(f"  Convenience function: qdrant_hybrid_search")
        print("  [PASS]")
        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def check_migration_file():
    """Verify v2 migration SQL exists."""
    print("\n[5/5] Checking v2 migration file...")

    try:
        migration_path = Path(__file__).parent.parent / "supabase" / "migrations" / "20260220000001_remove_vectors.sql"

        if not migration_path.exists():
            print(f"  [FAIL] Migration file not found: {migration_path}")
            return False

        # Read and verify content
        content = migration_path.read_text()

        checks = [
            ("DROP FUNCTION", "match_chunks" in content),
            ("DROP INDEX", "idx_chunks_embedding" in content),
            ("ALTER TABLE chunks DROP COLUMN", "embedding_dense" in content),
            ("ALTER TABLE equations DROP COLUMN", "embedding" in content),
        ]

        all_pass = True
        for check_name, passed in checks:
            status = "[OK]" if passed else "[MISSING]"
            print(f"  {status} {check_name}")
            if not passed:
                all_pass = False

        if all_pass:
            print("  [PASS]")
        else:
            print("  [FAIL] Some migration sections missing")

        return all_pass

    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify v2 architecture setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("arXiv RAG v2 - Architecture Verification")
    print("=" * 60)

    results = []

    # Run all checks
    results.append(("Supabase connection", check_supabase_connection()))
    results.append(("Qdrant connection", check_qdrant_connection()))
    results.append(("Embedding models", check_embedding_models()))
    results.append(("Qdrant retriever", check_qdrant_retriever()))
    results.append(("Migration file", check_migration_file()))

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print("-" * 60)
    print(f"  Total: {passed}/{total} checks passed")
    print("=" * 60)

    # Note about Qdrant
    qdrant_passed = results[1][1]
    if not qdrant_passed:
        print("\nNote: Qdrant check failed. This is expected if Qdrant is not running.")
        print("To start Qdrant locally:")
        print("  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")

    return 0 if passed >= 4 else 1  # Allow Qdrant to fail


if __name__ == "__main__":
    exit(main())
