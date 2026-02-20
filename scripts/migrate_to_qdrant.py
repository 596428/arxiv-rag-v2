#!/usr/bin/env python3
"""
arXiv RAG v1 - Migrate Vectors to Qdrant

Migrates embedding vectors from Supabase to Qdrant for improved search performance.

Usage:
    python scripts/migrate_to_qdrant.py                    # Migrate all chunks
    python scripts/migrate_to_qdrant.py --batch-size 500   # Custom batch size
    python scripts/migrate_to_qdrant.py --paper-id 2401.12345  # Single paper
    python scripts/migrate_to_qdrant.py --dry-run          # Preview only
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.supabase_client import get_supabase_client
from src.storage.qdrant_client import get_qdrant_client, COLLECTION_NAME
from src.utils.logging import get_logger

logger = get_logger("migrate_qdrant")


def migrate_chunks(
    batch_size: int = 100,
    paper_id: str = None,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict:
    """
    Migrate chunks from Supabase to Qdrant.

    Args:
        batch_size: Number of chunks per batch
        paper_id: Optional specific paper to migrate
        dry_run: Preview without writing
        skip_existing: Skip chunks already in Qdrant

    Returns:
        Migration statistics
    """
    stats = {
        "total_chunks": 0,
        "migrated": 0,
        "skipped": 0,
        "failed": 0,
        "elapsed_seconds": 0,
    }

    start_time = time.time()

    # Initialize clients
    supabase = get_supabase_client()
    qdrant = get_qdrant_client()

    # Ensure collection exists
    if not dry_run:
        qdrant.create_collection(recreate=False)

    # Get chunks from Supabase
    logger.info("Fetching chunks from Supabase...")

    if paper_id:
        # Single paper
        chunks = supabase.get_chunks_by_paper(paper_id)
        logger.info(f"Found {len(chunks)} chunks for paper {paper_id}")
    else:
        # All chunks - paginated fetch
        chunks = []
        offset = 0
        page_size = 1000

        while True:
            try:
                result = supabase.client.table("chunks").select(
                    "chunk_id, paper_id, content, section_title, metadata, "
                    "embedding_dense, embedding_sparse, embedding_openai, embedding_colbert"
                ).range(offset, offset + page_size - 1).execute()

                if not result.data:
                    break

                chunks.extend(result.data)
                offset += page_size

                logger.info(f"Fetched {len(chunks)} chunks...")

                if len(result.data) < page_size:
                    break

            except Exception as e:
                logger.error(f"Failed to fetch chunks at offset {offset}: {e}")
                break

        logger.info(f"Total chunks to migrate: {len(chunks)}")

    stats["total_chunks"] = len(chunks)

    if dry_run:
        logger.info("DRY RUN - No data will be written")
        for chunk in chunks[:5]:
            logger.info(f"  Would migrate: {chunk['chunk_id']}")
        if len(chunks) > 5:
            logger.info(f"  ... and {len(chunks) - 5} more")
        return stats

    # Process in batches
    batch = []

    for i, chunk in enumerate(chunks):
        try:
            # Prepare chunk data for Qdrant
            chunk_data = {
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "content": chunk["content"],
                "section_title": chunk.get("section_title"),
                "metadata": chunk.get("metadata", {}),
            }

            # Add dense BGE embedding
            if chunk.get("embedding_dense"):
                embedding = chunk["embedding_dense"]
                # Handle different formats (list or string)
                if isinstance(embedding, str):
                    import json
                    embedding = json.loads(embedding)
                chunk_data["dense_bge"] = embedding

            # Add OpenAI embedding
            if chunk.get("embedding_openai"):
                embedding = chunk["embedding_openai"]
                if isinstance(embedding, str):
                    import json
                    embedding = json.loads(embedding)
                chunk_data["dense_openai"] = embedding

            # Add sparse embedding
            if chunk.get("embedding_sparse"):
                sparse = chunk["embedding_sparse"]
                if isinstance(sparse, str):
                    import json
                    sparse = json.loads(sparse)

                # Extract indices and values
                if isinstance(sparse, dict):
                    if "indices" in sparse and "values" in sparse:
                        chunk_data["sparse_indices"] = sparse["indices"]
                        chunk_data["sparse_values"] = sparse["values"]
                    else:
                        # Token dict format: {token_id: weight}
                        indices = [int(k) for k in sparse.keys()]
                        values = list(sparse.values())
                        chunk_data["sparse_indices"] = indices
                        chunk_data["sparse_values"] = values

            # Add ColBERT embedding
            if chunk.get("embedding_colbert"):
                colbert = chunk["embedding_colbert"]
                if isinstance(colbert, str):
                    import json
                    colbert = json.loads(colbert)

                if isinstance(colbert, dict) and "token_embeddings" in colbert:
                    chunk_data["colbert_tokens"] = colbert["token_embeddings"]
                elif isinstance(colbert, list):
                    chunk_data["colbert_tokens"] = colbert

            batch.append(chunk_data)

            # Process batch when full
            if len(batch) >= batch_size:
                migrated = qdrant.batch_upsert_chunks(batch)
                stats["migrated"] += migrated
                stats["failed"] += len(batch) - migrated
                batch = []

                # Progress logging
                progress = (i + 1) / len(chunks) * 100
                logger.info(f"Progress: {i + 1}/{len(chunks)} ({progress:.1f}%)")

        except Exception as e:
            logger.warning(f"Failed to prepare chunk {chunk.get('chunk_id')}: {e}")
            stats["failed"] += 1

    # Process remaining batch
    if batch:
        migrated = qdrant.batch_upsert_chunks(batch)
        stats["migrated"] += migrated
        stats["failed"] += len(batch) - migrated

    stats["elapsed_seconds"] = round(time.time() - start_time, 1)

    return stats


def verify_migration():
    """Verify migration by comparing counts and sampling."""
    supabase = get_supabase_client()
    qdrant = get_qdrant_client()

    # Get counts
    supabase_count = supabase.get_chunk_count()
    qdrant_info = qdrant.get_collection_info()
    qdrant_count = qdrant_info.get("points_count", 0)

    logger.info(f"Supabase chunks: {supabase_count}")
    logger.info(f"Qdrant points: {qdrant_count}")

    if qdrant_count >= supabase_count:
        logger.info("Migration verification PASSED")
        return True
    else:
        logger.warning(f"Migration incomplete: {supabase_count - qdrant_count} chunks missing")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate vectors from Supabase to Qdrant"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100,
        help="Batch size for upserts (default: 100)"
    )
    parser.add_argument(
        "--paper-id", "-p",
        type=str,
        help="Migrate specific paper only"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview without writing"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration only (no migration)"
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Recreate collection (WARNING: deletes existing data)"
    )

    args = parser.parse_args()

    if args.verify:
        success = verify_migration()
        return 0 if success else 1

    if args.recreate_collection:
        logger.warning("Recreating Qdrant collection - existing data will be deleted!")
        qdrant = get_qdrant_client()
        qdrant.create_collection(recreate=True)
        logger.info("Collection recreated")

    # Run migration
    logger.info("Starting migration to Qdrant...")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Paper ID: {args.paper_id or 'all'}")
    logger.info(f"  Dry run: {args.dry_run}")

    stats = migrate_chunks(
        batch_size=args.batch_size,
        paper_id=args.paper_id,
        dry_run=args.dry_run,
    )

    # Print results
    print("\n" + "=" * 50)
    print("Migration Results")
    print("=" * 50)
    print(f"Total chunks:     {stats['total_chunks']}")
    print(f"Migrated:         {stats['migrated']}")
    print(f"Failed:           {stats['failed']}")
    print(f"Elapsed time:     {stats['elapsed_seconds']}s")

    if stats['total_chunks'] > 0:
        rate = stats['migrated'] / stats['elapsed_seconds'] if stats['elapsed_seconds'] > 0 else 0
        print(f"Rate:             {rate:.1f} chunks/sec")

    print("=" * 50)

    # Verify if not dry run
    if not args.dry_run and stats['migrated'] > 0:
        print("\nVerifying migration...")
        verify_migration()

    return 0


if __name__ == "__main__":
    exit(main())
